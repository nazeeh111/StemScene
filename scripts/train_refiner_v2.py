#!/usr/bin/env python3
"""Fixed v2 pilot on real train/selection proposals; never opens confirmation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import Tensor

from stemscene.joint_v2 import PeerRepairRefiner
from train_refiner import digest, load, records


def objective(output: Tensor, reference: Tensor, mixture: Tensor) -> tuple[Tensor, dict[str, float]]:
    energy = mixture.square().mean().clamp_min(1e-6)
    error = output-reference
    fit = error.square().mean()/energy
    retained = mixture[None]-reference
    retained_projection = ((error*retained).mean(dim=-1).square() /
                           (retained.square().mean(dim=-1)*energy).clamp_min(1e-8)).mean()
    source_power = reference.square().mean(dim=-1)
    numerical_present = source_power > 1e-12
    if numerical_present.any():
        # Numerically quiet sources cannot dominate this auxiliary constraint.
        denominator = source_power.clamp_min(0.05*energy)
        gain_error = ((output*reference).mean(dim=-1)-source_power)/denominator
        target_gain = gain_error[numerical_present].square().mean()
    else:
        target_gain = output.new_zeros(())
    k, n = output.shape
    if k > 1:
        predicted_covariance, true_covariance = output@output.T/n, reference@reference.T/n
        mask = 1-torch.eye(k, device=output.device)
        duplicate = (torch.relu(predicted_covariance.abs()-true_covariance.abs())*mask).square().sum()
        duplicate = duplicate/(k*(k-1)*energy.square())
    else:
        duplicate = output.new_zeros(())
    width = min(1024, n)
    envelope = torch.nn.functional.avg_pool1d(output.square()[:, None], width, width//2)[:, 0]
    truth_envelope = torch.nn.functional.avg_pool1d(reference.square()[:, None], width, width//2)[:, 0]
    continuity = ((torch.diff(envelope, dim=-1)-torch.diff(truth_envelope, dim=-1))/energy).square().mean()
    loss = fit+0.10*retained_projection+0.10*target_gain+0.02*duplicate+0.01*continuity
    terms = {"loss": loss, "waveform_nmse": fit, "retained_projection": retained_projection,
             "target_gain_penalty": target_gain, "excess_duplicate": duplicate, "activity_change": continuity}
    return loss, {key: float(value.detach()) for key, value in terms.items()}


def all_request_score(output: Tensor, reference: Tensor, mixture: Tensor, rate: int) -> float:
    n = len(mixture)
    first, last, fade = round(rate), round(4*rate), round(0.02*rate)
    if not 0 <= first < last <= n:
        raise ValueError("v2 pilot expects the frozen five-second recipe")
    envelope = torch.ones(n)
    local = torch.full((last-first,), 0.25)
    ramp = torch.linspace(1, 0.25, fade+1)[:-1]
    local[:fade], local[-fade:] = ramp, ramp.flip(0)
    envelope[first:last] = local
    return float(((envelope[None]-1)*(output-reference)).square().mean()/mixture.square().mean().clamp_min(1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 12 or not 0 < args.learning_rate <= 0.01 or not 1 <= args.threads <= 8:
        parser.error("v2 bounded pilot limits exceeded")
    training, selection = records(args.records, "train"), records(args.records, "selection")
    if len(training) != 30 or len(selection) != 10:
        raise ValueError("v2 pilot requires the complete fixed30/10 record set")
    source_sets = [{name for _, meta in split for name in meta["source_ids"]} for split in (training, selection)]
    if source_sets[0]&source_sets[1]:
        raise ValueError("source recording leakage")
    frozen = {meta["frozen_recipes_sha256"] for _, meta in training+selection}
    backend = {meta["backend_checkpoint_sha256"] for _, meta in training+selection}
    if len(frozen) != 1 or len(backend) != 1:
        raise ValueError("mixed recipe/backend provenance")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"architecture": PeerRepairRefiner.architecture,
                "primary_selection": "mean all-request edit NMSE; edit each query1-4s gain.25 fade.02s; average queries then recipes",
                "loss": "waveform_nmse+.10retained_projection+.10target_gain+.02excess_duplicate+.01activity_change",
                "retained_signal": "mixture-target_reference", "epochs": args.epochs,
                "learning_rate": args.learning_rate, "weight_decay": 0.0001,
                "seed": args.seed, "threads": args.threads, "confirmation_opened": False,
                "frozen_recipes_sha256": next(iter(frozen)), "backend_checkpoint_sha256": next(iter(backend))}
    (args.output/"protocol-before-training.json").write_text(json.dumps(protocol, indent=2)+"\n")
    snapshot = args.output/"source"; snapshot.mkdir()
    sources = [Path(__file__), Path(__file__).with_name("train_refiner.py"),
               Path(__file__).resolve().parents[1]/"src/stemscene/joint_v2.py",
               Path(__file__).resolve().parents[1]/"src/stemscene/joint.py"]
    source_hashes = {}
    for source in sources:
        shutil.copyfile(source, snapshot/source.name)
        source_hashes[source.name] = digest(source)
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = random.Random(args.seed)
    # Preload only authenticated training/selection files; no confirmation glob.
    train_data = [load(path) for path, _ in training]
    selection_data = [load(path) for path, _ in selection]
    model = PeerRepairRefiner()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    started = time.perf_counter()

    @torch.no_grad()
    def validate() -> dict[str, float]:
        model.eval(); losses, edits = [], []
        for x, p, r, e, rate in selection_data:
            prediction = model(x, p, e, rate)
            _, terms = objective(prediction, r, x)
            losses.append(terms["loss"])
            edits.append(all_request_score(prediction, r, x, rate))
        return {"loss": float(np.mean(losses)), "all_request_edit_nmse": float(np.mean(edits))}

    initial = validate()
    model.save(args.output/"identity.pt", provenance={**protocol, "trained": False, "selected_epoch": -1,
                                                     "initial_selection": initial})
    best, best_state, best_epoch = initial["all_request_edit_nmse"], copy.deepcopy(model.state_dict()), -1
    log = []
    for epoch in range(1, args.epochs+1):
        model.train(); order = list(range(len(train_data))); rng.shuffle(order); losses = []
        for index in order:
            x, p, r, e, rate = train_data[index]
            permutation = torch.randperm(len(p))
            p, r, e = p[permutation], r[permutation], e[permutation]
            optimizer.zero_grad(set_to_none=True)
            loss, terms = objective(model(x, p, e, rate), r, x)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite v2 training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
            losses.append(terms["loss"])
        scores = validate()
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "selection_loss": scores["loss"], "selection_all_request_edit_nmse": scores["all_request_edit_nmse"],
               "elapsed_seconds": time.perf_counter()-started}
        print(json.dumps(row), flush=True); log.append(row)
        (args.output/"training-log.json").write_text(json.dumps(log, indent=2)+"\n")
        if scores["all_request_edit_nmse"] < best:
            best, best_state, best_epoch = scores["all_request_edit_nmse"], copy.deepcopy(model.state_dict()), epoch
    last = copy.deepcopy(model.state_dict())
    provenance = {**protocol, "trained": best_epoch > 0, "selected_epoch": best_epoch,
                  "epochs_executed": args.epochs, "artifact_role": "selected", "state_epoch": best_epoch,
                  "initial_selection": initial, "selected_all_request_edit_nmse": best,
                  "selection_improved": best_epoch > 0, "source_hashes": source_hashes,
                  "training_records": [{"recipe_id": m["recipe_id"], "sha256": digest(p)} for p, m in training],
                  "selection_records": [{"recipe_id": m["recipe_id"], "sha256": digest(p)} for p, m in selection],
                  "seconds": time.perf_counter()-started, "parameters": sum(p.numel() for p in model.parameters()),
                  "note": "Selection-only controlled genuine-source experiment. No confirmation/human-quality claim."}
    model.load_state_dict(best_state); model.save(args.output/"selected.pt", provenance=provenance)
    model.load_state_dict(last); model.save(args.output/"last.pt", provenance={
        **provenance, "trained": True, "artifact_role": "last_optimized", "state_epoch": args.epochs,
        "experiment_selected_epoch": best_epoch, "selected_epoch": args.epochs,
        "selected_all_request_edit_nmse": log[-1]["selection_all_request_edit_nmse"]})
    (args.output/"report.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
