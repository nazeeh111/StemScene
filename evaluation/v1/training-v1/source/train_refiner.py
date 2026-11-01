#!/usr/bin/env python3
"""Train only the compact refiner on frozen, real-backend proposal records.

Confirmation records are never opened here. A sibling JSON file is required for
every NPZ. See docs/refinement-method.md for the data contract and limitations.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from stemscene.joint import JointRefiner


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def records(root: Path, split: str) -> list[tuple[Path, dict[str, Any]]]:
    directory = root / split
    result = []
    for path in sorted(directory.glob("*.npz")):
        if path.stat().st_size > 512_000_000:
            raise ValueError(f"oversized proposal archive: {path}")
        metadata = json.loads(path.with_suffix(".json").read_text())
        required = {"split", "recipe_id", "source_ids", "backend_checkpoint_sha256", "frozen_recipes_sha256"}
        if not required.issubset(metadata) or metadata["split"] != split:
            raise ValueError(f"incomplete or wrong split metadata: {path}")
        if not isinstance(metadata["source_ids"], list) or not metadata["source_ids"]:
            raise ValueError(f"missing original source IDs: {path}")
        if metadata.get("record_sha256") != digest(path):
            raise ValueError(f"proposal record differs from its frozen SHA-256: {path}")
        result.append((path, metadata))
    if not result:
        raise ValueError(f"no real-backend records in {directory}")
    return result


def load(path: Path) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    # This code is the training/evaluator boundary; inference never receives refs.
    with np.load(path, allow_pickle=False) as archive:
        required = {"mixture", "proposals", "references", "embeddings", "sample_rate"}
        if not required.issubset(archive.files):
            raise ValueError(f"missing arrays: {path}")
        arrays = [np.asarray(archive[key], dtype=np.float32) for key in
                  ("mixture", "proposals", "references", "embeddings")]
        rate = int(archive["sample_rate"].item())
    x, p, r, e = arrays
    if x.ndim == 2 and x.shape[-1] == 1:
        x = x[:, 0]
    if p.ndim == 3 and p.shape[-1] == 1:
        p = p[:, :, 0]
    if r.ndim == 3 and r.shape[-1] == 1:
        r = r[:, :, 0]
    if x.ndim != 1 or p.ndim != 2 or p.shape != r.shape or p.shape[1] != len(x) or e.shape != (len(p), 512):
        raise ValueError(f"invalid proposal data shape: {path}")
    if not all(np.isfinite(array).all() for array in (x, p, r, e)):
        raise ValueError(f"nonfinite training input: {path}")
    return *(torch.from_numpy(a) for a in (x, p, r, e)), rate


def objective(output: Tensor, reference: Tensor, mixture: Tensor) -> tuple[Tensor, dict[str, float]]:
    """Gain-sensitive fit, excess attribution and non-target edit damage.

    No SI-SDR-only objective: the same waveform residual is exactly the error
    introduced by completely attenuating a requested component of the original.
    """
    energy = mixture.square().mean().clamp_min(1e-6)
    fit = (output - reference).square().mean() / energy
    # Non-target projection estimates contamination in each independently edited
    # stem. Source references appear only in this loss, never forward features.
    non_target = reference.sum(dim=0, keepdim=True) - reference
    contamination = ((output - reference) * non_target).mean(dim=-1).square()
    contamination = (contamination / (non_target.square().mean(dim=-1) * energy).clamp_min(1e-8)).mean()
    # Penalize excess pairwise common waveform attribution beyond true overlap.
    k = len(output)
    if k > 1:
        corr = output @ output.T / output.shape[-1]
        truth = reference @ reference.T / reference.shape[-1]
        off_diagonal = 1 - torch.eye(k, device=output.device)
        duplicate = (torch.relu(corr.abs() - truth.abs()) * off_diagonal).square().sum() / (k * (k - 1) * energy.square())
    else:
        duplicate = output.new_zeros(())
    # Penalize excess envelope jumps without preferring constant silence.
    window = min(1024, output.shape[-1])
    envelope = torch.nn.functional.avg_pool1d(output.square()[:, None], window, window // 2)[:, 0]
    truth_envelope = torch.nn.functional.avg_pool1d(reference.square()[:, None], window, window // 2)[:, 0]
    continuity = ((torch.diff(envelope, dim=-1) - torch.diff(truth_envelope, dim=-1)) / energy).square().mean()
    loss = fit + 0.10 * contamination + 0.05 * duplicate + 0.01 * continuity
    return loss, {"loss": float(loss.detach()), "waveform_nmse": float(fit.detach()),
                  "contamination": float(contamination.detach()), "duplicate": float(duplicate.detach()),
                  "envelope_change_error": float(continuity.detach())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="directory containing train/ and selection/ only")
    parser.add_argument("output", type=Path, help="new output directory; never overwrite")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 200 or not 1 <= args.threads <= 16:
        raise ValueError("epochs/threads outside bounds")
    if not 0 < args.learning_rate <= 0.1:
        raise ValueError("learning rate outside bounds")
    training, selection = records(args.records, "train"), records(args.records, "selection")
    train_sources = {str(s) for _, m in training for s in m["source_ids"]}
    validation_sources = {str(s) for _, m in selection for s in m["source_ids"]}
    if train_sources & validation_sources:
        raise ValueError("original recording leakage between training and selection")
    hashes = {m["frozen_recipes_sha256"] for _, m in training + selection}
    backends = {m["backend_checkpoint_sha256"] for _, m in training + selection}
    if len(hashes) != 1 or len(backends) != 1:
        raise ValueError("inconsistent frozen recipe or backend provenance")
    recipe_ids = [m["recipe_id"] for _, m in training + selection]
    if len(set(recipe_ids)) != len(recipe_ids):
        raise ValueError("duplicate recipe IDs")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    model = JointRefiner()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    start = time.perf_counter()
    log = []
    best_score = float("inf")
    best_state = None
    best_epoch = -1

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        scores = []
        for path, _ in selection:
            x, p, r, e, sr = load(path)
            _, terms = objective(model(x, p, e, sr), r, x)
            scores.append(terms["loss"])
        return float(np.mean(scores))

    initial_validation = validate()
    model.save(args.output / "identity.pt", provenance={
        "trained": False, "selected_epoch": -1, "initial_selection_loss": initial_validation,
        "frozen_recipes_sha256": next(iter(hashes)), "backend_checkpoint_sha256": next(iter(backends)),
        "seed": args.seed, "confirmation_opened": False,
    })
    # Untrained identity is a legitimate candidate, but never called a trained
    # improvement. A negative result remains visible in the report.
    best_score, best_state = initial_validation, copy.deepcopy(model.state_dict())
    for epoch in range(args.epochs):
        model.train()
        order = list(training)
        rng.shuffle(order)
        terms_list = []
        for path, _ in order:
            x, p, r, e, sr = load(path)
            permutation = torch.randperm(len(p))
            p, r, e = p[permutation], r[permutation], e[permutation]
            optimizer.zero_grad(set_to_none=True)
            loss, terms = objective(model(x, p, e, sr), r, x)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at epoch {epoch}, record {path}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            terms_list.append(terms)
        score = validate()
        row = {"epoch": epoch + 1, "train_loss": float(np.mean([t["loss"] for t in terms_list])),
               "selection_loss": score, "elapsed_seconds": time.perf_counter() - start}
        log.append(row)
        print(json.dumps(row), flush=True)
        if score < best_score:
            best_score, best_state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch + 1
        (args.output / "training-log.json").write_text(json.dumps(log, indent=2) + "\n")
    assert best_state is not None
    last_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    provenance = {"seed": args.seed, "epochs_executed": args.epochs, "selected_epoch": best_epoch,
                  "learning_rate": args.learning_rate, "weight_decay": 1e-4, "threads": args.threads,
                  "trainer_sha256": digest(Path(__file__)),
                  "joint_module_sha256": digest(Path(__file__).resolve().parents[1] / "src/stemscene/joint.py"),
                  "initial_selection_loss": initial_validation, "selected_selection_loss": best_score,
                  "frozen_recipes_sha256": next(iter(hashes)), "backend_checkpoint_sha256": next(iter(backends)),
                  "training_records": [{"recipe_id": m["recipe_id"], "sha256": digest(p)} for p, m in training],
                  "selection_records": [{"recipe_id": m["recipe_id"], "sha256": digest(p)} for p, m in selection],
                  "confirmation_opened": False, "seconds": time.perf_counter() - start,
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "selection_improved": best_epoch > 0,
                  "note": "Controlled genuine-source mixtures; no human quality judgment or unseen foundation-pretraining claim."}
    model.save(args.output / "selected.pt", provenance=provenance)
    model.load_state_dict(last_state)
    model.save(args.output / "last.pt", provenance={**provenance, "selected_epoch": args.epochs})
    (args.output / "report.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
