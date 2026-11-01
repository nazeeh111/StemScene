#!/usr/bin/env python3
"""Source-reference evaluation only. This program never supplies references to inference."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any
import zipfile

import numpy as np
import torch

from stemscene.joint import JointRefiner, consistency_projection

FROZEN_SHA256 = "bd19a599b3165a0648e1ed9fef17e085e18fd4878f090e25e24945454c3d0728"
SILENCE_POWER = 1e-12
EDIT_PROTOCOL = "all-request-separate-and-simultaneous-v2"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite_mean(values: list[float | None]) -> float | None:
    selected = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(selected)) if selected else None


def sdr(reference: np.ndarray, estimate: np.ndarray) -> float | None:
    """Gain-sensitive direct waveform ratio; silent references have no SDR."""
    energy = float(np.dot(reference, reference))
    if energy / max(len(reference), 1) <= SILENCE_POWER:
        return None
    error = float(np.sum((estimate - reference) ** 2))
    return float(np.clip(10 * np.log10(energy / max(error, energy * 1e-12)), -120, 120))


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float | None:
    truth, prediction = reference - reference.mean(), estimate - estimate.mean()
    energy = float(np.dot(truth, truth))
    if energy / max(len(truth), 1) <= SILENCE_POWER:
        return None
    if np.mean(prediction**2) <= SILENCE_POWER:
        return -120.0
    target = float(np.dot(truth, prediction)) / energy * truth
    target_energy = float(np.sum(target**2))
    error = float(np.sum((prediction - target) ** 2))
    return float(np.clip(10 * np.log10(max(target_energy, energy * 1e-12) / max(error, energy * 1e-12)), -120, 120))


def edit_envelope(length: int, sample_rate: int, edit: dict[str, Any]) -> np.ndarray:
    start = round(float(edit.get("start_seconds", 1.0)) * sample_rate)
    end = round(float(edit.get("end_seconds", 4.0)) * sample_rate)
    gain = float(edit.get("gain", 0.25))
    fade = round(float(edit.get("fade_seconds", 0.02)) * sample_rate)
    if not 0 <= start < end <= length or not math.isfinite(gain) or not 0 <= gain <= 4 or fade < 0:
        raise ValueError("Invalid evaluator edit")
    fade = min(fade, (end - start) // 2)
    envelope = np.ones(length, dtype=np.float64)
    selected = np.full(end - start, gain)
    if fade:
        ramp = np.linspace(1.0, gain, fade + 1)[:-1]
        selected[:fade], selected[-fade:] = ramp, ramp[::-1]
    envelope[start:end] = selected
    return envelope


def power_envelopes(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    width = max(1, round(0.1 * sample_rate))
    frames = audio.shape[-1] // width
    if frames < 2:
        raise ValueError("At least 200 ms needed for activity metrics")
    return (audio[:, :frames * width].reshape(len(audio), frames, width) ** 2).mean(axis=-1)


def metrics(mixture: np.ndarray, estimate: np.ndarray, references: np.ndarray,
            sample_rate: int, metadata: dict[str, Any], absence_threshold: float,
            *, present_only_oracle: bool = False, all_sources: np.ndarray | None = None) -> dict[str, Any]:
    energy = max(float(np.mean(mixture**2)), SILENCE_POWER)
    present = np.array(metadata["presence_by_request"], dtype=bool)
    ref_present = np.mean(references**2, axis=-1) > SILENCE_POWER
    # Label presence is clip membership, which can differ from numerical silence.
    present_for_sdr = present & ref_present
    per_query = []
    for index, (truth, prediction) in enumerate(zip(references, estimate)):
        true_energy = float(np.dot(truth, truth))
        extracted_ratio = float(np.mean(prediction**2) / energy)
        per_query.append({"query": metadata["queries"][index], "present_label": bool(present[index]),
                          "reference_numerically_silent": not bool(ref_present[index]),
                          "sdr_db": sdr(truth, prediction) if present_for_sdr[index] else None,
                          "si_sdr_db": si_sdr(truth, prediction) if present_for_sdr[index] else None,
                          "gain_projection": float(np.dot(prediction, truth) / true_energy) if ref_present[index] else None,
                          "extracted_rms": float(np.sqrt(np.mean(prediction**2))),
                          "extracted_energy_over_mixture": extracted_ratio})
    edit = metadata.get("edit", {})
    qi = int(edit.get("request_index", 0))
    if not 0 <= qi < len(estimate):
        raise ValueError("Edit query index out of bounds")
    envelope = edit_envelope(len(mixture), sample_rate, edit)
    ideal = mixture + (envelope - 1) * references[qi]
    rendered = mixture + (envelope - 1) * estimate[qi]
    error = rendered - ideal
    retained = mixture - references[qi]
    retained_energy = float(np.dot(retained, retained))
    projection = float(np.dot(error, retained) / retained_energy) if retained_energy > SILENCE_POWER * len(mixture) else None
    prediction_covariance = estimate @ estimate.T / len(mixture)
    truth_covariance = references @ references.T / len(mixture)
    off_diagonal = ~np.eye(len(estimate), dtype=bool)
    excess = np.maximum(np.abs(prediction_covariance) - np.abs(truth_covariance), 0)
    activity = power_envelopes(estimate, sample_rate)
    true_activity = power_envelopes(references, sample_rate)
    residual = mixture - estimate.sum(axis=0)
    output: dict[str, Any] = {
        "target_sdr_db": finite_mean([query["sdr_db"] for query in per_query]),
        "target_si_sdr_db": finite_mean([query["si_sdr_db"] for query in per_query]),
        "target_gain_absolute_error": finite_mean([abs(query["gain_projection"] - 1) for query in per_query if query["gain_projection"] is not None]),
        "waveform_nmse": float(np.mean((estimate - references)**2) / energy),
        "edit_nmse": float(np.mean(error**2) / energy),
        "edit_mse": float(np.mean(error**2)),
        "retained_source_projection_nmse": projection**2 if projection is not None else None,
        "absence_energy_over_mixture": finite_mean([query["extracted_energy_over_mixture"] for query in per_query if not query["present_label"]]),
        "absence_false_extraction_rate": finite_mean([float(query["extracted_energy_over_mixture"] > absence_threshold) for query in per_query if not query["present_label"]]),
        "excess_pairwise_attribution": float(np.mean(excess[off_diagonal]**2) / energy**2) if len(estimate) > 1 else 0.0,
        "activity_envelope_nmse": float(np.mean((activity - true_activity)**2) / energy**2),
        "activity_change_nmse": float(np.mean((np.diff(activity, axis=-1) - np.diff(true_activity, axis=-1))**2) / energy**2),
        "resynthesis_max_absolute_error": float(np.max(np.abs(residual + estimate.sum(axis=0) - mixture))),
        "residual_energy_over_mixture": float(np.mean(residual**2) / energy),
        "outside_edit_max_delta": float(np.max(np.abs((rendered - mixture)[envelope == 1]), initial=0)),
        "boundary_error_nmse": None,
        "nonboundary_error_nmse": None,
        "per_query": per_query,
    }
    # Every requested layer receives the same predeclared edit in turn. Average
    # within a scene before averaging scenes; requests are not independent data.
    all_errors = (envelope - 1)[None] * (estimate - references)
    each_error = np.mean(all_errors**2, axis=1) / energy
    all_retained = mixture[None] - references
    each_retained_energy = np.sum(all_retained**2, axis=1)
    projection_error = np.sum(all_errors * all_retained, axis=1) / np.maximum(each_retained_energy, SILENCE_POWER*len(mixture))
    output["all_request_edit_nmse"] = float(np.mean(each_error))
    output["all_request_present_edit_nmse"] = float(np.mean(each_error[present])) if present.any() else None
    output["all_request_absent_edit_nmse"] = float(np.mean(each_error[~present])) if (~present).any() else None
    output["all_request_retained_projection_nmse"] = float(np.mean(projection_error**2))
    output["all_request_edit_errors"] = each_error.tolist()
    target_change = (envelope - 1)[None] * references
    target_power = np.sum(target_change**2, axis=1)
    beta = np.divide(np.sum(all_errors*target_change, axis=1), target_power,
                     out=np.zeros(len(estimate)), where=target_power>SILENCE_POWER*len(mixture))
    target_error = beta[:, None]*target_change
    orthogonal_error = all_errors-target_error
    scored = present if present_only_oracle else np.ones(len(estimate), dtype=bool)
    output["all_request_target_projection_edit_nmse"] = float(np.mean(target_error[scored]**2)/energy) if scored.any() else None
    output["all_request_orthogonal_edit_nmse"] = float(np.mean(orthogonal_error[scored]**2)/energy) if scored.any() else None
    output["source_duplicate_energy_over_mixture"] = None
    output["source_attribution_condition"] = None
    output["source_attribution_coefficients"] = None
    if all_sources is not None:
        sources = np.asarray(all_sources, dtype=np.float64)
        if sources.ndim != 2 or sources.shape[1] != len(mixture) or len(sources)>8 or not np.isfinite(sources).all():
            raise ValueError("Invalid evaluator-only source attribution basis")
        power = np.mean(sources**2, axis=1)
        active_sources = power>SILENCE_POWER
        sources, power = sources[active_sources], power[active_sources]
        if len(sources) and scored.any():
            gram = sources@sources.T
            condition = float(np.linalg.cond(gram))
            output["source_attribution_condition"] = condition if math.isfinite(condition) else None
            # Ill-conditioned / indistinguishable references do not establish
            # exclusive source identities; expose missing attribution instead.
            if math.isfinite(condition) and condition<1e8:
                inverse = np.linalg.pinv(gram, rcond=1e-8)
                coefficients = (estimate[scored]@sources.T)@inverse
                true_coefficients = (references[scored]@sources.T)@inverse
                attribution = coefficients**2
                true_attribution = true_coefficients**2
                duplicate = attribution.sum(axis=0)-attribution.max(axis=0)
                true_duplicate = true_attribution.sum(axis=0)-true_attribution.max(axis=0)
                output["source_duplicate_energy_over_mixture"] = float(np.sum(np.maximum(duplicate-true_duplicate,0)*power)/energy)
                output["source_attribution_coefficients"] = coefficients.tolist()
    # A separate operation attenuates all requested components simultaneously.
    # The residual is kept unchanged, including unrequested original audio.
    joint_edit_error = (envelope - 1) * (estimate.sum(axis=0) - references.sum(axis=0))
    output["simultaneous_edit_nmse"] = float(np.mean(joint_edit_error**2) / energy)
    boundaries = metadata.get("chunk_boundaries_samples", [])
    if boundaries:
        mask = np.zeros(len(mixture), dtype=bool)
        width = max(1, round(0.02 * sample_rate))
        for boundary in boundaries:
            if type(boundary) is not int or not 0 < boundary < len(mixture):
                raise ValueError("Invalid observed chunk boundary")
            mask[max(0, boundary-width):min(len(mixture), boundary+width)] = True
        squared = np.mean((estimate - references)**2, axis=0)
        output["boundary_error_nmse"] = float(np.mean(squared[mask]) / energy)
        output["nonboundary_error_nmse"] = float(np.mean(squared[~mask]) / energy) if (~mask).any() else None
    output["comparison_scope"] = "present-request evaluator oracle; no semantic assignment/absence detection" if present_only_oracle else "all requested queries"
    if present_only_oracle:
        # An unlabeled universal separator cannot choose an absent semantic
        # target. Zero unmatched rows are placeholders, never successful absence.
        output["absence_energy_over_mixture"] = None
        output["absence_false_extraction_rate"] = None
        output["all_request_edit_nmse"] = output["all_request_present_edit_nmse"]
        output["all_request_absent_edit_nmse"] = None
        output["all_request_retained_projection_nmse"] = float(np.mean(projection_error[present]**2)) if present.any() else None
        output["all_request_edit_errors"] = [float(error) if is_present else None for error, is_present in zip(each_error, present)]
        output["simultaneous_edit_nmse"] = None
        for query in per_query:
            if not query["present_label"]:
                query["extracted_rms"] = None
                query["extracted_energy_over_mixture"] = None
                query["not_evaluated_reason"] = "Universal separator has no semantic absence detector"
        if not present[qi]:
            for key in ("edit_nmse", "edit_mse", "retained_source_projection_nmse", "outside_edit_max_delta"):
                output[key] = None
        if present.any():
            scored, true_scored = estimate[present], references[present]
            output["waveform_nmse"] = float(np.mean((scored-true_scored)**2) / energy)
            activity = power_envelopes(scored, sample_rate)
            true_activity = power_envelopes(true_scored, sample_rate)
            output["activity_envelope_nmse"] = float(np.mean((activity-true_activity)**2) / energy**2)
            output["activity_change_nmse"] = float(np.mean((np.diff(activity, axis=-1)-np.diff(true_activity, axis=-1))**2) / energy**2)
            off_diagonal = ~np.eye(len(scored), dtype=bool)
            excess = np.maximum(np.abs(scored @ scored.T / len(mixture)) - np.abs(true_scored @ true_scored.T / len(mixture)), 0)
            output["excess_pairwise_attribution"] = float(np.mean(excess[off_diagonal]**2) / energy**2) if len(scored)>1 else 0.0
            if boundaries:
                squared = np.mean((scored-true_scored)**2, axis=0)
                output["boundary_error_nmse"] = float(np.mean(squared[mask]) / energy)
                output["nonboundary_error_nmse"] = float(np.mean(squared[~mask]) / energy) if (~mask).any() else None
        else:
            for key in ("waveform_nmse", "activity_envelope_nmse", "activity_change_nmse", "excess_pairwise_attribution", "boundary_error_nmse", "nonboundary_error_nmse"):
                output[key] = None
    return output


@torch.inference_mode()
def residual_wiener(mixture: np.ndarray, proposals: np.ndarray) -> np.ndarray:
    """Known relative-power masks with an explicit residual candidate; no learning."""
    x, p = torch.as_tensor(mixture, dtype=torch.float64), torch.as_tensor(proposals, dtype=torch.float64)
    window = torch.hann_window(1024, dtype=torch.float64)
    stft = lambda signal: torch.stft(signal, 1024, 256, window=window, center=True, pad_mode="constant", return_complex=True)
    xz, pz = stft(x), stft(p)
    powers = pz.abs().square()
    residual_power = (xz - pz.sum(dim=0)).abs().square()
    mask = powers / (powers.sum(dim=0) + residual_power).clamp_min(1e-20)
    return torch.istft(mask * xz, 1024, 256, window=window, center=True, length=len(x)).numpy()


def oracle_fuss_assignment(outputs: np.ndarray, references: np.ndarray, present: np.ndarray) -> np.ndarray:
    """One-to-one present-only evaluator matching; never semantic absence detection."""
    from scipy.optimize import linear_sum_assignment
    present_indices = np.flatnonzero(present)
    assigned = np.zeros_like(references)
    if len(present_indices) == 0:
        return assigned
    cost = np.mean((outputs[:, None, :] - references[None, present_indices, :])**2, axis=-1)
    rows, columns = linear_sum_assignment(cost)
    assigned[present_indices[columns]] = outputs[rows]
    return assigned


def load_record(path: Path, split: str, expected_hash: str, backend_hash: str) -> tuple[dict[str, Any], dict[str, np.ndarray], int]:
    if path.stat().st_size > 512_000_000:
        raise ValueError(f"Oversized proposal archive: {path}")
    with zipfile.ZipFile(path) as zipped:
        if sum(entry.file_size for entry in zipped.infolist()) > 512_000_000:
            raise ValueError("Expanded proposal arrays exceed evaluation memory bound")
    meta_path = path.with_suffix(".json")
    if meta_path.stat().st_size > 1_000_000:
        raise ValueError("Oversized metadata")
    metadata = json.loads(meta_path.read_text())
    if metadata.get("split") != split or metadata.get("frozen_recipes_sha256") != expected_hash or metadata.get("backend_checkpoint_sha256") != backend_hash:
        raise ValueError(f"Record provenance mismatch: {path}")
    if metadata.get("record_sha256") != digest(path):
        raise ValueError("Proposal archive differs from its recorded SHA-256")
    required = {"mixture", "proposals", "references", "embeddings", "sample_rate"}
    with np.load(path, allow_pickle=False) as archive:
        if not required.issubset(archive.files):
            raise ValueError("Missing evaluation arrays")
        arrays = {name: np.asarray(archive[name], dtype=np.float64) for name in archive.files if name != "sample_rate"}
        rate = int(archive["sample_rate"].item())
    x, p, truth, embeddings = (arrays[key] for key in ("mixture", "proposals", "references", "embeddings"))
    if x.ndim == 2 and x.shape[1] == 1:
        x = arrays["mixture"] = x[:, 0]
    for key in ("proposals", "references", "sequential_forward", "sequential_reverse", "fuss_all_sources"):
        if key in arrays and arrays[key].ndim == 3 and arrays[key].shape[-1] == 1:
            arrays[key] = arrays[key][:, :, 0]
    p, truth = arrays["proposals"], arrays["references"]
    if (x.ndim != 1 or p.ndim != 2 or p.shape != truth.shape or p.shape[1] != len(x)
            or not 1 <= len(p) <= 8 or not 1024 <= len(x) <= 960000 or embeddings.shape != (len(p), 512)
            or not 8000 <= rate <= 96000 or not all(np.isfinite(a).all() for a in arrays.values())):
        raise ValueError("Invalid evaluation array shape, bounds or values")
    if len(metadata.get("queries", [])) != len(p) or len(metadata.get("presence_by_request", [])) != len(p):
        raise ValueError("Missing query-aligned presence labels")
    if not all(type(value) is bool for value in metadata["presence_by_request"]):
        raise ValueError("Presence labels must be boolean")
    absent = ~np.asarray(metadata["presence_by_request"], dtype=bool)
    if np.any(np.mean(truth[absent]**2, axis=-1) > SILENCE_POWER):
        raise ValueError("Absent request has nonzero clean reference")
    for key in ("sequential_forward", "sequential_reverse"):
        if key in arrays and arrays[key].shape != p.shape:
            raise ValueError("Sequential output must retain original query axis order")
    if "all_sources" in arrays:
        if (arrays["all_sources"].ndim != 2 or arrays["all_sources"].shape[1] != len(x)
                or not np.allclose(arrays["all_sources"].sum(axis=0), x, atol=2e-7, rtol=1e-6)):
            raise ValueError("All-source references fail mixture alignment")
    if "fuss_all_sources" in arrays:
        if (arrays["fuss_all_sources"].shape != (4, len(x))
                or not isinstance(metadata.get("fuss_checkpoint_sha256"), str)
                or len(metadata["fuss_checkpoint_sha256"]) != 64):
            raise ValueError("FUSS outputs need aligned waves and checkpoint provenance")
    return metadata, arrays, rate


def validate_recipe_identity(metadata: dict[str, Any], recipes: dict[str, dict[str, Any]], split: str) -> None:
    identifier = metadata.get("recipe_id")
    if identifier not in recipes or recipes[identifier]["split"] != split:
        raise ValueError("Unknown recipe or wrong frozen split")
    recipe = recipes[identifier]
    expected = recipe["evaluator_only"]
    if (sorted(metadata.get("source_ids", [])) != sorted(expected["source_files"])
            or metadata.get("queries") != recipe["inference"]["queries"]
            or metadata.get("presence_by_request") != expected["presence_by_request"]
            or metadata.get("edit") != expected["edit"]):
        raise ValueError("Record source/query/presence/edit identity differs from frozen recipe")


def validate_confirmation_gate(split: str, freeze: Path | None, checkpoint_hash: str, recipe_hash: str) -> dict | None:
    if split != "confirmation":
        return None
    if freeze is None:
        raise ValueError("Confirmation requires a frozen final decision record before reading data")
    document = json.loads(freeze.read_text())
    if (document.get("frozen") is not True or document.get("selection_finished") is not True
            or document.get("checkpoint_sha256") != checkpoint_hash
            or document.get("frozen_recipes_sha256") != recipe_hash):
        raise ValueError("Final decision record does not match checkpoint and frozen recipes")
    return {"path": str(freeze), "sha256": digest(freeze), "record": document}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    methods = sorted({method for row in rows for method in row["methods"]})
    for method in methods:
        values = [row["methods"][method] for row in rows if method in row["methods"]]
        result[method] = {"recipes": len(values), "metrics": {}}
        keys = {key for value in values for key, item in value.items() if isinstance(item, (float, int)) or item is None}
        for key in sorted(keys):
            present = [value[key] for value in values if value.get(key) is not None]
            result[method]["metrics"][key] = ({"mean": float(np.mean(present)), "median": float(np.median(present)),
                "minimum": float(np.min(present)), "maximum": float(np.max(present)), "recipes_measured": len(present)} if present else None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path, help="Fresh output directory")
    parser.add_argument("--split", choices=("train", "selection", "confirmation"), default="selection")
    parser.add_argument("--frozen-recipes", type=Path, required=True)
    parser.add_argument("--expected-recipes-sha256", default=FROZEN_SHA256)
    parser.add_argument("--decision-freeze", type=Path)
    parser.add_argument("--absence-threshold", type=float, default=0.01, help="Frozen extracted-energy/mixture threshold")
    parser.add_argument("--scalar-gain", type=float, default=0.60, help="One global gain calibrated on selection only; frozen for confirmation")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--allow-partial", action="store_true", help="Report a clearly labeled incomplete pilot; never imply full frozen coverage")
    args = parser.parse_args()
    if digest(args.frozen_recipes) != args.expected_recipes_sha256:
        raise ValueError("Frozen recipe manifest hash mismatch")
    if not 1 <= args.threads <= 16 or not 0 < args.absence_threshold < 1 or not 0 <= args.scalar_gain <= 4:
        raise ValueError("Invalid threads or absence threshold")
    if args.checkpoint.stat().st_size > 1_000_000:
        raise ValueError("Refiner checkpoint exceeds the artifact size bound")
    checkpoint_hash = digest(args.checkpoint)
    gate = validate_confirmation_gate(args.split, args.decision_freeze, checkpoint_hash, args.expected_recipes_sha256)
    if gate and gate["record"].get("absence_threshold", 0.01) != args.absence_threshold:
        raise ValueError("Absence threshold differs from frozen confirmation decision")
    if gate and (gate["record"].get("edit_protocol") != EDIT_PROTOCOL
                 or gate["record"].get("baseline_scalar_gain") != args.scalar_gain):
        raise ValueError("Edit protocol or scalar calibration differs from frozen confirmation decision")
    frozen = json.loads(args.frozen_recipes.read_text())
    frozen_recipes = {recipe["id"]: recipe for recipe in frozen["recipes"]}
    expected_ids = {recipe["id"] for recipe in frozen["recipes"] if recipe["split"] == args.split}
    document = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    provenance = document["provenance"]
    if provenance["frozen_recipes_sha256"] != args.expected_recipes_sha256:
        raise ValueError("Checkpoint was trained against different frozen recipes")
    model = JointRefiner.load(args.checkpoint)
    torch.set_num_threads(args.threads)
    paths = sorted((args.records / args.split).glob("*.npz"))
    if not paths:
        raise ValueError("No proposal records for requested split")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in paths:
        meta, arrays, rate = load_record(path, args.split, args.expected_recipes_sha256, provenance["backend_checkpoint_sha256"])
        validate_recipe_identity(meta, frozen_recipes, args.split)
        x, p, truth, embeddings = (arrays[key] for key in ("mixture", "proposals", "references", "embeddings"))
        if len(x) != round(frozen["duration_seconds"] * rate):
            raise ValueError("Record duration differs from frozen recipe")
        start = time.perf_counter()
        learned = model.refine(x[:, None], rate, p[:, :, None], embeddings).samples[:, :, 0]
        elapsed = time.perf_counter() - start
        estimates = {"independent": p, "unweighted_consistency": consistency_projection(x[:, None], p[:, :, None])[:, :, 0],
                     "residual_wiener": residual_wiener(x, p), "scalar_fixed_0_75": p*0.75,
                     "scalar_calibrated": p*args.scalar_gain, "trained": learned}
        for key in ("sequential_forward", "sequential_reverse"):
            if key in arrays:
                estimates[key] = arrays[key]
        if "fuss_all_sources" in arrays:
            estimates["fuss_present_evaluator_oracle"] = oracle_fuss_assignment(arrays["fuss_all_sources"], truth, np.asarray(meta["presence_by_request"], dtype=bool))
        method_metrics = {name: metrics(x, prediction, truth, rate, meta, args.absence_threshold,
                                       present_only_oracle=name == "fuss_present_evaluator_oracle",
                                       all_sources=arrays.get("all_sources")) for name, prediction in estimates.items()}
        permutations = list(itertools.permutations(range(len(p)))) if len(p) <= 3 else [tuple(reversed(range(len(p))))]
        permutation_error = 0.0
        for order in permutations:
            if order == tuple(range(len(p))):
                continue
            order_array = np.asarray(order)
            reordered = model.refine(x[:, None], rate, p[order_array, :, None], embeddings[order_array]).samples[np.argsort(order_array), :, 0]
            permutation_error = max(permutation_error, float(np.max(np.abs(reordered - learned))))
        row = {"recipe_id": meta["recipe_id"], "record_sha256": digest(path), "source_ids": meta["source_ids"],
               "methods": method_metrics, "trained_permutation_max_absolute_error": permutation_error,
               "trained_refinement_seconds": elapsed,
               "backend_performance_observed": {"inference": meta.get("inference"), "peak_rss_bytes": meta.get("peak_rss_bytes"), "peak_memory_note": meta.get("peak_memory_note"), "backend_settings": meta.get("backend_settings"), "resampling": meta.get("resampling"), "fuss": meta.get("fuss"), "fuss_checkpoint_sha256": meta.get("fuss_checkpoint_sha256"), "parent_record_sha256": meta.get("parent_record_sha256")},
               "sequential_order_difference_rms": float(np.sqrt(np.mean((arrays["sequential_forward"] - arrays["sequential_reverse"])**2))) if "sequential_forward" in arrays and "sequential_reverse" in arrays else None,
               "failure_flags": [f"trained_edit_worse_than_{baseline}" for baseline in ("independent", "residual_wiener") if method_metrics["trained"]["edit_nmse"] > method_metrics[baseline]["edit_nmse"]]}
        rows.append(row)
        with (args.output / "per-recipe.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps({"recipe": meta["recipe_id"], "trained_edit_nmse": method_metrics["trained"]["edit_nmse"], "failure_flags": row["failure_flags"]}), flush=True)
    observed_ids = [row["recipe_id"] for row in rows]
    if len(set(observed_ids)) != len(observed_ids):
        raise ValueError("Duplicate recipe records would bias the aggregate")
    if set(observed_ids) != expected_ids and not args.allow_partial:
        raise ValueError("Frozen split coverage incomplete; partial per-recipe evidence retained, no aggregate emitted")
    report = {"schema_version": 1, "split": args.split, "checkpoint_sha256": checkpoint_hash,
              "primary_metric": "all_request_edit_nmse", "edit_protocol": EDIT_PROTOCOL,
              "baseline_scalar_gain": args.scalar_gain,
              "frozen_coverage": {"expected_recipes": len(expected_ids), "observed_recipes": len(observed_ids), "complete": set(observed_ids) == expected_ids},
              "checkpoint_training_selection": {key: provenance.get(key) for key in ("epochs_executed", "selected_epoch", "selection_improved", "initial_selection_loss", "selected_selection_loss")},
              "frozen_recipes_sha256": args.expected_recipes_sha256, "decision_freeze": gate,
              "absence_threshold": args.absence_threshold, "aggregate": aggregate(rows),
              "trained_improvement_over_independent_edit_nmse": float(np.mean([row["methods"]["independent"]["edit_nmse"] - row["methods"]["trained"]["edit_nmse"] for row in rows])),
              "trained_improvement_over_wiener_edit_nmse": float(np.mean([row["methods"]["residual_wiener"]["edit_nmse"] - row["methods"]["trained"]["edit_nmse"] for row in rows])),
              "failure_recipes": [{"recipe_id": row["recipe_id"], "flags": row["failure_flags"]} for row in rows if row["failure_flags"]],
              "unmeasured_or_limited": ["No listening-panel judgment", "No natural multitrack reference accuracy", "No certified speaker/session or base-pretraining isolation", "Boundary metrics require observed boundary metadata", "No long-clip/stereo/lock/undo product evidence in this source-reference run", "Sequential/FUSS baselines absent unless actual output arrays supplied", "Latency/memory are unavailable where record metadata lacks observations", "Small recipe aggregate; no source-cluster confidence interval implemented", "FUSS reference assignment, if present, is evaluator-only oracle and not text extraction"]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
