#!/usr/bin/env python3
"""Compute actual forward/reverse residual extraction on verified originals.

No clean references or presence labels enter inference. Copies parent arrays
into a fresh root and adds two query-axis-aligned sequential output arrays.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import zipfile

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stemscene.backend import AudioSepBackend, BackendLimits, resample_aligned

BOUND = 256_000_000


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def peak_rss() -> int:
    result = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(result if sys.platform == "darwin" else result * 1024)


def metadata(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > 1_000_000:
        raise ValueError("Missing or oversized metadata")
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError("Metadata must be a JSON object")
    return result


def validate_archive(path: Path) -> None:
    if path.stat().st_size > BOUND:
        raise ValueError("Source NPZ exceeds compressed size bound")
    with zipfile.ZipFile(path) as zipped:
        entries = zipped.infolist()
        if (len(entries) > 32 or len({e.filename for e in entries}) != len(entries)
                or sum(e.file_size for e in entries) > BOUND):
            raise ValueError("Source NPZ exceeds expansion/member bounds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("mixtures", type=Path, help="Original mixture root; only inference subfolders are read")
    parser.add_argument("output", type=Path, help="Fresh separate augmented record root")
    parser.add_argument("--frozen-recipes", type=Path, required=True)
    parser.add_argument("--expected-recipes-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "selection", "confirmation"), default="selection")
    parser.add_argument("--decision-freeze", type=Path)
    parser.add_argument("--refiner-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 10:
        parser.error("threads must be in [1, 10]")
    recipe_hash = sha(args.frozen_recipes)
    if recipe_hash != args.expected_recipes_sha256:
        parser.error("Frozen recipe manifest hash mismatch")
    gate = None
    if args.split == "confirmation":
        if args.decision_freeze is None or args.refiner_checkpoint is None:
            parser.error("Confirmation requires matching final model and decision freeze before opening records")
        decision = metadata(args.decision_freeze)
        model_hash = sha(args.refiner_checkpoint)
        if (decision.get("frozen") is not True or decision.get("selection_finished") is not True
                or decision.get("checkpoint_sha256") != model_hash
                or decision.get("frozen_recipes_sha256") != recipe_hash):
            parser.error("Confirmation freeze differs from model or recipes")
        gate = {"sha256": sha(args.decision_freeze), "checkpoint_sha256": model_hash}
    frozen = json.loads(args.frozen_recipes.read_text())
    recipes = {r["id"]: r for r in frozen["recipes"] if r["split"] == args.split}
    paths = sorted((args.records / args.split).glob("*.npz"))
    if not paths or {p.stem for p in paths} != set(recipes):
        parser.error("Source split must completely match the frozen recipe IDs")
    source_root, output_root = args.records.resolve(), args.output.resolve()
    if (output_root.exists() or output_root.is_relative_to(source_root)
            or output_root.is_relative_to(args.mixtures.resolve())):
        parser.error("Use a fresh output root outside source records and original mixtures")
    checkpoint_hash = sha(args.checkpoint)
    verified = []
    limits = None
    for path in paths:
        validate_archive(path)
        meta = metadata(path.with_suffix(".json"))
        parent_hash = sha(path)
        recipe = recipes[path.stem]
        if (meta.get("record_sha256") != parent_hash or meta.get("split") != args.split
                or meta.get("recipe_id") != path.stem
                or meta.get("frozen_recipes_sha256") != recipe_hash
                or meta.get("backend_checkpoint_sha256") != checkpoint_hash
                or meta.get("queries") != recipe["inference"]["queries"]):
            raise ValueError("Source record hash/model/recipe/query provenance mismatch")
        setting = BackendLimits(**meta["backend_settings"])
        if setting.threads != args.threads or (limits is not None and setting != limits):
            raise ValueError("Sequential run must retain independent baseline settings")
        limits = setting
        inf = args.mixtures / path.stem / "inference"
        mixture_path, request_path = inf / "mixture.wav", inf / "request.json"
        if (mixture_path.stat().st_size > BOUND or sha(mixture_path) != meta["input_sha256"]
                or metadata(request_path) != recipe["inference"]):
            raise ValueError("Original mixed recording/request differs from primary computation")
        if meta.get("sequential_computed"):
            raise ValueError("Source record is already augmented with sequential computation")
        verified.append((path, meta, parent_hash, mixture_path))
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / args.split).mkdir()
    started = time.perf_counter()
    backend = AudioSepBackend(args.checkpoint, args.tokenizer, limits=limits, device=args.device)
    cold_load = time.perf_counter() - started
    print(json.dumps({"backend_loaded_seconds": cold_load, "device": args.device}), flush=True)
    rows = []
    for path, parent, parent_hash, mixture_path in verified:
        x, original_rate = sf.read(mixture_path, dtype="float64", always_2d=True)
        if (x.ndim != 2 or x.shape[1] != 1 or original_rate != parent["original_rate"]
                or len(x) != round(frozen["duration_seconds"] * original_rate) or not np.isfinite(x).all()):
            raise ValueError("Original recording shape/rate/duration invalid")
        with np.load(path, allow_pickle=False) as archive:
            rate = int(archive["sample_rate"].item())
            stored_mixture = archive["mixture"]
            expected = resample_aligned(x, original_rate, rate)[:, 0].astype(stored_mixture.dtype)
            if not np.array_equal(stored_mixture, expected):
                raise ValueError("Original-rate mixture does not reproduce primary stored mixture")
        queries = parent["queries"]
        if not 1 <= len(queries) <= 8:
            raise ValueError("Query count exceeds comparator bound")
        computed, timings = {}, {}
        for name, order in (("sequential_forward", list(range(len(queries)))),
                            ("sequential_reverse", list(reversed(range(len(queries)))))):
            began = time.perf_counter()
            residual = x.copy()
            outputs = np.empty((len(queries), len(x), 1), dtype=np.float64)
            calls = []
            for index in order:
                before = time.perf_counter()
                proposal = backend.separate(residual, original_rate, [queries[index]])[0]
                calls.append({"query_index": index, "seconds": time.perf_counter()-before})
                outputs[index] = proposal
                residual -= proposal
            computed[name] = np.stack([resample_aligned(y, original_rate, rate)[:, 0]
                                       for y in outputs]).astype(np.float32)
            timings[name + "_seconds"] = time.perf_counter()-began
            timings[name + "_calls"] = calls
        # Only after both real inference orders have completed, copy all source
        # arrays. Clean references never entered the extraction API.
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        if any(name in arrays for name in computed):
            raise ValueError("Source NPZ already contains sequential output arrays")
        if (sha(path) != parent_hash or sha(mixture_path) != parent["input_sha256"]
                or any(a.dtype.hasobject or not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all()
                       for a in arrays.values())):
            raise ValueError("Parent/original changed or arrays are unsupported")
        arrays.update(computed)
        out = output_root / args.split / path.name
        with out.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        meta = dict(parent)
        lineage = list(parent.get("record_ancestry", []))
        if parent.get("parent_record_sha256"):
            lineage.append({"role": "prior-parent", "record_sha256": parent["parent_record_sha256"],
                            "metadata_sha256": parent.get("parent_metadata_sha256")})
        lineage.append({"role": "sequential-input-parent", "record_sha256": parent_hash,
                        "metadata_sha256": sha(path.with_suffix(".json"))})
        inference = dict(parent.get("inference", {}))
        inference.update({k: v for k, v in timings.items() if k.endswith("_seconds")})
        meta.update({"record_sha256": sha(out), "parent_record_sha256": parent_hash,
                     "parent_metadata_sha256": sha(path.with_suffix(".json")),
                     "record_ancestry": lineage, "sequential_computed": True, "inference": inference,
                     "sequential": {"checkpoint_sha256": checkpoint_hash, "settings": asdict(limits),
                        "device": args.device, "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
                        "cold_model_load_seconds": cold_load, "timings": timings,
                        "peak_rss_bytes_cumulative": peak_rss(), "confirmation_gate": gate,
                        "original_sample_rate": original_rate, "record_sample_rate": rate,
                        "model_calls_per_order": len(queries), "stored_query_axis": "original query order",
                        "residual_update": "subtract returned proposal at original recording sample rate",
                        "array_sha256": {k: hashlib.sha256(v.astype('<f4').tobytes()).hexdigest() for k,v in computed.items()}}})
        with out.with_suffix(".json").open("x") as stream:
            json.dump(meta, stream, indent=2); stream.write("\n")
        row = {"recipe_id": path.stem, "record_sha256": meta["record_sha256"],
               "parent_record_sha256": parent_hash, "forward_seconds": timings["sequential_forward_seconds"],
               "reverse_seconds": timings["sequential_reverse_seconds"],
               "order_difference_rms": float(np.sqrt(np.mean((computed["sequential_forward"].astype(float)-computed["sequential_reverse"])**2)))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {"split": args.split, "frozen_recipes_sha256": recipe_hash,
              "backend_checkpoint_sha256": checkpoint_hash, "device": args.device,
              "cold_model_load_seconds": cold_load, "records": rows,
              "peak_rss_bytes_cumulative": peak_rss(), "confirmation_gate": gate,
              "frozen_split_coverage_complete": len(rows) == len(recipes),
              "comparison": "actual residual extraction at original mixture rate, then aligned record-rate conversion"}
    with (output_root / "sequential-augmentation.json").open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")


if __name__ == "__main__":
    main()
