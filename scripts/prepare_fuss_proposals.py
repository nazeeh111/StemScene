#!/usr/bin/env python3
"""Append actual unlabelled FUSS outputs to fresh copies of verified records.

Selection only by default. References are copied after inference, never passed
to the separator. Semantic source assignment belongs to the evaluator alone.
No network access, training or modification of original records occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import tarfile
import time
import zipfile

import numpy as np

# Supports a standalone TensorFlow environment without importing the installed
# package initializer (which may import the main PyTorch backend).
import importlib.util

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stemscene_fuss_comparator", REPO / "src/stemscene/fuss.py")
FUSS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FUSS)
MAX_ARCHIVE_BYTES = 256_000_000


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def memory() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def checked_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > 1_000_000:
        raise ValueError(f"Missing or oversized metadata: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Metadata must be a JSON object")
    return value


def verify_model(archive: Path, folder: Path) -> dict:
    if archive.stat().st_size != FUSS.FUSS_DRY_BYTES or sha(archive) != FUSS.FUSS_DRY_SHA256:
        raise ValueError("FUSS dry checkpoint archive size/SHA256 mismatch")
    expected = {"baseline_dry_inference.meta", "baseline_dry_model.index",
                "baseline_dry_model.data-00000-of-00001"}
    found = {}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.isdir():
                continue
            if (not member.isfile() or member.name != "baseline_dry_model/" + Path(member.name).name
                    or Path(member.name).name not in expected or member.size > 200_000_000):
                raise ValueError("Unexpected member in verified FUSS archive")
            with tar.extractfile(member) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            name = Path(member.name).name
            if name in found or not (folder / name).is_file() or sha(folder / name) != digest:
                raise ValueError("Extracted FUSS artifact does not match verified archive")
            found[name] = digest
    if set(found) != expected:
        raise ValueError("FUSS archive lacks required checkpoint files")
    return found


def verify_record(path: Path, split: str, recipe_hash: str, recipes: dict) -> tuple[dict, str]:
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Source NPZ exceeds compressed size bound")
    with zipfile.ZipFile(path) as zipped:
        entries = zipped.infolist()
        if (len(entries) > 32 or len({e.filename for e in entries}) != len(entries)
                or sum(e.file_size for e in entries) > MAX_ARCHIVE_BYTES):
            raise ValueError("Source NPZ exceeds expanded size/member bounds")
    metadata = checked_json(path.with_suffix(".json"))
    digest = sha(path)
    if (metadata.get("record_sha256") != digest or metadata.get("split") != split
            or metadata.get("frozen_recipes_sha256") != recipe_hash):
        raise ValueError("Source NPZ hash/split/frozen recipe provenance mismatch")
    identifier = metadata.get("recipe_id")
    if identifier not in recipes or recipes[identifier]["split"] != split:
        raise ValueError("Unknown recipe identity")
    recipe = recipes[identifier]
    if metadata.get("queries") != recipe["inference"]["queries"]:
        raise ValueError("Record query identity differs from frozen recipe")
    if metadata.get("source_ids") != recipe["evaluator_only"]["source_files"]:
        raise ValueError("Record source identity differs from frozen recipe")
    if "fuss_checkpoint_sha256" in metadata:
        raise ValueError("Source already augmented with FUSS")
    return metadata, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="Original proposal root; read only")
    parser.add_argument("output", type=Path, help="Fresh, separate augmented root")
    parser.add_argument("--frozen-recipes", type=Path, required=True)
    parser.add_argument("--expected-recipes-sha256", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-archive", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "selection", "confirmation"), default="selection")
    parser.add_argument("--decision-freeze", type=Path)
    parser.add_argument("--refiner-checkpoint", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Explicit partial pilot; never a complete aggregate")
    args = parser.parse_args()
    if not 1 <= args.threads <= 16 or (args.limit is not None and not 1 <= args.limit <= 1000):
        parser.error("Invalid resource bound")
    recipe_hash = sha(args.frozen_recipes)
    if recipe_hash != args.expected_recipes_sha256:
        parser.error("Frozen recipe manifest SHA256 mismatch")
    gate = None
    if args.split == "confirmation":
        if args.decision_freeze is None or args.refiner_checkpoint is None:
            parser.error("Confirmation requires final model and matching decision freeze before reading records")
        decision = checked_json(args.decision_freeze)
        if (decision.get("frozen") is not True or decision.get("selection_finished") is not True
                or decision.get("checkpoint_sha256") != sha(args.refiner_checkpoint)
                or decision.get("frozen_recipes_sha256") != recipe_hash):
            parser.error("Confirmation freeze does not match final model and recipes")
        gate = {"sha256": sha(args.decision_freeze), "checkpoint_sha256": sha(args.refiner_checkpoint)}
    # Confirmation records have not been listed/opened before the gate above.
    manifest = json.loads(args.frozen_recipes.read_text())
    recipes = {r["id"]: r for r in manifest["recipes"]}
    paths = sorted((args.records / args.split).glob("*.npz"))
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        parser.error("No source NPZ records exist for the requested split")
    expected_ids = {r["id"] for r in manifest["recipes"] if r["split"] == args.split}
    if args.limit is None and {p.stem for p in paths} != expected_ids:
        parser.error("Source split coverage differs from frozen recipes; use an explicit pilot limit if intended")
    source_root, target_root = args.records.resolve(), args.output.resolve()
    if target_root == source_root or target_root.is_relative_to(source_root):
        parser.error("Augmented output must be separate from original record root")
    if target_root.exists():
        parser.error("Output already exists; use a fresh root and preserve prior evidence")
    model_files = verify_model(args.model_archive, args.model_dir)
    verified = [(p, *verify_record(p, args.split, recipe_hash, recipes)) for p in paths]
    target_root.mkdir(parents=True, exist_ok=False)
    (target_root / args.split).mkdir()
    began = time.perf_counter()
    started = time.perf_counter()
    backend = FUSS.FussSeparator(args.model_dir, max_seconds=30, cpu_threads=args.threads)
    cold_load = time.perf_counter() - started
    rows = []
    try:
        for path, original_meta, parent_hash in verified:
            started = time.perf_counter()
            with np.load(path, allow_pickle=False) as source:
                rate = int(source["sample_rate"].item())
                mixture = np.asarray(source["mixture"], dtype=np.float64)
                if mixture.ndim != 1 or not 1024 <= len(mixture) <= 960000:
                    raise ValueError("Mixture shape/duration exceeds comparator bounds")
                # Only mixed audio and its rate are supplied, no references.
                pred = backend.separate(mixture[:, None], rate)[:, :, 0]
                inference_seconds = time.perf_counter() - started
                if pred.shape != (4, len(mixture)) or not np.isfinite(pred).all():
                    raise ValueError("FUSS output is unaligned or nonfinite")
                arrays = {name: source[name].copy() for name in source.files}
            if any(name in arrays for name in ("fuss_all_sources", "fuss_proposals")):
                raise ValueError("Source record already contains FUSS estimates")
            if any(a.dtype.hasobject or not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all()
                   for a in arrays.values()):
                raise ValueError("Original archive contains unsupported/nonfinite arrays")
            if sha(path) != parent_hash:
                raise ValueError("Source NPZ changed during inference")
            arrays["fuss_all_sources"] = pred.astype(np.float32)
            output = target_root / args.split / path.name
            with output.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            metadata = dict(original_meta)
            metadata.update({"record_sha256": sha(output), "parent_record_sha256": parent_hash,
                "parent_metadata_sha256": sha(path.with_suffix(".json")),
                "fuss_checkpoint_sha256": FUSS.FUSS_DRY_SHA256,
                "fuss_checkpoint_files_sha256": model_files,
                "fuss_inference_input_sha256": hashlib.sha256(mixture.astype('<f8').tobytes()).hexdigest(),
                "fuss_all_sources_sha256": hashlib.sha256(arrays["fuss_all_sources"].astype('<f4').tobytes()).hexdigest(),
                "fuss": {"assignment_policy": "evaluator-only-oracle; no preparation assignment",
                    "semantic_absence_supported": False, "native_rate": 16000, "record_rate": rate,
                    "bandwidth": "16 kHz mono model; converted back to record rate, no recovered high frequencies",
                    "inference_seconds": inference_seconds, "cold_model_load_seconds": cold_load,
                    "peak_rss_bytes_cumulative": memory(), "threads": args.threads,
                    "memory_note": "process cumulative maximum RSS; not isolated per-record allocation",
                    "code_license": "Apache-2.0", "model_record_license": "CC-BY-4.0",
                    "model_record": "https://doi.org/10.5281/zenodo.4012661",
                    "provenance": backend.provenance, "confirmation_gate": gate}})
            with output.with_suffix(".json").open("x") as stream:
                json.dump(metadata, stream, indent=2); stream.write("\n")
            rows.append({"record": str(output.relative_to(target_root)), "record_sha256": metadata["record_sha256"],
                         "parent_record_sha256": parent_hash, "inference_seconds": inference_seconds})
            print(json.dumps(rows[-1]), flush=True)
    finally:
        backend.close()
    report = {"split": args.split, "frozen_recipes_sha256": recipe_hash,
              "fuss_checkpoint_sha256": FUSS.FUSS_DRY_SHA256,
              "cold_model_load_seconds": cold_load, "elapsed_seconds": time.perf_counter()-began,
              "peak_rss_bytes_cumulative": memory(), "records": rows, "explicit_limit": args.limit,
              "frozen_split_expected_records": len(expected_ids),
              "frozen_split_coverage_complete": {p.stem for p in paths} == expected_ids,
              "assignment_policy": "raw unlabelled outputs; evaluator-only oracle for present queries",
              "confirmation_gate": gate}
    with (target_root / "fuss-augmentation.json").open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")


if __name__ == "__main__":
    main()
