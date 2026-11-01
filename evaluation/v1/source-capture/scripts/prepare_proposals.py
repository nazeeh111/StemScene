#!/usr/bin/env python3
"""Compute real AudioSep proposals; references enter only the record writer.

This is an offline training/evaluation utility, outside the installed inference
API. Inference always runs on inference/mixture.wav and request.json only.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import numpy as np
import soundfile as sf

from stemscene.backend import AudioSepBackend, BackendLimits, resample_aligned


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixtures", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--split", nargs="+", choices=["train", "selection", "confirmation"], default=["train", "selection"])
    parser.add_argument("--decision-freeze", type=Path)
    parser.add_argument("--sequential", action="store_true", help="compute both query orders on actual residual audio")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest_hash = sha(args.manifest)
    if "confirmation" in args.split:
        if not args.decision_freeze:
            parser.error("confirmation requires a frozen final model decision")
        freeze = json.loads(args.decision_freeze.read_text())
        if not freeze.get("frozen") or not freeze.get("selection_finished") or freeze.get("frozen_recipes_sha256") != manifest_hash:
            parser.error("confirmation freeze is incomplete or mismatched")
    manifest = json.loads(args.manifest.read_text())
    recipes = [r for r in manifest["recipes"] if r["split"] in args.split]
    if args.limit is not None:
        recipes = recipes[:args.limit]
    limits = BackendLimits(threads=args.threads)
    before = time.perf_counter()
    backend = AudioSepBackend(args.checkpoint, args.tokenizer, limits=limits, device=args.device)
    load_seconds = time.perf_counter()-before
    print(json.dumps({"backend_loaded_seconds": load_seconds}), flush=True)
    for index, recipe in enumerate(recipes):
        root = args.mixtures / recipe["id"]
        output = args.output / recipe["split"] / f"{recipe['id']}.npz"
        sidecar = output.with_suffix(".json")
        if output.exists() and sidecar.exists():
            previous = json.loads(sidecar.read_text())
            if (previous.get("record_sha256") != sha(output) or previous.get("frozen_recipes_sha256") != manifest_hash
                    or previous.get("backend_checkpoint_sha256") != backend.checkpoint_sha256):
                raise ValueError("resume record provenance does not match; use fresh output directory")
            if args.sequential and not previous.get("sequential_computed"):
                raise ValueError("resume missing requested sequential baselines; use a fresh output")
            print(f"verified cached real computation {recipe['id']}", flush=True)
            continue
        request = json.loads((root / "inference/request.json").read_text())
        if request != recipe["inference"]:
            raise ValueError("inference request changed from frozen recipe")
        x, rate = sf.read(root / "inference/mixture.wav", dtype="float64", always_2d=True)
        queries = request["queries"]
        began = time.perf_counter()
        proposals = backend.separate(x, rate, queries)
        independent_seconds = time.perf_counter()-began
        embeddings = backend.embed_queries(queries)
        arrays = {"mixture": resample_aligned(x, rate, 32000)[:, 0].astype(np.float32),
                  "proposals": np.stack([resample_aligned(y, rate, 32000)[:, 0] for y in proposals]).astype(np.float32),
                  "embeddings": embeddings.astype(np.float32), "sample_rate": np.array(32000)}
        timings = {"independent_seconds": independent_seconds, "backend_load_seconds": load_seconds}
        if args.sequential:
            for label, order in [("sequential_forward", list(range(len(queries)))),
                                 ("sequential_reverse", list(reversed(range(len(queries)))) )]:
                began = time.perf_counter()
                residual, sequential = x.copy(), np.zeros_like(proposals)
                for qi in order:
                    extracted = backend.separate(residual, rate, [queries[qi]])[0]
                    sequential[qi] = extracted
                    residual -= extracted
                arrays[label] = np.stack([resample_aligned(y, rate, 32000)[:, 0] for y in sequential]).astype(np.float32)
                timings[label+"_seconds"] = time.perf_counter()-began
        # Clean files/presence are opened only after all inference computation.
        labels = json.loads((root / "evaluator/labels.json").read_text())
        sources = []
        for si in range(len(labels["source_files"])):
            source, source_rate = sf.read(root / f"evaluator/source-{si}.wav", dtype="float64", always_2d=True)
            if source_rate != rate or source.shape != x.shape:
                raise ValueError("unaligned evaluator source")
            sources.append(resample_aligned(source, rate, 32000)[:, 0])
        arrays["all_sources"] = np.stack(sources).astype(np.float32)
        by_category = dict(zip(labels["source_categories"], sources))
        arrays["references"] = np.stack([by_category.get(q.replace(" ", "_"), np.zeros(len(sources[0])))
                                           for q in queries]).astype(np.float32)
        if not np.allclose(arrays["all_sources"].sum(axis=0), arrays["mixture"], atol=2e-7, rtol=1e-6):
            raise ValueError("resampled evaluator sources fail mixture alignment")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".partial.npz")
        if temporary.exists():
            raise FileExistsError("incomplete prior record retained; inspect before choosing new output")
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        os.rename(temporary, output)
        metadata = {"split": recipe["split"], "recipe_id": recipe["id"],
                    "source_ids": labels["source_files"], "queries": queries,
                    "presence_by_request": labels["presence_by_request"], "edit": labels["edit"],
                    "backend_checkpoint_sha256": backend.checkpoint_sha256,
                    "frozen_recipes_sha256": manifest_hash, "record_sha256": sha(output),
                    "input_sha256": sha(root / "inference/mixture.wav"),
                    "inference": timings, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "peak_memory_note": "process cumulative maximum RSS on macOS, not isolated per-record memory",
                    "model_rate": 32000, "original_rate": rate,
                    "resampling": "zero-phase polyphase FIR, right trim/pad, no clean-reference delay fitting",
                    "backend_settings": asdict(limits), "sequential_computed": args.sequential}
        with sidecar.open("x") as stream:
            json.dump(metadata, stream, indent=2); stream.write("\n")
        print(json.dumps({"completed": recipe["id"], "index": index+1, "total": len(recipes), **timings}), flush=True)


if __name__ == "__main__":
    main()
