#!/usr/bin/env python3
"""Explicit, reproducible ESC-10 preparation. Never imported by the inference API."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
import urllib.request
import wave

REVISION = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
BASE = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{REVISION}/"
METADATA_HASH = "ca660da60191a97de289983a05821c9382d852a38a2ba8428980816b68cf6246"
LICENSE_HASH = "f6aec3a3283a2efdcbb1feff334adad7ab17c097bdec0cbb4eebdb9e16886074"
SEED = 20261029
SMOKE = {"1-100032-A-0.wav", "1-17367-A-10.wav", "1-116765-A-41.wav"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, destination: Path, expected: str | None = None) -> bytes:
    if destination.is_symlink():
        raise ValueError(f"Refusing symlink cache entry: {destination}")
    if destination.exists():
        data = destination.read_bytes()
        if expected and digest(data) != expected:
            raise ValueError(f"Cached content hash mismatch: {destination}")
        return data
    error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "STEMSCENE-data-preparation/0.1"})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(2_000_001)
            if len(data) > 2_000_000:
                raise ValueError("Unexpected file larger than the per-file download bound")
            if expected and digest(data) != expected:
                raise ValueError(f"Upstream content hash mismatch: {url}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(prefix=".download-", dir=destination.parent)
            try:
                with os.fdopen(handle, "wb") as stream:
                    stream.write(data)
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return data
        except (OSError, ValueError) as exc:
            error = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Download failed: {url}: {error}") from error


def metadata(cache: Path, offline: bool) -> tuple[list[dict], str]:
    paths = [("meta/esc50.csv", METADATA_HASH), ("LICENSE", LICENSE_HASH)]
    content = []
    for remote, expected in paths:
        destination = cache / remote
        if offline:
            data = destination.read_bytes()
            if digest(data) != expected:
                raise ValueError(f"Metadata hash mismatch: {remote}")
        else:
            data = fetch(BASE + remote, destination, expected)
        content.append(data.decode("utf-8"))
    rows = [dict(row) for row in csv.DictReader(io.StringIO(content[0])) if row["esc10"] == "True"]
    entries = {}
    for match in re.finditer(r"- \[([^\]]+)\]:([^\n]*(?:\n(?!- \[)[^\n]*)*)", content[1]):
        link = re.search(r"https?://(?:www\.)?freesound\.org/people/([^/]+)/sounds/(\d+)/?", match[2])
        if link:
            entries[match[1]] = {
                "uploader": urllib.parse.unquote(link[1]).casefold(),
                "source_url": link[0],
                "attribution": " ".join(match[2].split()),
            }
    if len(rows) != 400:
        raise ValueError("Expected exactly 400 official ESC-10 clips")
    for row in rows:
        name = row["filename"]
        if not re.fullmatch(r"[1-5]-\d+-[A-Z]-\d+\.wav", name):
            raise ValueError(f"Unsafe or unexpected filename: {name}")
        old_name = name.removesuffix(".wav").rsplit("-", 1)[0] + ".ogg"
        row.update(entries[old_name])
        row["fold"] = int(row["fold"])
        row["target"] = int(row["target"])
        row.pop("esc10")
        row["url"] = BASE + "audio/" + name
        row["dataset_license"] = "CC-BY-3.0 (official ESC-10 subset)"
    return rows, content[1]


def verify_audio(data: bytes) -> None:
    with wave.open(io.BytesIO(data), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()) != (1, 2, 44100, 220500):
            raise ValueError("Unexpected ESC-10 WAV audio contract")
        if len(audio.readframes(220500)) != 441000:
            raise ValueError("Truncated ESC-10 WAV")


def split_rows(rows: list[dict]) -> dict[str, list[dict]]:
    protected_test = {row["uploader"] for row in rows if row["fold"] == 5}
    protected_selection = {row["uploader"] for row in rows if row["fold"] == 4}
    result = {
        "train": [row for row in rows if row["fold"] <= 3 and row["uploader"] not in protected_test | protected_selection],
        "selection": [row for row in rows if row["fold"] == 4 and row["uploader"] not in protected_test],
        "confirmation": [row for row in rows if row["fold"] == 5],
    }
    for a, b in itertools.combinations(result, 2):
        for key in ("src_file", "uploader", "sha256"):
            if {row[key] for row in result[a]} & {row[key] for row in result[b]}:
                raise ValueError(f"Split leakage in {key}: {a}/{b}")
    if SMOKE & {row["filename"] for row in result["confirmation"]}:
        raise ValueError("Smoke examples entered confirmation")
    if [len(result[x]) for x in ("train", "selection", "confirmation")] != [222, 74, 80]:
        raise ValueError("Unexpected source split counts")
    return result


def recipes(splits: dict[str, list[dict]], seed: int) -> list[dict]:
    rng = random.Random(seed)
    categories = sorted({row["category"] for row in splits["train"]})
    combinations = list(itertools.combinations(categories, 3))
    rng.shuffle(combinations)
    groups = {"train": combinations[:60], "selection": combinations[60:80], "confirmation": combinations[80:]}
    counts = {"train": 30, "selection": 10, "confirmation": 20}
    result = []
    for split, count in counts.items():
        by_category = {category: [row for row in splits[split] if row["category"] == category] for category in categories}
        for index in range(count):
            present = groups[split][index]
            sources = [rng.choice(by_category[category]) for category in present]
            queries = list(present)
            if index % 3 == 2:
                queries[-1] = rng.choice([category for category in categories if category not in present])
            rng.shuffle(queries)
            source_gain = [round(rng.uniform(0.25, 0.65), 8) for _ in sources]
            identifier = f"{split}-{index:03d}"
            result.append({
                "id": identifier,
                "split": split,
                "class_tuple": list(present),
                "inference": {"input": "mixture.wav", "queries": [query.replace("_", " ") for query in queries], "anchors": []},
                "evaluator_only": {
                    "source_files": [row["filename"] for row in sources],
                    "source_categories": list(present),
                    "pre_peak_bound_gains": source_gain,
                    "presence_by_request": [query in present for query in queries],
                    "reference_policy": "full 5-second clips; each scaled source plus shared peak-bound factor; no clean source or presence labels passed to inference",
                    "edit": {"request_index": 0, "start_seconds": 1.0, "end_seconds": 4.0, "gain": 0.25, "fade_seconds": 0.02},
                },
            })
    for a, b in itertools.combinations(counts, 2):
        left = {tuple(recipe["class_tuple"]) for recipe in result if recipe["split"] == a}
        right = {tuple(recipe["class_tuple"]) for recipe in result if recipe["split"] == b}
        if left & right:
            raise ValueError("Class-combination leakage")
    return result


def freeze(path: Path, manifest: dict) -> None:
    content = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Refusing to overwrite different frozen manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(content)


def materialize(manifest: dict, cache: Path, output: Path) -> None:
    import numpy as np
    import soundfile as sf

    if output.exists():
        raise ValueError("Materialization destination must be fresh")
    output.mkdir(parents=True)
    for recipe in manifest["recipes"]:
        destination = output / recipe["id"]
        public = destination / "inference"
        evaluator = destination / "evaluator"
        public.mkdir(parents=True)
        evaluator.mkdir()
        sources = []
        info = recipe["evaluator_only"]
        for filename, gain in zip(info["source_files"], info["pre_peak_bound_gains"]):
            data, rate = sf.read(cache / "audio" / filename, dtype="float64", always_2d=True)
            if rate != 44100 or data.shape != (220500, 1):
                raise ValueError("Unexpected cached source")
            sources.append(data * gain)
        mixture = sum(sources, np.zeros_like(sources[0]))
        factor = min(1.0, 0.9 / max(float(np.max(np.abs(mixture))), 1e-12))
        sources = [source * factor for source in sources]
        mixture = sum(sources, np.zeros_like(sources[0]))
        sf.write(public / "mixture.wav", mixture, 44100, subtype="DOUBLE")
        (public / "request.json").write_text(json.dumps(recipe["inference"], indent=2) + "\n")
        for index, source in enumerate(sources):
            sf.write(evaluator / f"source-{index}.wav", source, 44100, subtype="DOUBLE")
        (evaluator / "labels.json").write_text(json.dumps({**info, "shared_peak_bound_factor": factor, "actual_gains": [factor * gain for gain in info["pre_peak_bound_gains"]]}, indent=2) + "\n")
        recombined = sum((sf.read(evaluator / f"source-{i}.wav", always_2d=True)[0] for i in range(len(sources))), np.zeros_like(mixture))
        decoded_mix = sf.read(public / "mixture.wav", always_2d=True)[0]
        if not np.array_equal(recombined, decoded_mix):
            raise ValueError("Reference recombination is not sample-exact")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "stemscene-data")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Explicitly allow public metadata/audio network requests")
    parser.add_argument("--materialize", type=Path, help="Fresh output folder; needs NumPy and soundfile")
    args = parser.parse_args()
    cache = args.cache.expanduser().resolve()
    rows, _ = metadata(cache, offline=not args.download)

    def prepare(row: dict) -> dict:
        path = cache / "audio" / row["filename"]
        data = fetch(row["url"], path) if args.download else path.read_bytes()
        verify_audio(data)
        return {**row, "sha256": digest(data), "bytes": len(data)}

    with ThreadPoolExecutor(max_workers=6) as executor:
        rows = list(executor.map(prepare, rows))
    splits = split_rows(rows)
    manifest = {
        "schema_version": 1,
        "dataset": "ESC-10 official subset; controlled mixtures of real recordings",
        "upstream_commit": REVISION,
        "metadata_sha256": METADATA_HASH,
        "license_sha256": LICENSE_HASH,
        "seed": SEED,
        "duration_seconds": 5,
        "sample_rate": 44100,
        "channels": 1,
        "split_policy": "folds1-3 train,4 selection,5 confirmation; confirmation uploader priority; remove original selection+confirmation uploaders from train; URL-decode+casefold uploaders; no shared original source/uploader/byte hash",
        "claim_limits": ["Unknown base separator pretraining overlap", "Uploader grouping is not certified speaker/session grouping", "Weak clip class labels do not establish source purity", "Controlled mixes are not naturally captured multitrack truth", "Main recipe combinations are unfamiliar in later splits; familiar-combination evaluation is a separate future stratum"],
        "development_smoke_files": sorted(SMOKE),
        "splits": splits,
        "recipes": recipes(splits, SEED),
    }
    freeze(args.manifest, manifest)
    if args.materialize:
        materialize(manifest, cache, args.materialize)
    print(json.dumps({"manifest": str(args.manifest), "sha256": digest(args.manifest.read_bytes()), "split_clips": {key: len(value) for key, value in splits.items()}, "recipes": len(manifest["recipes"]), "cache": str(cache)}, indent=2))


if __name__ == "__main__":
    main()
