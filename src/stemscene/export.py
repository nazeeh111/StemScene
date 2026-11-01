"""Aligned floating WAV exports and a standalone, verifiable edit recipe."""
from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Literal

import numpy as np
import soundfile as sf

from .audio import AudioContractError, Samples, sha256_file, validate_samples
from .project import Project, _read_json, _safe_path, _write_json


def _verify_written_audio(root: Path, manifest: dict[str, Any], original: Samples,
                          final: Samples) -> dict[str, Any]:
    """Measure actual WAV round trips in bounded blocks before publishing a folder.

    Tolerances are absolute waveform amplitude units, never scaled by the large
    cancelling stems. Algebra before serialization is insufficient evidence.
    """
    tolerance = 1e-12 if manifest["subtype"] == "DOUBLE" else 1e-6
    errors = {"unity_max_abs_error": 0.0, "edit_replay_max_abs_error": 0.0,
              "mix_render_max_abs_error": 0.0, "original_decode_max_abs_error": 0.0}
    shape = (manifest["frames"], manifest["channels"])
    with ExitStack() as stack:
        streams: dict[str, sf.SoundFile] = {}

        def open_wav(relative: str, channels: int) -> None:
            stream = stack.enter_context(sf.SoundFile(_safe_path(root, relative)))
            if ((stream.frames, stream.channels) != (shape[0], channels)
                    or stream.samplerate != manifest["sample_rate"]
                    or stream.format not in {"WAV", "WAVEX", "RF64"}):
                raise AudioContractError("Written WAV does not satisfy export alignment")
            streams[relative] = stream

        for relative in ("original-decoded.wav", "residual.wav", "mix.wav"):
            open_wav(relative, shape[1])
        for layer in manifest["layers"]:
            open_wav(layer["stem"], shape[1])
            open_wav(layer["envelope"], 1)

        for start in range(0, shape[0], 65536):
            stop = min(shape[0], start + 65536)
            count = stop - start

            def read(relative: str) -> np.ndarray:
                values = streams[relative].read(count, dtype="float64", always_2d=True)
                if len(values) != count or not np.isfinite(values).all():
                    raise AudioContractError("Written WAV has missing or nonfinite samples")
                return values

            decoded = read("original-decoded.wav")
            unity = read("residual.wav")
            mix = read("mix.wav")
            replay = decoded.copy()
            for layer in manifest["layers"]:
                stem, envelope = read(layer["stem"]), read(layer["envelope"])
                unity += stem
                active = envelope[:, 0] != 1.0
                replay[active] += (envelope[active] - 1.0) * stem[active]
            if manifest["clipping"] == "hard":
                replay = np.clip(replay, -1, 1)
            measured = {
                "unity_max_abs_error": np.max(np.abs(unity - original[start:stop])),
                "edit_replay_max_abs_error": np.max(np.abs(replay - mix)),
                "mix_render_max_abs_error": np.max(np.abs(mix - final[start:stop])),
                "original_decode_max_abs_error": np.max(np.abs(decoded - original[start:stop])),
            }
            for key, value in measured.items():
                errors[key] = max(errors[key], float(value))
    failures = {key: value for key, value in errors.items()
                if not np.isfinite(value) or value > tolerance}
    if failures:
        detail = ", ".join(f"{key}={value:.6g}" for key, value in failures.items())
        remedy = " Try DOUBLE export." if manifest["subtype"] == "FLOAT" else " Reduce incompatible cancelling estimates."
        raise AudioContractError(f"{manifest['subtype']} export round-trip validation failed "
                                 f"(absolute tolerance {tolerance:g}): {detail}.{remedy}")
    return {"method": "actual written aligned WAVs, read in blocks of at most 65536 frames",
            "absolute_tolerance": tolerance, "units": "decoded waveform amplitude",
            **errors}


def apply_peak_policy(project: Project, rendered: Samples, clipping: str) -> Samples:
    """Never change unselected samples, including over-unity float originals."""
    if clipping not in {"reject", "hard", "preserve"}:
        raise AudioContractError("Unsupported clipping policy")
    peak = float(np.max(np.abs(rendered)))
    if clipping == "reject" and peak > 1:
        raise AudioContractError(f"Rendered mix peak {peak:.6g} exceeds unity; select an explicit clipping policy")
    if clipping == "hard":
        cursor = 0
        for edit in sorted(project.edits, key=lambda row: row["start_sample"]):
            start, end = edit["start_sample"], edit["end_sample"]
            if start > cursor and np.any(np.abs(rendered[cursor:start]) > 1):
                raise AudioContractError("Hard clipping would change unedited original samples; choose preserve")
            cursor = max(cursor, end)
        if cursor < len(rendered) and np.any(np.abs(rendered[cursor:]) > 1):
            raise AudioContractError("Hard clipping would change unedited original samples; choose preserve")
        return np.clip(rendered, -1, 1)
    return rendered


def export_project(project: Project, destination: str | Path, *,
                   subtype: Literal["DOUBLE", "FLOAT"] = "DOUBLE",
                   clipping: Literal["reject", "hard", "preserve"] = "reject") -> Path:
    """Export without resampling or normalization; never overwrite a destination.

    DOUBLE preserves internal float64 arrays. FLOAT quantizes each exported array
    independently to float32. Stem/residual floats may exceed unity. ``reject``
    checks the rendered mix peak; ``hard`` explicitly clips the final mix only;
    ``preserve`` retains out-of-range floating samples without hidden limiting.
    """
    if subtype not in {"DOUBLE", "FLOAT"} or clipping not in {"reject", "hard", "preserve"}:
        raise AudioContractError("Unsupported export precision or clipping policy")
    target = Path(destination).absolute()
    if target.exists():
        raise FileExistsError(target)
    rendered = project.render()
    peak = float(np.max(np.abs(rendered)))
    final = apply_peak_policy(project, rendered, clipping)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        (temporary / "stems").mkdir()
        (temporary / "envelopes").mkdir()
        manifest: dict[str, Any] = {
            "schema": "stemscene-export/1", "project_id": project._manifest["id"],
            "revision": project.revision, "sample_rate": project.sample_rate,
            "frames": project.original.frames, "channels": project.original.channels,
            "subtype": subtype, "clipping": clipping, "pre_policy_mix_peak": peak,
            "clipping_locality": "hard clipping is rejected if it would alter any sample outside selected edit spans",
            "normalization": "none", "resampling": "none at export", "delay_samples": 0,
            "equation": "mix = original_decoded + sum((envelope - 1) * stem); apply final clipping policy",
            "unity_reconstruction": "original_decoded = residual + sum(stem), within floating point tolerance",
            "stem_interpretation": "estimated sound components; reconstruction is not proof of isolation",
            "precision_note": "DOUBLE float64" if subtype == "DOUBLE" else "FLOAT float32 quantization per array",
            "layers": [], "files": {}, "edit_history": project.edits,
        }

        def write_wav(relative: str, values: Samples | np.ndarray) -> None:
            if not np.isfinite(values).all() or (subtype == "FLOAT" and np.max(np.abs(values)) > np.finfo(np.float32).max):
                raise AudioContractError("Exported array exceeds the selected floating point precision")
            sf.write(temporary / relative, values, project.sample_rate, subtype=subtype, format="WAV")
            manifest["files"][relative] = sha256_file(temporary / relative)

        source_relative = "source" + Path(project._manifest["source"]["path"]).suffix
        preserved_source = _safe_path(project.root, project._manifest["source"]["path"])
        if sha256_file(preserved_source) != project.original.source_sha256:
            raise AudioContractError("Original source bytes changed since project load")
        shutil.copyfile(preserved_source, temporary / source_relative)
        if sha256_file(temporary / source_relative) != project.original.source_sha256:
            raise AudioContractError("Original source bytes changed during export")
        manifest["source"] = {"path": source_relative, **project.original.metadata()}
        manifest["files"][source_relative] = sha256_file(temporary / source_relative)
        write_wav("original-decoded.wav", project.original.samples)
        write_wav("residual.wav", project.residual())
        write_wav("mix.wav", final)
        for layer in project.layers:
            stem = f"stems/{layer.id}.wav"
            envelope_path = f"envelopes/{layer.id}.wav"
            recipe = f"envelopes/{layer.id}.json"
            write_wav(stem, layer.samples)
            write_wav(envelope_path, project.envelope(layer.id))
            operations = [edit for edit in project.edits if edit["layer_id"] == layer.id]
            _write_json(temporary / recipe, {
                "schema": "stemscene-envelope/1", "frames": project.original.frames,
                "sample_rate": project.sample_rate, "default_gain": 1.0,
                "combination": "multiply operations in listed order",
                "interval": "start_sample inclusive, end_sample exclusive",
                "fade_rule": "For F fade_samples: first F gains are linspace(1,gain,F+1)[:-1]; last F are reversed; middle equals gain",
                "operations": operations,
            })
            manifest["files"][recipe] = sha256_file(temporary / recipe)
            manifest["layers"].append({"id": layer.id, "query": layer.query,
                                       "stem": stem, "envelope": envelope_path,
                                       "envelope_recipe": recipe, "locked": layer.locked,
                                       "anchors": [vars(anchor) for anchor in layer.anchors],
                                       "metadata": layer.metadata})
        manifest["verification"] = _verify_written_audio(temporary, manifest, project.original.samples, final)
        _write_json(temporary / "manifest.json", manifest)
        if target.exists():
            raise FileExistsError(target)
        os.rename(temporary, target)
        return target / "manifest.json"
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def replay_export(destination: str | Path, *, verify_hashes: bool = True) -> Samples:
    """Recombine an exported recipe using WAVs only, without an inference backend."""
    root = Path(destination).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.stat().st_size > 32 * 1024 * 1024:
        raise AudioContractError("Export manifest exceeds 32 MiB")
    try:
        manifest = _read_json(manifest_path)
        if manifest["schema"] != "stemscene-export/1":
            raise AudioContractError("Unsupported export schema")
        shape = (manifest["frames"], manifest["channels"])
        if (any(type(v) is not int or v < 1 for v in shape) or shape[1] > 2
                or shape[0] * shape[1] * 8 > 512 * 1024 * 1024
                or len(manifest["layers"]) > 12):
            raise AudioContractError("Export exceeds replay bounds")
        if verify_hashes:
            if len(manifest["files"]) > 4 + 3 * len(manifest["layers"]):
                raise AudioContractError("Export contains excessive asset references")
            max_asset_bytes = max(32 * 1024 * 1024, shape[0] * shape[1] * 8 + 65536)
            for relative, expected in manifest["files"].items():
                if sha256_file(_safe_path(root, relative), max_bytes=max_asset_bytes) != expected:
                    raise AudioContractError("Export asset checksum mismatch")

        def read(relative: str, expected_shape: tuple[int, int]) -> Samples:
            path = _safe_path(root, relative)
            with sf.SoundFile(path) as stream:
                if (stream.frames, stream.channels) != expected_shape or stream.samplerate != manifest["sample_rate"]:
                    raise AudioContractError("Export WAV is not aligned")
                if stream.format not in {"WAV", "WAVEX", "RF64"}:
                    raise AudioContractError("Export replay expects WAV arrays")
                return validate_samples(stream.read(dtype="float64", always_2d=True), shape=expected_shape)

        output = read("original-decoded.wav", shape).copy()
        for row in manifest["layers"]:
            stem = read(row["stem"], shape)
            envelope = read(row["envelope"], (shape[0], 1))
            active = envelope[:, 0] != 1.0
            output[active] += (envelope[active] - 1.0) * stem[active]
        if manifest["clipping"] == "hard":
            output = np.clip(output, -1, 1)
        elif manifest["clipping"] not in {"reject", "preserve"}:
            raise AudioContractError("Unknown export clipping policy")
        return validate_samples(output, shape=shape)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise AudioContractError("Malformed export manifest") from exc
