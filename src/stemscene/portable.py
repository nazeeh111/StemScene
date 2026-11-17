"""Exact editable project payload inside a verified playback export."""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import tempfile

import numpy as np
import soundfile as sf

from .audio import AudioContractError, AudioLimits, sha256_file
from .project import Project, _read_json, _safe_path, _write_json
from .storage import publish_directory


def _identity(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _assets(project):
    assets = {project._manifest["source"]["path"]: project.original.source_sha256}
    for state in [project._manifest, *project.history]:
        for row in state["layers"]:
            if row["path"] in assets and assets[row["path"]] != row["sha256"]:
                raise AudioContractError("Conflicting portable asset identity")
            assets[row["path"]] = row["sha256"]
    return assets


def _copy(source, target, expected, maximum):
    if sha256_file(source, max_bytes=maximum) != expected:
        raise AudioContractError("Portable source asset changed")
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        while block := incoming.read(65536):
            count += len(block)
            if count > maximum:
                raise AudioContractError("Portable asset exceeds its byte limit")
            outgoing.write(block)
    if sha256_file(target, max_bytes=maximum) != expected:
        raise AudioContractError("Portable asset changed during copying")


def include_project(project, root, manifest):
    folder = root / "project"
    assets = _assets(project)
    if sum(_safe_path(project.root, name).stat().st_size for name in assets) > AudioLimits().max_project_array_bytes:
        raise AudioContractError("Exact portable project assets exceed2 GiB; keep the source project locally")
    for name, expected in assets.items():
        maximum = (project.limits.max_source_bytes if name == project._manifest["source"]["path"]
                   else project.original.samples.nbytes + 65536)
        target = _safe_path(folder, name)
        _copy(_safe_path(project.root, name), target, expected, maximum)
        manifest["files"]["project/" + name] = expected
    _write_json(folder / "project.json", project._manifest)
    manifest["files"]["project/project.json"] = sha256_file(folder / "project.json")
    Project.load(folder)
    manifest["editable_project"] = {"manifest": "project/project.json", "version": 1,
                                    "precision": "exact original source and float64 .npy estimates, including undo history"}


def reopen_export(source, destination):
    from .export import replay_export, apply_peak_policy
    root, target = Path(source).resolve(), Path(destination).absolute()
    if target.exists():
        raise FileExistsError(target)
    manifest = _read_json(root / "manifest.json")
    if _identity(manifest.get("editable_project")) != _identity({"manifest": "project/project.json", "version": 1,
                                            "precision": "exact original source and float64 .npy estimates, including undo history"}):
        raise AudioContractError("This playback-only export has no exact editable project; replay it or export again from a current StemScene project")
    replayed = replay_export(root)
    portable = _safe_path(root, "project")
    if manifest["files"].get("project/project.json") != sha256_file(portable / "project.json"):
        raise AudioContractError("Editable project manifest is not bound by export")
    project = Project.load(portable)
    if project._manifest["id"] != manifest["project_id"] or type(manifest["revision"]) is not int or project.revision != manifest["revision"] or _identity(project.edits) != _identity(manifest["edit_history"]):
        raise AudioContractError("Portable project identity/events disagree with playback export")
    expected_layers = [{"id": layer.id, "query": layer.query, "locked": layer.locked,
                        "anchors": [vars(a) for a in layer.anchors], "metadata": layer.metadata}
                       for layer in project.layers]
    if len(expected_layers) != len(manifest["layers"]):
        raise AudioContractError("Portable layer count disagrees with export")
    actual_layers = [{key: layer[key] for key in expected_layers[i]} for i, layer in enumerate(manifest["layers"])]
    if _identity(expected_layers) != _identity(actual_layers) or _identity({key: manifest["source"].get(key) for key in project.original.metadata()}) != _identity(project.original.metadata()):
        raise AudioContractError("Portable layer/source identity disagrees with export")
    if manifest["subtype"] not in {"DOUBLE", "FLOAT"} or type(manifest["sample_rate"]) is not int or manifest["sample_rate"] != project.sample_rate:
        raise AudioContractError("Portable playback precision/rate is invalid")
    final = apply_peak_policy(project, project.render(), manifest["clipping"])
    if sf.info(_safe_path(root, "mix.wav")).subtype != manifest["subtype"]:
        raise AudioContractError("Playback precision disagrees with export declaration")
    mix, rate = sf.read(_safe_path(root, "mix.wav"), dtype="float64", always_2d=True)
    tolerance = 1e-12 if manifest["subtype"] == "DOUBLE" else 1e-6
    if rate != project.sample_rate or mix.shape != final.shape or replayed.shape != final.shape or not np.isfinite(mix).all() or np.max(np.abs(mix - final)) > tolerance or np.max(np.abs(mix - replayed)) > tolerance:
        raise AudioContractError("Portable project cannot reproduce the exported playback mix")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for name, expected in _assets(project).items():
            relative = "project/" + name
            if manifest["files"].get(relative) != expected:
                raise AudioContractError("Portable asset is not bound by export manifest")
            maximum = project.limits.max_source_bytes if name == project._manifest["source"]["path"] else project.original.samples.nbytes + 65536
            _copy(_safe_path(portable, name), _safe_path(temporary, name), expected, maximum)
        document = deepcopy(project._manifest)
        document["reopened_from_export"] = {"manifest_sha256": sha256_file(root / "manifest.json"),
                                             "clipping": manifest["clipping"], "subtype": manifest["subtype"],
                                             "note": "Edits preserve the pre-policy floating mix; select the recorded peak policy on subsequent export"}
        _write_json(temporary / "project.json", document)
        Project.load(temporary)
        publish_directory(temporary, target)
        return Project.load(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
