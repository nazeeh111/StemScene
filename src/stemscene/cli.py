"""Installed CLI for the same persistent engine used by the local browser service."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence
from uuid import uuid4

import numpy as np
import soundfile as sf

from . import __version__
from .audio import AudioContractError, Samples
from .export import export_project, replay_export
from .project import Anchor, Project
from .models import DownloadCancelled


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stemscene", description="Local estimated sound layers and reversible recording edits.")
    parser.add_argument("--version", action="version", version=f"stemscene {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Preserve an authorized WAV/FLAC recording in a new project")
    init.add_argument("source", type=Path)
    init.add_argument("project", type=Path)
    inspect = commands.add_parser("inspect", help="Print project timing, layers, anchors and edits as JSON")
    inspect.add_argument("project", type=Path)
    extract = commands.add_parser("extract", help="Compute local AudioSep proposals and apply the trained joint refiner")
    extract.add_argument("project", type=Path)
    extract.add_argument("--query", action="append", required=True, help="Requested sound description; repeat for a joint request")
    extract.add_argument("--checkpoint", type=Path, required=True)
    extract.add_argument("--tokenizer-dir", type=Path, required=True)
    extract.add_argument("--refiner", type=Path, help="Explicit trained refiner artifact; otherwise use a bundled artifact if present")
    extract.add_argument("--device", choices=("cpu", "mps"), default="cpu", help="MPS remains experimental")
    extract.add_argument("--threads", type=int, choices=range(1, 11), default=4)
    edit = commands.add_parser("edit", help="Multiply one estimated layer's gain inside a selected time span")
    edit.add_argument("project", type=Path)
    edit.add_argument("layer_id")
    edit.add_argument("--start", type=float, required=True, help="Start in seconds, inclusive")
    edit.add_argument("--end", type=float, required=True, help="End in seconds, exclusive")
    edit.add_argument("--gain", type=float, required=True, help="Linear amplitude gain from 0 to 4")
    edit.add_argument("--fade", type=float, default=.005, help="Fade duration in seconds, inside the selected span")
    edit.add_argument("--replace", help="ID of a saved event to update instead of multiplying another operation")
    remove = commands.add_parser("remove-edit", help="Remove one saved gain event, with undo")
    remove.add_argument("project", type=Path)
    remove.add_argument("edit_id")
    for name, help_text in (("lock", "Keep a stem unchanged by subsequent extraction"), ("unlock", "Allow subsequent extraction to refine a stem")):
        item = commands.add_parser(name, help=help_text)
        item.add_argument("project", type=Path)
        item.add_argument("layer_id")
    anchors = commands.add_parser("anchors", help="Replace mixed-recording present/absent anchors on one layer")
    anchors.add_argument("project", type=Path)
    anchors.add_argument("layer_id")
    anchors.add_argument("--present", type=float, nargs=2, action="append", metavar=("START", "END"), default=[])
    anchors.add_argument("--absent", type=float, nargs=2, action="append", metavar=("START", "END"), default=[])
    undo = commands.add_parser("undo", help="Restore the preceding persistent project revision")
    undo.add_argument("project", type=Path)
    render = commands.add_parser("render", help="Render edits to a new aligned DOUBLE WAV")
    render.add_argument("project", type=Path)
    render.add_argument("output", type=Path)
    render.add_argument("--clipping", choices=("reject", "hard", "preserve"), default="reject")
    export = commands.add_parser("export", help="Write verified stems, residual, envelopes and reproducible mix")
    export.add_argument("project", type=Path)
    export.add_argument("destination", type=Path)
    export.add_argument("--subtype", choices=("DOUBLE", "FLOAT"), default="DOUBLE")
    export.add_argument("--clipping", choices=("reject", "hard", "preserve"), default="reject")
    replay = commands.add_parser("replay", help="Recombine an export without loading any inference model")
    replay.add_argument("export", type=Path)
    replay.add_argument("output", type=Path)
    serve = commands.add_parser("serve", help="Run the local browser mixer on loopback only")
    serve.add_argument("--workspace", type=Path, required=True)
    serve.add_argument("--checkpoint", type=Path)
    serve.add_argument("--tokenizer-dir", type=Path)
    serve.add_argument("--refiner", type=Path)
    serve.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    serve.add_argument("--port", type=int, default=8765, help="Loopback port; 0 selects an available port")
    models = commands.add_parser("models", help="Explicit public model setup; normal inference stays offline")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    download = model_commands.add_parser("download", help="Download approximately 1.27 GB of pinned free public model assets",
                                          description="Download approximately 1.27 GB of pinned free public model assets into a new directory.")
    download.add_argument("directory", type=Path)
    download.add_argument("--timeout", type=float, default=30, help="Per-network-operation timeout in seconds (maximum 120)")
    return parser


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, allow_nan=False))


def _inspect(project: Project) -> dict[str, Any]:
    return {"project": str(project.root), "revision": project.revision,
            "source": project.original.metadata(), "duration_seconds": project.original.duration,
            "layers": [{"id": layer.id, "query": layer.query, "locked": layer.locked,
                        "anchors": [asdict(anchor) for anchor in layer.anchors],
                        "interpretation": "estimated sound class; isolation is uncertain"}
                       for layer in project.layers],
            "edits": project.edits, "undo_steps": len(project.history)}


def _refiner_path(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise ValueError(f"Trained refiner artifact does not exist: {explicit}")
        return explicit
    bundled = Path(__file__).parent / "artifacts" / "joint-refiner-v1.pt"
    if bundled.is_file():
        return bundled
    raise ValueError("A trained refiner is required: pass --refiner PATH. No bundled trained artifact is installed.")


def _write_new_wav(path: Path, samples: Samples, sample_rate: int, clipping: str = "preserve") -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"Output already exists: {path}")
    if clipping == "reject" and np.max(np.abs(samples)) > 1:
        raise AudioContractError("Rendered mix exceeds unity; select an explicit clipping policy")
    output = np.clip(samples, -1, 1) if clipping == "hard" else samples
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        sf.write(temporary, output, sample_rate, subtype="DOUBLE", format="WAV")
        # Hard-link creation is atomic and refuses an existing output, including
        # a destination which appeared after the initial check.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "models":
            from .models import download_models
            previous_asset: list[str | None] = [None]

            def progress(name: str, received: int, total: int) -> None:
                if previous_asset[0] != name:
                    print(f"Downloading {name} ({total} bytes)", file=sys.stderr, flush=True)
                    previous_asset[0] = name

            directory = download_models(args.directory, timeout=args.timeout, progress=progress)
            _print({"models": str(directory), "checkpoint": str(directory / "audiosep_base_4M_steps.ckpt"),
                    "tokenizer_directory": str(directory / "roberta")})
            return 0
        if args.command == "serve":
            from .server import serve
            if not 0 <= args.port <= 65535:
                raise ValueError("Port must be between 0 and 65535 (0 selects an available port)")
            artifact = _refiner_path(args.refiner) if args.checkpoint and args.tokenizer_dir else args.refiner
            serve(workspace=args.workspace, checkpoint=args.checkpoint,
                  tokenizer_directory=args.tokenizer_dir, refiner_path=artifact,
                  device=args.device, port=args.port)
            return 0
        if args.command == "init":
            _print(_inspect(Project.create(args.source, args.project)))
            return 0
        if args.command == "replay":
            from .project import _read_json
            manifest = _read_json(args.export / "manifest.json")
            _write_new_wav(args.output, replay_export(args.export), manifest["sample_rate"])
            _print({"output": str(args.output.absolute()), "sample_rate": manifest["sample_rate"]})
            return 0
        project = Project.load(args.project)
        if args.command == "inspect":
            _print(_inspect(project))
        elif args.command == "extract":
            artifact = _refiner_path(args.refiner)
            from .backend import AudioSepBackend, BackendLimits
            from .engine import SceneEngine
            from .joint import JointRefiner
            refiner = JointRefiner.load(artifact)
            backend = AudioSepBackend(args.checkpoint, args.tokenizer_dir,
                                      device=args.device, limits=BackendLimits(threads=args.threads))
            engine = SceneEngine(backend, refiner)
            layers = engine.extract(project, args.query, progress=lambda done, total:
                                    print(f"Computed proposal {done}/{total}", file=sys.stderr, flush=True))
            _print({"project": str(project.root), "revision": project.revision, "layers": layers})
        elif args.command == "edit":
            project.edit(args.layer_id, args.start, args.end, args.gain, args.fade, edit_id=args.replace)
            _print({"revision": project.revision, "layer_id": args.layer_id})
        elif args.command == "remove-edit":
            project.remove_edit(args.edit_id)
            _print({"revision": project.revision, "removed_edit": args.edit_id})
        elif args.command in {"lock", "unlock"}:
            project.set_locked(args.layer_id, args.command == "lock")
            _print({"revision": project.revision, "layer_id": args.layer_id, "locked": args.command == "lock"})
        elif args.command == "anchors":
            anchors = [Anchor(start, end, True) for start, end in args.present]
            anchors += [Anchor(start, end, False) for start, end in args.absent]
            project.set_anchors(args.layer_id, anchors)
            _print({"revision": project.revision, "layer_id": args.layer_id,
                    "anchors": [asdict(anchor) for anchor in anchors]})
        elif args.command == "undo":
            changed = project.undo()
            _print({"revision": project.revision, "changed": changed})
        elif args.command == "render":
            from .export import apply_peak_policy
            _write_new_wav(args.output, apply_peak_policy(project, project.render(), args.clipping), project.sample_rate)
            _print({"output": str(args.output.absolute()), "sample_rate": project.sample_rate})
        elif args.command == "export":
            manifest = export_project(project, args.destination, subtype=args.subtype, clipping=args.clipping)
            _print({"manifest": str(manifest)})
        return 0
    except DownloadCancelled as exc:
        print(f"stemscene: {exc}", file=sys.stderr)
        return 130
    except KeyboardInterrupt:
        print("stemscene: cancelled", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"stemscene: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
