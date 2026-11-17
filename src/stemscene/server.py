"""Loopback-only browser adapter to the installed project and inference engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit
import hmac
import os
import stat
import io
import json
import secrets
import shutil
import tempfile
from uuid import uuid4
from weakref import WeakValueDictionary
import zipfile

import numpy as np
import soundfile as sf

from .audio import AudioContractError, AudioLimits
from .backend import AudioSepBackend, BackendLimits, InferenceCancelled
from .export import export_project
from .project import Anchor, Project, _safe_path
from .session import ProjectSession
from .portable import reopen_export


@dataclass
class BrowserSession:
    workspace: Path
    checkpoint: Path | None = None
    tokenizer_directory: Path | None = None
    refiner_path: Path | None = None
    device: str = "cpu"
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    jobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    exports: dict[str, Path] = field(default_factory=dict)
    engine: Any = None
    sessions: WeakValueDictionary[str, ProjectSession] = field(default_factory=WeakValueDictionary)
    guard: Lock = field(default_factory=Lock)

    def __post_init__(self) -> None:
        self.workspace = self.workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        _safe_path(self.workspace, "projects").mkdir(exist_ok=True)
        _safe_path(self.workspace, "exports").mkdir(exist_ok=True)

    @property
    def configured(self) -> bool:
        return bool(self.checkpoint and self.checkpoint.is_file()
                    and self.tokenizer_directory and self.tokenizer_directory.is_dir()
                    and self.refiner_path and self.refiner_path.is_file())

    def project_session(self, identifier: str) -> ProjectSession:
        if len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
            raise AudioContractError("Unknown project")
        with self.guard:
            owner = self.sessions.get(identifier)
            if owner is None:
                owner = ProjectSession(_safe_path(self.workspace, "projects/" + identifier))
                self.sessions[identifier] = owner
            return owner

    def state(self, identifier: str) -> dict[str, Any]:
        with self.project_session(identifier).read() as project:
            return self._state(identifier, project)

    def _state(self, identifier, project):
        x = project.original.samples
        edges = np.linspace(0, len(x), min(1800, len(x))+1, dtype=int)
        peaks = [{"min": float(x[a:b].min()), "max": float(x[a:b].max())}
                 for a, b in zip(edges[:-1], edges[1:]) if b > a]
        title_path = _safe_path(project.root, "browser-title.json")
        title = identifier[:8]
        if title_path.is_file() and title_path.stat().st_size < 2048:
            title = str(json.loads(title_path.read_text()).get("title", title))[:200]
        return {"id": identifier, "title": title, "revision": project.revision,
                "export_origin": project.export_origin, "sample_rate": project.sample_rate, "frames": project.original.frames,
                "channels": project.original.channels, "duration": project.original.duration,
                "waveform": peaks, "layers": [
                    {"id": layer.id, "query": layer.query, "locked": layer.locked,
                     "anchors": [vars(a) for a in layer.anchors], "metadata": layer.metadata}
                    for layer in project.layers], "edits": project.edits}

    def start(self, identifier: str, queries: list[str], expected_revision=None) -> str:
        if not self.configured:
            raise AudioContractError("Configure a local separator and trained refiner before extracting layers")
        if (not isinstance(queries, list) or not queries or len(queries) > 8
                or any(not isinstance(q, str) or not q.strip() or len(q) > 512 for q in queries)):
            raise AudioContractError("Request one to eight sound descriptions, at most512 characters each")
        project_session = self.project_session(identifier)
        with project_session.read(expected_revision) as project:
            if project.original.channels != 1:
                raise AudioContractError("Learned extraction currently supports mono recordings only")
        with self.guard:
            if any(job["status"] == "running" for job in self.jobs.values()):
                raise AudioContractError("Another local extraction is running; cancel it or wait")
            token = project_session.begin_extraction(expected_revision)
            while len(self.jobs) >= 32:
                self.jobs.pop(next(iter(self.jobs)))
            key = uuid4().hex
            self.jobs[key] = {"status": "running", "completed": 0, "total": len(queries),
                              "project_id": identifier, "operation": token, "error": None}

        def run() -> None:
            try:
                if self.engine is None:
                    try:
                        from .engine import SceneEngine
                        from .joint import JointRefiner
                    except ImportError as exc:
                        raise AudioContractError("Extraction needs the optional local model runtime; saved-project editing and replay remain available") from exc
                    backend = AudioSepBackend(self.checkpoint, self.tokenizer_directory,
                                              limits=BackendLimits(threads=4), device=self.device)
                    self.engine = SceneEngine(backend, JointRefiner.load(self.refiner_path))
                job = self.jobs[key]
                project_session.compute(token, self.engine, queries,
                                        progress=lambda done, total: job.update(completed=done, total=total))
                job["status"] = "done"
            except InferenceCancelled:
                self.jobs[key]["status"] = "cancelled"
            except Exception as error:
                self.jobs[key].update(status="error", error=str(error)[:2000])
            finally:
                project_session.abandon(token)
        try:
            Thread(target=run, name="stemscene-inference", daemon=True).start()
        except BaseException:
            project_session.abandon(token)
            self.jobs[key].update(status="error", error="Could not start the local worker")
            raise
        return key


def make_server(session: BrowserSession, port: int = 8765) -> ThreadingHTTPServer:
    """Return a loopback server; useful for actual HTTP integration tests."""
    assets = Path(__file__).parent / "web"

    class Handler(BaseHTTPRequestHandler):
        server_version = "STEMSCENE"

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, *_: Any) -> None:
            # No file names, request bodies or local session tokens in access logs.
            pass

        def headers_for(self, status: int, mime: str, length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")

        def send_bytes(self, data: bytes, mime: str, status: int = 200) -> None:
            self.headers_for(status, mime, len(data)); self.end_headers(); self.wfile.write(data)

        def send_json(self, value: Any, status: int = 200) -> None:
            self.send_bytes(json.dumps(value, allow_nan=False).encode(), "application/json; charset=utf-8", status)

        def allowed(self, *, api: bool) -> bool:
            expected = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != expected:
                self.send_json({"error": "Use the printed 127.0.0.1 address"}, 403); return False
            origin = self.headers.get("Origin")
            if origin is not None and origin != "http://"+expected:
                self.send_json({"error": "Cross-origin requests are not allowed"}, 403); return False
            if api and not hmac.compare_digest(self.headers.get("X-StemScene-Token", ""), session.token):
                self.send_json({"error": "Reload the local mixer to renew its session"}, 403); return False
            return True

        def body_length(self, maximum: int) -> int:
            if self.headers.get("Transfer-Encoding"):
                raise AudioContractError("Chunked request bodies are not supported")
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError as exc:
                raise AudioContractError("Invalid request size") from exc
            if not 0 <= length <= maximum:
                raise AudioContractError("Request exceeds the local size limit")
            return length

        def read_json(self) -> dict[str, Any]:
            if self.headers.get_content_type() != "application/json":
                raise AudioContractError("Expected a JSON request")
            length = self.body_length(1024*1024)
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise AudioContractError("Incomplete request")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise AudioContractError("Expected a JSON object")
            return value

        def do_GET(self) -> None:
            path = urlsplit(self.path)
            if not self.allowed(api=path.path.startswith("/api/")):
                return
            query = parse_qs(path.query)
            identifier = query.get("id", [""])[0]
            try:
                if path.path == "/":
                    source = (assets/"index.html").read_text().replace("__SESSION_TOKEN__", session.token)
                    return self.send_bytes(source.encode(), "text/html; charset=utf-8")
                if path.path in {"/app.js", "/styles.css"}:
                    return self.send_bytes((assets/path.path[1:]).read_bytes(),
                                           "text/javascript; charset=utf-8" if path.path.endswith(".js") else "text/css; charset=utf-8")
                if path.path == "/api/config":
                    return self.send_json({"model_ready": session.configured,
                        "model_message": "Local files configured. First extraction loads the model." if session.configured else "Saved editing and export are ready. New layer estimates need local AudioSep and trained refiner paths.",
                        "limits": {"max_seconds": 600, "max_queries": 8, "inference_channels": 1,
                                   "max_source_bytes": AudioLimits().max_source_bytes,
                                   "processing": "offline; CPU or tested MPS with CPU fallback"}})
                if path.path == "/api/projects":
                    rows = []
                    for item in sorted((session.workspace/"projects").iterdir())[:1000]:
                        if item.is_dir() and len(item.name) == 32:
                            try:
                                state = session.state(item.name)
                                rows.append({"id": item.name, "title": state["title"]})
                            except (ValueError, OSError, KeyError):
                                continue
                    return self.send_json({"projects": rows})
                if path.path == "/api/state":
                    if query.get("reload", [""])[0] == "1":
                        session.project_session(identifier).reload()
                    return self.send_json(session.state(identifier))
                if path.path == "/api/job":
                    job = session.jobs.get(identifier)
                    if job is None:
                        raise AudioContractError("Unknown extraction")
                    return self.send_json({k: v for k, v in job.items() if k != "cancel"})
                if path.path == "/api/audio":
                    with session.project_session(identifier).read() as project:
                        return self.send_audio(project, query)
                if path.path == "/api/download":
                    archive = session.exports.get(identifier)
                    if archive is None:
                        raise AudioContractError("Unknown export")
                    self.headers_for(200, "application/zip", archive.stat().st_size)
                    self.send_header("Content-Disposition", f'attachment; filename="stemscene-{identifier}.zip"')
                    self.end_headers()
                    with archive.open("rb") as stream:
                        shutil.copyfileobj(stream, self.wfile, length=65536)
                    return
                self.send_json({"error": "Not found"}, 404)
            except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile, RuntimeError) as error:
                self.send_json({"error": str(error)[:2000]}, 400)

        def send_audio(self, project, query):
            kind = query.get("kind", ["mix"])[0]
            if kind == "original":
                audio = project.original.samples
            elif kind == "mix":
                audio = project.render()
            elif kind == "stem":
                layer_id = query.get("layer_id", [""])[0]
                matches = [layer.samples for layer in project.layers if layer.id == layer_id]
                if not matches:
                    raise AudioContractError("Unknown layer")
                audio = matches[0]
            else:
                raise AudioContractError("Unknown playback source")
            if audio.nbytes > 256*1024*1024:
                raise AudioContractError("Recording exceeds browser preview size; use CLI export")
            buffer = io.BytesIO(); sf.write(buffer, audio, project.sample_rate, format="WAV", subtype="FLOAT")
            return self.send_bytes(buffer.getvalue(), "audio/wav")
        def do_POST(self) -> None:
            if not self.allowed(api=True):
                return
            path = urlsplit(self.path).path
            try:
                if path == "/api/import":
                    if self.headers.get_content_type() != "application/octet-stream":
                        raise AudioContractError("Import expects the recording bytes")
                    length = self.body_length(AudioLimits().max_source_bytes)
                    name = Path(unquote(self.headers.get("X-Filename", "recording.wav"))).name[:200]
                    suffix = Path(name).suffix.lower()
                    if suffix not in {".wav", ".flac"}:
                        raise AudioContractError("Choose a WAV or FLAC recording")
                    identifier = uuid4().hex
                    with tempfile.TemporaryDirectory(prefix="stemscene-import-") as temp:
                        source = Path(temp)/("input"+suffix)
                        with source.open("xb") as output:
                            remaining = length
                            while remaining:
                                block = self.rfile.read(min(65536, remaining))
                                if not block:
                                    raise AudioContractError("Incomplete recording upload")
                                output.write(block); remaining -= len(block)
                        project = Project.create(source, session.workspace/"projects"/identifier)
                        (project.root/"browser-title.json").write_text(json.dumps({"title": name}))
                    return self.send_json(session.state(identifier))
                if path == "/api/reopen":
                    if self.headers.get_content_type() != "application/zip":
                        raise AudioContractError("Reopen expects an exported ZIP bundle")
                    length = self.body_length(512 * 1024 * 1024)
                    identifier = uuid4().hex
                    with tempfile.TemporaryDirectory(prefix="stemscene-reopen-") as folder:
                        root = Path(folder)
                        archive = root / "bundle.zip"
                        with archive.open("xb") as stream:
                            remaining = length
                            while remaining:
                                block = self.rfile.read(min(65536, remaining))
                                if not block:
                                    raise AudioContractError("Incomplete bundle upload")
                                stream.write(block)
                                remaining -= len(block)
                        unpacked = root / "unpacked"
                        unpacked.mkdir()
                        with zipfile.ZipFile(archive) as bundle:
                            entries = bundle.infolist()
                            if len(entries) > 1300 or sum(item.file_size for item in entries) > 512 * 1024 * 1024:
                                raise AudioContractError("Portable bundle exceeds browser bounds")
                            seen = set()
                            for item in entries:
                                if item.filename in seen or item.is_dir() or "\\" in item.filename or stat.S_IFMT(item.external_attr >> 16) not in (0, stat.S_IFREG):
                                    raise AudioContractError("Bundle entries must be unique ordinary files")
                                seen.add(item.filename)
                                target = _safe_path(unpacked, item.filename)
                                target.parent.mkdir(parents=True, exist_ok=True)
                                count = 0
                                with bundle.open(item) as incoming, target.open("xb") as output:
                                    while block := incoming.read(65536):
                                        count += len(block)
                                        if count > item.file_size:
                                            raise AudioContractError("Bundle entry exceeds declared size")
                                        output.write(block)
                        reopened = reopen_export(unpacked, session.workspace / "projects" / identifier)
                        (reopened.root / "browser-title.json").write_text(json.dumps({"title": "Reopened export " + identifier[:8]}))
                    return self.send_json(session.state(identifier))
                body = self.read_json()
                identifier = body.get("id", "")
                if path == "/api/extract":
                    if type(body.get("revision")) is not int:
                        raise AudioContractError("Reload the project; a current revision is required")
                    return self.send_json({"job_id": session.start(identifier, body.get("queries", []), body.get("revision"))})
                if path == "/api/cancel":
                    job = session.jobs.get(body.get("job_id", ""))
                    if job is None:
                        raise AudioContractError("Unknown extraction")
                    if job["status"] == "running":
                        session.project_session(job["project_id"]).cancel(job["operation"])
                    return self.send_json({"status": "cancellation_requested" if job["status"] == "running" else job["status"]})
                project_session = session.project_session(identifier)
                if type(body.get("revision")) is not int:
                    raise AudioContractError("Reload the project; a current revision is required")
                revision = body["revision"]
                if path == "/api/edit":
                    project_session.change("edit", body["layer_id"], float(body["start"]), float(body["end"]),
                                           float(body["gain"]), float(body.get("fade_seconds", .005)),
                                           edit_id=body.get("edit_id"), expected_revision=revision)
                elif path == "/api/remove-edit":
                    project_session.change("remove_edit", body["edit_id"], expected_revision=revision)
                elif path == "/api/lock":
                    project_session.change("set_locked", body["layer_id"], body["locked"], expected_revision=revision)
                elif path == "/api/anchors":
                    project_session.change("set_anchors", body["layer_id"], [Anchor(**a) for a in body["anchors"]], expected_revision=revision)
                elif path == "/api/undo":
                    project_session.change("undo", expected_revision=revision)
                elif path == "/api/export":
                    key = uuid4().hex
                    target = session.workspace / "exports" / key
                    manifest = project_session.export(target, expected_revision=revision,
                                                      subtype=body.get("subtype", "DOUBLE"), clipping=body.get("clipping", "reject"))
                    document = json.loads(manifest.read_text())
                    members = ["manifest.json", *document["files"]]
                    if sum(_safe_path(target, member).stat().st_size for member in members) > 512 * 1024 * 1024:
                        raise AudioContractError("Export is saved locally but exceeds the browser512 MiB limit; use CLI export")
                    archive = target.with_suffix(".zip")
                    staged = archive.with_name("." + archive.name + ".tmp")
                    try:
                        with zipfile.ZipFile(staged, "x", compression=zipfile.ZIP_STORED) as bundle:
                            for member in members:
                                bundle.write(_safe_path(target, member), arcname=member)
                        os.link(staged, archive)
                    finally:
                        staged.unlink(missing_ok=True)
                    session.exports[key] = archive
                    return self.send_json({"export_id": key, "filename": f"stemscene-{key}.zip", "revision": document["revision"]})
                else:
                    return self.send_json({"error": "Not found"}, 404)
                self.send_json(session.state(identifier))
            except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile, RuntimeError) as error:
                self.send_json({"error": str(error)[:2000]}, 400)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(workspace: Path, checkpoint: Path | None = None,
          tokenizer_directory: Path | None = None, refiner_path: Path | None = None,
          device: str = "cpu", port: int = 8765) -> None:
    session = BrowserSession(workspace, checkpoint, tokenizer_directory, refiner_path, device)
    server = make_server(session, port)
    print(f"STEMSCENE local mixer: http://127.0.0.1:{server.server_port}/", flush=True)
    print("Only this computer. Recording bytes stay in the selected workspace.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for job in session.jobs.values():
            session.project_session(job["project_id"]).cancel(job["operation"])
        server.server_close()
