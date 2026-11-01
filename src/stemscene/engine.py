"""The installed engine shared by CLI and local browser service."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence
import hashlib
import json
import os
import stat
import time
from uuid import uuid4

import numpy as np

from .audio import AudioContractError, sha256_file
from .backend import AudioSepBackend, InferenceCancelled, MODEL_RATE, resample_aligned
from .joint import JointRefiner, RefinerAnchor
from .project import Project, LockedLayerError, normalize_query, _safe_path


class AnchorNeededError(AudioContractError):
    """Two indistinguishable descriptions would overlap a locked estimate."""


class SceneEngine:
    def __init__(self, backend: AudioSepBackend, refiner: JointRefiner):
        self.backend, self.refiner = backend, refiner

    def _proposal(self, project: Project, query: str,
                  cancelled: Callable[[], bool] | None) -> np.ndarray:
        settings = {"source": project.original.source_sha256, "query": query,
                    "backend": self.backend.checkpoint_sha256,
                    "chunk": self.backend.limits.chunk_seconds,
                    "overlap": self.backend.limits.overlap_seconds,
                    "device": self.backend.device, "version": "audiosep-proposal-v1"}
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        raw_cache = project.root / "proposal-cache"
        if raw_cache.is_symlink():
            raise AudioContractError("proposal cache must be an ordinary project directory")
        cache = _safe_path(project.root, "proposal-cache")
        cache.mkdir(exist_ok=True)
        raw_path, raw_digest = cache/f"{key}.npy", cache/f"{key}.sha256"
        if raw_path.is_symlink() or raw_digest.is_symlink():
            raise AudioContractError("proposal cache assets must be regular files")
        path = _safe_path(project.root, f"proposal-cache/{key}.npy")
        digest_path = _safe_path(project.root, f"proposal-cache/{key}.sha256")
        if path.exists() and digest_path.exists():
            descriptor = os.open(digest_path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > 65:
                    raise AudioContractError("cached checksum must be a bounded regular file")
                expected = stream.read(66).decode("ascii").strip()
            if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                raise AudioContractError("cached checksum is malformed")
            if sha256_file(path, max_bytes=project.original.samples.nbytes+4096) != expected:
                raise AudioContractError("cached proposal checksum mismatch")
            # Mapping validates physical file length before any declared-shape
            # allocation. Inspect shape/dtype before creating an owned copy.
            output = np.load(path, mmap_mode="r", allow_pickle=False)
            if (output.shape != project.original.samples.shape or output.dtype != np.float64
                    or not np.isfinite(output).all()):
                raise AudioContractError("cached proposal is malformed")
            return np.array(output, copy=True)
        output = self.backend.separate(project.original.samples, project.sample_rate,
                                       [query], cancelled=cancelled)[0]
        temporary = path.with_name(f".{key}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                np.save(stream, output, allow_pickle=False)
            os.replace(temporary, path)
            digest_temporary = digest_path.with_name(f".{key}.{uuid4().hex}.sha256.tmp")
            try:
                with digest_temporary.open("x") as stream:
                    stream.write(sha256_file(path)+"\n")
                os.replace(digest_temporary, digest_path)
            finally:
                if digest_temporary.exists():
                    digest_temporary.unlink()
        finally:
            if temporary.exists():
                temporary.unlink()
        return output

    def extract(self, project: Project, queries: Sequence[str], *,
                cancelled: Callable[[], bool] | None = None,
                progress: Callable[[int, int], None] | None = None) -> dict[str, str]:
        """Compute against original audio, jointly refine, commit once.

        Existing locked arrays are never passed back to mutation with changed
        bytes. Exact duplicate requests reuse the same stable layer identity.
        """
        if not queries:
            raise AudioContractError("request at least one sound layer")
        started = time.monotonic()
        external_cancelled = cancelled
        budget = getattr(getattr(self.backend, "limits", None), "timeout_seconds", 3600)
        def operation_cancelled() -> bool:
            if time.monotonic()-started > budget:
                raise TimeoutError("whole extraction exceeded its inference time budget")
            return external_cancelled is not None and external_cancelled()
        cancelled = operation_cancelled
        if cancelled is not None and cancelled():
            raise InferenceCancelled("cancelled before query encoding")
        existing = {normalize_query(layer.query): layer for layer in project.layers}
        requested = [normalize_query(q) for q in queries]
        canonical = sorted(set(existing) | set(requested))
        if len(canonical) > 8:
            raise AudioContractError("joint extraction supports at most eight distinct classes")
        if project.original.channels != 1:
            raise AudioContractError("current learned engine supports mono; core stereo edits/export are separate")
        for query, layer in existing.items():
            if not layer.locked:
                continue
            prior_queries = layer.metadata.get("query_set", [])
            prior_refinement = layer.metadata.get("refinement", {})
            prior = []
            if isinstance(prior_queries, list) and query in prior_queries and isinstance(prior_refinement, dict):
                index = prior_queries.index(query)
                prior = [(a.get("start"), a.get("end"), a.get("present"))
                         for a in prior_refinement.get("anchors", [])
                         if isinstance(a, dict) and a.get("query_index") == index]
            requested_anchors = [(a.start, a.end, a.present) for a in layer.anchors]
            if requested_anchors != prior:
                raise LockedLayerError(f"Anchors for locked '{query}' have changed. Unlock that estimate before applying them; its saved samples are preserved.")
        embeddings = self.backend.embed_queries(canonical)
        similarity = embeddings @ embeddings.T
        for i, query in enumerate(canonical):
            if query in existing:
                continue
            for j, other in enumerate(canonical):
                layer = existing.get(other)
                if layer and layer.locked and similarity[i, j] >= self.refiner.config.ambiguity_cosine:
                    raise AnchorNeededError(f"'{query}' overlaps locked '{other}'. Group the request or unlock it and supply a useful mixed-recording anchor.")
        proposals = []
        anchors = []
        for index, query in enumerate(canonical):
            if cancelled is not None and cancelled():
                raise InferenceCancelled("cancelled before proposal computation")
            proposals.append(self._proposal(project, query, cancelled))
            if query in existing:
                anchors.extend(RefinerAnchor(index, a.start, a.end, a.present)
                               for a in existing[query].anchors)
            if progress:
                progress(index+1, len(canonical))
        if cancelled is not None and cancelled():
            raise InferenceCancelled("cancelled before joint refinement")
        model_mix = resample_aligned(project.original.samples, project.sample_rate, MODEL_RATE)
        model_proposals = np.stack([resample_aligned(p, project.sample_rate, MODEL_RATE)
                                    for p in proposals])
        try:
            result = self.refiner.refine_long(model_mix, MODEL_RATE, model_proposals, embeddings, anchors,
                                              cancelled=cancelled)
        except InterruptedError as error:
            raise InferenceCancelled(str(error)) from error
        if cancelled is not None and cancelled():
            raise InferenceCancelled("cancelled before project mutation")
        estimates = {q: (existing[q].samples if q in existing and existing[q].locked
                         else resample_aligned(result.samples[i], MODEL_RATE, project.sample_rate,
                                               project.original.frames)) for i, q in enumerate(canonical)}
        ids = project.add_estimates(estimates, metadata={
            "backend": "AudioSep", "checkpoint_sha256": self.backend.checkpoint_sha256,
            "query_set": canonical,
            "refiner_sha256": getattr(self.refiner, "artifact_sha256", None),
            "refiner_method": self.refiner.config.version,
            "device": self.backend.device,
            "processing_seconds_excluding_model_load": time.monotonic()-started,
            "refinement": result.diagnostics,
            "resampling": "zero-phase polyphase FIR to/from 32k; length right-trim/pad; no fitted delay",
            "chunk_seconds": self.backend.limits.chunk_seconds,
            "overlap_seconds": self.backend.limits.overlap_seconds,
            "stereo": "unsupported by learned engine", "context": "offline full clip",
            "locks": "locked arrays preserved byte-for-byte; residual carries unresolved energy"})
        return {q: ids[normalize_query(q)] for q in queries}
