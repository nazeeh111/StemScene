"""Admission/cache/cancel fixtures, not evidence of separation quality."""
from types import SimpleNamespace
import tracemalloc

import numpy as np
import pytest
import soundfile as sf

from stemscene import Project
from stemscene.audio import sha256_file
from stemscene.backend import AudioSepBackend, BackendLimits, InferenceCancelled
from stemscene.engine import SceneEngine


def test_pre_cancelled_request_never_encodes_or_mutates(tmp_path):
    source = tmp_path / "source.wav"
    sf.write(source, np.full((32, 1), .1), 8000, subtype="DOUBLE")
    project = Project.create(source, tmp_path / "project")
    engine = SceneEngine(SimpleNamespace(), SimpleNamespace())
    with pytest.raises(InferenceCancelled):
        engine.extract(project, ["rain"], cancelled=lambda: True)
    assert Project.load(project.root).revision == 0
    assert not list(project.root.glob("proposal-cache/*"))


def test_backend_rejects_huge_view_before_float64_allocation():
    backend = AudioSepBackend.__new__(AudioSepBackend)
    backend._torch = None
    backend.limits = BackendLimits()
    oversized = np.broadcast_to(np.zeros((1, 1), dtype=np.float32), (100_000_000, 1))
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="bounds"):
            backend.separate(oversized, 192000, ["rain"])
        assert tracemalloc.get_traced_memory()[1] < 1_000_000
    finally:
        tracemalloc.stop()
    with pytest.raises(ValueError):
        BackendLimits(max_seconds=float("inf"))


def test_malformed_cache_header_is_rejected_without_declared_shape_allocation(tmp_path):
    source = tmp_path / "source.wav"
    sf.write(source, np.full((32, 1), .1), 8000, subtype="DOUBLE")
    project = Project.create(source, tmp_path / "project")
    calls = []

    def separate(samples, rate, queries, **kwargs):
        calls.append(queries)
        return samples[None].copy()

    backend = SimpleNamespace(checkpoint_sha256="fixture", limits=BackendLimits(),
                              device="cpu", separate=separate)
    engine = SceneEngine(backend, SimpleNamespace())
    engine._proposal(project, "rain", None)
    path = next((project.root / "proposal-cache").glob("*.npy"))
    with path.open("wb") as stream:
        np.lib.format.write_array_header_1_0(stream, {"descr": "<f8", "fortran_order": False,
                                                     "shape": (100_000_000, 1)})
    path.with_suffix(".sha256").write_text(sha256_file(path)+"\n")
    tracemalloc.start()
    try:
        with pytest.raises((ValueError, OSError)):
            engine._proposal(project, "rain", None)
        # SHA256 admission uses a bounded 1 MiB read buffer; the declared
        # 800 MB array must never be allocated.
        assert tracemalloc.get_traced_memory()[1] < 2*1024*1024
    finally:
        tracemalloc.stop()
    assert len(calls) == 1
    assert Project.load(project.root).revision == 0


def test_changed_anchors_on_locked_stem_fail_before_model_work(tmp_path):
    from stemscene import Anchor, LockedLayerError
    source = tmp_path / 'source.wav'
    sf.write(source, np.full((32,1),.1),8000,subtype='DOUBLE')
    project = Project.create(source,tmp_path/'project')
    identity = project.add_estimates({'rain':project.original.samples})['rain']
    project.set_locked(identity,True)
    project.set_anchors(identity,[Anchor(0,.002,False)])
    revision = project.revision
    engine = SceneEngine(SimpleNamespace(),SimpleNamespace())
    with pytest.raises(LockedLayerError,match='Anchors for locked'):
        engine.extract(project,['rain'])
    reloaded = Project.load(project.root)
    assert reloaded.revision == revision
    np.testing.assert_array_equal(reloaded.layers[0].samples,project.original.samples)
