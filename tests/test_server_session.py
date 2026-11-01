"""Browser-session transaction checks without a model or live socket."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest
import soundfile as sf

from stemscene import Project
from stemscene.audio import AudioContractError
from stemscene.server import BrowserSession


def test_extraction_admission_serializes_with_saved_project_mutations(tmp_path, monkeypatch):
    """A second browser request cannot edit a stale worker's loaded revision."""
    import stemscene.server as server_module

    source = tmp_path / "source.wav"
    sf.write(source, np.zeros((32, 1)), 8000, subtype="DOUBLE")
    session = BrowserSession(tmp_path / "workspace")
    identifier = "a" * 32
    project = Project.create(source, session.workspace / "projects" / identifier)
    monkeypatch.setattr(BrowserSession, "configured", property(lambda self: True))
    # Leave the admitted job running; this checks admission, not inference.
    monkeypatch.setattr(server_module, "Thread", lambda **kwargs: type("Dormant", (), {"start": lambda self: None})())
    loaded, release, attempted, mutated = Event(), Event(), Event(), Event()

    def load_project(_identifier):
        loaded.set()
        assert release.wait(5)
        return project

    monkeypatch.setattr(session, "project", load_project)

    def mutation_request():
        attempted.set()
        with session.project_guard:
            session.idle(identifier)
            mutated.set()

    with ThreadPoolExecutor(max_workers=2) as workers:
        admission = workers.submit(session.start, identifier, ["rain"])
        assert loaded.wait(5)
        mutation = workers.submit(mutation_request)
        assert attempted.wait(5)
        try:
            # Must remain blocked until admission publishes the running job.
            assert not mutated.wait(.1)
        finally:
            release.set()
        key = admission.result(timeout=5)
        with pytest.raises(AudioContractError, match="Wait for extraction"):
            mutation.result(timeout=5)
    assert session.jobs[key]["status"] == "running"
    assert Project.load(project.root).revision == 0


@pytest.mark.parametrize("queries", ["rain", {"rain": True}, 3, ["rain", 3]])
def test_malformed_query_collection_never_starts_worker(tmp_path, monkeypatch, queries):
    import stemscene.server as server_module

    source = tmp_path / "source.wav"
    sf.write(source, np.zeros((32, 1)), 8000, subtype="DOUBLE")
    session = BrowserSession(tmp_path / "workspace")
    identifier = "a" * 32
    Project.create(source, session.workspace / "projects" / identifier)
    monkeypatch.setattr(BrowserSession, "configured", property(lambda self: True))

    def unexpected_worker(**kwargs):
        pytest.fail("Malformed input created an inference worker")

    monkeypatch.setattr(server_module, "Thread", unexpected_worker)
    with pytest.raises(AudioContractError, match="one to eight"):
        session.start(identifier, queries)
    assert session.jobs == {}
