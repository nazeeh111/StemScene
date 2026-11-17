"""One accepted project revision owns admission, edits and read/export snapshots."""
from contextlib import contextmanager
from threading import Event, RLock
from uuid import uuid4

from .audio import AudioContractError
from .project import Project
from .storage import project_lock


class ProjectSession:
    def __init__(self, root):
        self._guard = RLock()
        self._project = Project.load(root)
        self._operation = None

    @property
    def revision(self):
        return self._project.revision

    @property
    def root(self):
        return self._project.root

    def _check(self, expected_revision=None, *, idle=True):
        if expected_revision is not None and (type(expected_revision) is not int or expected_revision != self.revision):
            raise AudioContractError("This view is stale; reload the project before changing or exporting it")
        if idle and self._operation is not None:
            raise AudioContractError("Wait for extraction to finish or cancel it before changing this project")
        self._project.assert_current()

    def reload(self):
        with self._guard:
            if self._operation is not None:
                raise AudioContractError("Cancel extraction and wait before reloading")
            self._project = Project.load(self.root)
            return self.revision

    @contextmanager
    def read(self, expected_revision=None):
        with self._guard, project_lock(self.root, shared=True):
            self._check(expected_revision, idle=False)
            yield self._project

    def change(self, operation, *args, expected_revision=None, **kwargs):
        # Closed vocabulary; HTTP never chooses arbitrary Python attributes.
        if operation not in {"edit", "remove_edit", "set_locked", "set_anchors", "undo", "add_estimates"}:
            raise AudioContractError("Unknown project operation")
        with self._guard:
            self._check(expected_revision)
            return getattr(self._project, operation)(*args, **kwargs)

    def render(self, expected_revision=None):
        with self.read(expected_revision) as project:
            return project.render()

    def export(self, destination, *, expected_revision=None, **options):
        from .export import export_project
        with self._guard:
            self._check(expected_revision)
            return export_project(self._project, destination, **options)

    def begin_extraction(self, expected_revision=None):
        with self._guard:
            self._check(expected_revision)
            token = uuid4().hex
            self._operation = {"token": token, "cancel": Event(), "project": self._project}
            return token

    def cancel(self, token):
        with self._guard:
            if self._operation is not None and self._operation["token"] == token:
                self._operation["cancel"].set()

    def compute(self, token, engine, queries, *, progress=None):
        from .backend import InferenceCancelled
        with self._guard:
            operation = self._operation
            if operation is None or operation["token"] != token:
                raise InferenceCancelled("Extraction admission expired")
        try:
            proposal = engine.propose(operation["project"], queries,
                                      cancelled=operation["cancel"].is_set, progress=progress)
            with self._guard:
                if self._operation is not operation or operation["cancel"].is_set():
                    raise InferenceCancelled("Cancelled before saving; project was not changed")
                self._check(idle=False)
                return proposal.commit(self._project)
        finally:
            with self._guard:
                if self._operation is operation:
                    self._operation = None

    def abandon(self, token):
        """Release admission when model setup fails before computation begins."""
        with self._guard:
            if self._operation is not None and self._operation["token"] == token:
                self._operation = None
