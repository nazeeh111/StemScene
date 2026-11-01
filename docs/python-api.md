# Python API

The installed CLI and loopback browser service call `SceneEngine`. Normal inference uses explicit local model files and has no clean-reference or presence-label argument.

```python
from pathlib import Path

from stemscene import Project, export_project
from stemscene.backend import AudioSepBackend, BackendLimits
from stemscene.engine import SceneEngine
from stemscene.joint import JointRefiner

model_directory = Path("/absolute/path/to/models")
refiner_artifact = Path("/absolute/path/to/selected-refiner.pt")

project = Project.create(
    "/absolute/path/to/authorized-recording.wav",
    "/absolute/path/to/new-project",
)
backend = AudioSepBackend(
    model_directory / "audiosep_base_4M_steps.ckpt",
    model_directory / "roberta",
    limits=BackendLimits(threads=4),
    device="cpu",
)
engine = SceneEngine(backend, JointRefiner.load(refiner_artifact))
ids = engine.extract(project, ["doorbell ringing", "a door opening and closing", "keys jangling"])

layer_id = ids["doorbell ringing"]
event_id = project.edit(layer_id, start=4.65, end=6.29, gain=0.25, fade_seconds=0.02)
project.set_locked(layer_id, True)
engine.extract(project, ["clock ticking"])
export_project(project, "/absolute/path/to/new-export")
```

The explicit artifact must match the loader's supported architecture; do not interchange differently tagged experimental architectures merely by renaming a file. V1 was selected and frozen before confirmation. The installed CLI and browser default to the bundled `joint-refiner-v1.pt`; an explicit path remains available for reproducible experiments.

`Project.create` copies the original bytes into a fresh folder. `Project.load(path)` checks saved metadata and arrays before returning a persistent project. Mutations persist automatically and reject stale concurrent writers. `project.layers` contains `Layer` objects with stable IDs, query text, read-only sample arrays, lock state, anchors and uncertainty metadata. `project.edits` contains the sample-based edit operations.

## Revise a saved event

`edit` returns its persistent event ID. Supplying it again replaces that event rather than multiplying a second operation on top:

```python
project.edit(layer_id, start=4.70, end=6.20, gain=0.50, fade_seconds=0.02, edit_id=event_id)
project.remove_edit(event_id)
project.undo()  # Restore the removed event.
```

Calling `edit` without an ID adds a new operation. This distinction lets callers deliberately combine edits while keeping ordinary event revisions predictable.

## Anchors and reversible edits

```python
from stemscene import Anchor

project.set_locked(layer_id, False)
project.set_anchors(layer_id, [
    Anchor(start=4.65, end=6.29, present=True),
    Anchor(start=0.0, end=0.5, present=False),
])
engine.extract(project, ["doorbell ringing"])
project.undo()  # Restore the preceding saved state.
```

Anchor spans must come from inspection of the mixed recording. They do not prove clean isolation. Changing a locked stem's anchors requires unlocking before re-estimation; incompatible locks are reported. A sound class can contain multiple physical sources, and identical or indistinguishable sources need not become separate identities.

The engine currently rejects stereo inference. Core project arrays use `(samples, channels)` float64 and can support stereo edits/exports of estimates supplied through the core API. That separate capability is not evidence of stereo source separation.

## Cancellation and progress

```python
from threading import Event
from stemscene.backend import InferenceCancelled

cancel = Event()
try:
    engine.extract(
        project,
        ["rain", "dog barking"],
        cancelled=cancel.is_set,
        progress=lambda completed, total: print(completed, total),
    )
except InferenceCancelled:
    print("No partial layer update was committed.")
```

Set `cancel` from your own controlling thread or UI. Cancellation is observed between bounded inference steps and before persistent mutation, not inside a currently executing native kernel. Valid proposal caches may remain for reuse, while saved source bytes and the preceding project state remain intact.

## Replay without inference

```python
from stemscene import replay_export

samples = replay_export("/absolute/path/to/export")
```

This path reads the exported WAVs and envelopes without constructing a separator or refiner. For a separate implementation that does not import STEMSCENE, run `examples/recombine.py` with an export folder or ZIP. Reconstruction tolerance and unchanged outside-span samples are audio-contract properties; neither establishes correct sound attribution.

## Bounds and errors

`AudioContractError` is the common invalid-media/project error. `LockedLayerError` reports a protected estimate conflict, and `AnchorNeededError` identifies a new request that may overlap a locked estimate. Existing destinations are not overwritten. Peak policy and export precision are explicit; extreme cancellation or FLOAT quantization that prevents reproducible replay is rejected.

The backend verifies the pinned checkpoint SHA-256 before restricted tensor loading. All ordinary methods remain offline. Only the explicit model-download utility makes public asset requests. See [the audio contract](audio-contract.md), [method](refinement-method.md) and [licenses](../THIRD_PARTY_NOTICES.md) before making quality, streaming or redistribution claims.
