# STEMSCENE

STEMSCENE edits selected events in persistent **estimated sound layers**, keeps the original recording intact, and exports aligned WAVs with an independently reproducible mix. The browser mixer, Python API and CLI share one installed local engine. New estimates use the optional local extractor; saved projects can be edited without it.

A layer is an estimate of a sound class. It can contain leakage, miss a quiet event or share energy with another description. It is not proof that one physical source was isolated.

**Experimental beta. Current source: 0.1.1b0.** The original 0.1.0b1 evaluation is retained unchanged: actual local inference, a trained 579-parameter refiner, browser editing and independent WAV replay have been exercised. On the frozen 20-scene study, trained refinement reduced mean all-request edit error by **7.25%** versus the strongest comparable non-oracle baseline. However, **29 of 60 individual edits worsened**, target gain accuracy declined and **five of six absent requests falsely extracted audio**. This is a bounded edit-integrity result, not a general separation-quality or scientific-novelty claim. See [all comparisons and failures](docs/comparisons.md).

![Current installed editor: a supplied synthetic layer and saved event reopened without model setup](docs/demo/assets/editor-current.png)

[Current capture provenance](docs/demo/assets/current-capture.json).

[Watch the earlier 0.1.0b1 actual-session walkthrough (23 seconds)](docs/demo/assets/stemscene-walkthrough.mp4) or open the [audio comparison and credits](docs/demo/index.html). The walkthrough uses captured session states and actual original/edited audio; inference waiting is omitted.

## Use it

### Try saved editing without model setup

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). From this checkout:

```sh
git clone https://github.com/nazeeh111/StemScene.git
cd StemScene
uv sync --locked --python 3.12
uv run --locked stemscene serve \
  --workspace "$HOME/.local/share/stemscene/workspace" \
  --port 0
```

Open the printed loopback URL. Choose **Open exported project** and select [the synthetic editable example](docs/demo/assets/model-free-editing.zip). It supplies one deterministic layer and a saved 0.10–0.30 s event at 25%. Select that event, change its level, choose **Update event**, then **Undo**. Export a new bundle and reopen it to keep editing. [Fixture provenance](docs/demo/assets/model-free-editing.json).

This route installs NumPy, SciPy and soundfile, and downloads no model weights. The supplied layer is 0.6 times the synthetic recording, demonstrating editing and handoff rather than source separation. Existing exports carry original audio and undo history; review them before sharing. The tagged 0.1.0b1 release retains its original command set; use current source for portable project reopen.

### Estimate new layers locally

The optional extraction route starts with an authorized mono recording. From this checkout:

```sh
uv sync --locked --extra models --python 3.12
uv run --locked --extra models stemscene models download "$HOME/.local/share/stemscene/models"
uv run --locked --extra models stemscene serve \
  --workspace "$HOME/.local/share/stemscene/workspace" \
  --checkpoint "$HOME/.local/share/stemscene/models/audiosep_base_4M_steps.ckpt" \
  --tokenizer-dir "$HOME/.local/share/stemscene/models/roberta" \
  --port 0
```

The bundled V1 refiner is the frozen beta artifact. Open the printed `127.0.0.1` URL. Import WAV/FLAC, request sound descriptions, listen to the estimates, select an event span and change its level. Revise a saved event without stacking another attenuation. Lock an estimate before adding requests that must not change it. Undo restores the preceding saved state.

Setup downloads approximately **1.27 GB** of pinned public model/tokenizer/license files into a **new** directory. Normal extraction does not download models or upload recordings. CPU is the default. For optional tested Apple MPS execution, prefix the command with `PYTORCH_ENABLE_MPS_FALLBACK=1` and add `--device mps`; some operations require CPU fallback. Set that environment variable before starting Python, including when using the API. SAM Audio is not required.

See the [complete quickstart](docs/quickstart.md) for CLI editing, anchors, export, offline replay and the Linux CPU installation route. The [Python API](docs/python-api.md) uses the same engine.

## What an export contains

| File | Purpose |
| --- | --- |
| Preserved source file | Original imported bytes and SHA-256 |
| `original-decoded.wav` | Decoded original at its declared rate and channel count |
| `stems/*.wav` | Aligned estimated components with stable layer IDs |
| `residual.wav` | Original minus the sum of estimates |
| `envelopes/*.wav` and JSON recipes | Applied gains, exact sample spans and inside-span fades |
| `mix.wav`, `manifest.json` | Rendered edit, processing metadata, checksums and measured round-trip errors |

The edit equation is `original + sum((gain_envelope - 1) * estimated_stem)`. Samples outside selected edit spans remain untouched in the decoded render. Export verifies the actual written WAVs before publishing the output folder. Default DOUBLE WAV has a maximum absolute round-trip tolerance of `1e-12`; FLOAT uses `1e-6`. Incompatible numerical cancellation or quantization is rejected. Reconstruction is a contract check, not a separation score.

An independent replay needs NumPy and soundfile, but no engine or model:

```sh
python examples/recombine.py /path/to/export-or.zip --output replay.wav
```

## Method and evidence

AudioSep supplies text-conditioned proposals from the unchanged mixed recording. A compact trained query-set refiner uses proposals, mixture features and optional user anchors. Persistent class identities, locks, explicit residuals and original-based gain edits connect those estimates to a production workflow. Text prompts, chunking, anchors and mixture consistency are established methods, not claimed inventions.

The controlled pilot uses licensed genuine ESC-10 recordings with source/uploader/recipe separation. Its generated mixtures are **controlled mixtures of recordings**, not naturally captured multitrack truth. Clean references and presence labels are training/evaluator-only. A separate RealDESED recording has been used for natural-scene operation; it has no clean stems and is outside the refiner's ESC-10 training domain.

The final candidate, edit protocol, scalar controls and thresholds were frozen before confirmation. Trained mean edit NMSE was 0.0441843 versus 0.0476391 for fixed 0.75 proposal scaling, with 16/20 scene wins but only 31/60 individual-request wins. Direct SDR fell from 3.606 to 3.145 dB versus independent extraction, and target gain error rose from 0.226 to 0.472. Sequential reverse extraction beat trained refinement in 12/20 scenes despite its worse mean. Source reuse leaves only five connected scene groups; no statistical confidence or listening-quality claim follows. The [model card](docs/model-card.md) records training, provenance and intended use.

An independent non-editable wheel installation downloaded fresh public models and a different natural recording, ran three-query CPU inference, and independently replayed its exported edit. It took 328.28 seconds including model load, with 3.20 GiB peak process RSS on the tested Mac. A separate 48-second development mixture exercised long refinement, locks, order, cancellation, silence and boundaries; [its failures and measurements](docs/long-audio.md) remain part of the result.

- [Audio contract](docs/audio-contract.md)
- [Foundations and prior-work scope](docs/foundations.md)
- [Frozen confirmation and failures](docs/comparisons.md), [long-audio stress](docs/long-audio.md), and [installation evidence](docs/verification.md)
- [Refinement method](docs/refinement-method.md) and [v2 experiment](docs/refinement-v2.md)
- [Frozen data protocol](docs/data-protocol.md) and [metric definitions](docs/evaluation-metrics.md)
- [Sequential comparison](docs/sequential-comparison.md) and [FUSS comparison](docs/fuss-comparison.md)
- [Editable architecture](docs/architecture.md), with [Mermaid source](docs/architecture.mmd)

## Limits

- Learned separation is **mono only**. The core can edit/export stereo arrays; that is not evaluated stereo separation.
- This is offline processing with full-clip context, not causal or live streaming.
- The admission ceiling is 600 seconds and eight requested classes, with additional byte and time bounds. That ceiling is not a claim of measured quality over every admitted length.
- Similar descriptions and indistinguishable sources may need grouping or anchors. Automatic prompt-overlap warnings missed tested near-synonyms, so users must review and group descriptions themselves. Similarity is not isolation confidence. Locked stems cannot silently change; incompatible requests or changed anchors produce a conflict.
- There is no hidden loudness normalization or speech regeneration. Peak policy is explicit; hard clipping is rejected if it would alter unedited original samples.
- No human listening-panel or general speech/wind/vehicle quality claim has been established. Review estimates before attenuating them.

The local test machine had 32 GiB memory and ten CPU cores. That is an observed machine, not a verified minimum requirement. Cold loading, warmed inference, MPS fallback, clip length and shared machine load affect performance.

## Development

```sh
uv sync --locked --extra test --extra models --python 3.12
uv run --locked --extra test --extra models python -m pytest -q
```

The full correctness suite needs the optional model runtime, including CPU PyTorch, but downloads no large model checkpoint. CI exercises the installed CLI, alignment, edits, locks, undo, malformed inputs, memory admission and refiner invariants. It does not download the large checkpoint or replace real local-inference verification with synthetic fixtures.

## Attribution

Original STEMSCENE code is [MIT](LICENSE). Retained AudioSep and Torchlibrosa code keeps its own MIT notices. The optional FUSS adapter is [Apache-2.0](LICENSES/Apache-2.0.txt); its deposited checkpoint has separate CC BY 4.0 terms. Model-host and code licenses are recorded separately. See [third-party notices](THIRD_PARTY_NOTICES.md) before redistributing code, models or examples.

The retained Apache text is byte-identical to the [official FUSS repository license](https://raw.githubusercontent.com/google-research/sound-separation/master/LICENSE), SHA-256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`.

## Reopen, recover and replay

The editor opens a portable export as a new project without running a model. From the same checkout, the CLI provides the matching workflow:

```sh
uv run --locked stemscene export /path/to/project /path/to/new-export
uv run --locked stemscene replay /path/to/new-export /path/to/new-replay.wav
uv run --locked stemscene reopen /path/to/new-export /path/to/new-editable-project
uv run --locked stemscene undo /path/to/new-editable-project
```

New exports contain an exact editable project alongside the existing aligned
playback WAVs. Original recording bytes, float64 estimate arrays, layer and event
identities, anchors, uncertainty metadata and undo history travel together. FLOAT
playback exports keep the original FLOAT readback tolerances; their editable
project still preserves exact estimates. Reopened rendering remains the
pre-policy floating mix, and the recorded export peak policy is shown/restored in
the editor. No resampling or loudness normalization is introduced. Older
playback-only exports remain replayable; reopening them refuses rather than
inventing history or treating quantized playback as original estimates.

In the browser, **Open exported project** accepts the downloaded ZIP and creates a
new project. **Reload saved state** recovers an out-of-date view after another tab
or process changes the project. Every mutation and export supplies its accepted
revision; a stale edit, undo or export refuses instead of silently applying to a
newer revision. Extraction computes proposals outside committed state and checks
its source revision and cancellation before one final save. Cancellation requests
that arrive after the final atomic commit can finish as completed; undo remains
available.

Projects use a persistent `.write.lock` inode with crash-released kernel advisory
ownership. Do not delete it: its existence does not mean a writer is active.
macOS and Linux local filesystems with the required flock/no-replace-rename
support are the supported writer targets; network filesystems and other platforms
are not verified. A legacy PID-only marker converts only when that PID is
definitely absent. Live/reused PIDs, empty or unknown legacy markers and ambiguous
ownership refuse conservatively; preserve the marker, close older versions and
verify the owner before manual recovery. No PID-age deletion is used. Initialized
lock publication and migration are atomic, so interruption leaves recoverable
ownership. A killed process can leave unreferenced arrays or owned temporary
files; they are ignored, never guessed away or treated as a committed revision.

Export/reopen publishes a fully verified new directory without replacing even a
concurrently created empty destination. Failed writes preserve the last valid
project; a failed ZIP does not expose a partial download. Browser bundles are
limited to 512 MiB and exact referenced project assets to 2 GiB. Exports now also
include the original audio and undo history: review that content before sharing.
These are synthetic-tested persistence/audio contracts, not new separation,
physical-source identity or listening-quality claims. The beta limitations and
negative model results above remain unchanged.
