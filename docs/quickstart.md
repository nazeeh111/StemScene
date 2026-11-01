# Quickstart and owner walkthrough

The commands below use the installed engine. Start with an authorized **mono WAV or FLAC**. Keep the source and project destinations separate, and choose fresh model/project/export destinations: commands refuse to overwrite existing outputs.

This beta guide uses the shared installed engine. A fresh non-editable wheel
installation with public model downloads and a different recording passed
three-query CPU inference, API editing, export and standalone replay. Exact
versions, measurements and remaining platform limits are in
[verification](verification.md). A command listing alone is not an executed check.

## Install the locked checkout

Python 3.11 and 3.12 are admitted; the local executed runtime is Python 3.12.13. From the repository root, with uv installed:

```sh
uv sync --locked --python 3.12
```

`uv.lock` pins transitive distributions and hashes. It was generated with uv 0.11.20. Installation uses public package indexes; extraction later stays offline. Do not run `uv sync` in another project's environment.

On Linux, the default PyPI PyTorch distribution can bring GPU runtime packages even when using CPU. To keep a CPU-only installation in an isolated environment, use this alternative:

```sh
python3.12 -m venv .venv-cpu
. .venv-cpu/bin/activate
python -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
```

This alternative honors the directly pinned project dependencies and uses the official CPU wheel. It is distinct from the universal uv lock route. CI exercises the CPU route on Linux; remote CI status must be checked for the actual revision.

## Download public assets explicitly

Choose a new directory. The downloader verifies exact byte counts and SHA-256 values before promoting the directory. It retains verified files and partial bytes in a clearly named sibling folder if setup fails.

```sh
uv run --locked stemscene models download "$HOME/.local/share/stemscene/models"
```

The approximately 1.27 GB download contains the official AudioSep checkpoint, its already included CLAP text weights, RoBERTa tokenizer/config files and license evidence. It does not need a model API key, Hugging Face sign-in, a separate RoBERTa weight download, SAM Audio access or CUDA. See [attribution](../THIRD_PARTY_NOTICES.md) for the distinction between AudioSep code MIT and the weight host's MIT metadata declaration.

For subsequent examples, set paths to your files. Do not reuse an existing project/output destination:

```sh
STEMSCENE_MODELS="$HOME/.local/share/stemscene/models"
STEMSCENE_REFINER="$(pwd)/src/stemscene/artifacts/joint-refiner-v1.pt"
STEMSCENE_SOURCE="/absolute/path/to/authorized-recording.wav"
STEMSCENE_PROJECT="$HOME/.local/share/stemscene/scene-one"
```

`joint-refiner-v1.pt` is the frozen experimental beta artifact. The explicit
refiner argument is optional in CLI/browser commands; omitting it uses the
bundled V1 artifact. The engine never fetches a refiner implicitly. Read the
[model card](model-card.md) before making quality or identity claims.

The following commands use `uv run --locked`. If you selected the activated Linux CPU environment instead, replace that prefix with the directly installed `stemscene` command.

## Use the browser mixer

```sh
uv run --locked stemscene serve \
  --workspace "$HOME/.local/share/stemscene/browser-workspace" \
  --checkpoint "$STEMSCENE_MODELS/audiosep_base_4M_steps.ckpt" \
  --tokenizer-dir "$STEMSCENE_MODELS/roberta" \
  --refiner "$STEMSCENE_REFINER" \
  --port 0
```

Open the printed `http://127.0.0.1:PORT/` address. The service binds only to loopback. `--port 0` chooses an available port instead of displacing another app. Stop it with Ctrl-C.

1. Import your recording. The engine preserves its bytes and timing.
2. Request the sounds you want, separated by commas or new lines. Estimation can take longer on the first request while local weights load.
3. Listen to each estimate. Select a time span by dragging the waveform or entering start/end seconds, then apply a layer level and fade.
4. Lock any estimate that must survive later requests unchanged. Locks protect stem samples; gain edits remain available. Unlock before changing separation anchors.
5. Add requests, undo saved changes and export the verified aligned bundle.

The player uses FLOAT WAV and the browser/device playback path. Auditioning is not bit-exact export verification. The engine makes no remote model request; choose a workspace appropriate to your own OS backup/sync preferences.

## Use the CLI

Create a project and compute three queries from its unchanged original:

```sh
uv run --locked stemscene init "$STEMSCENE_SOURCE" "$STEMSCENE_PROJECT"
uv run --locked stemscene extract "$STEMSCENE_PROJECT" \
  --checkpoint "$STEMSCENE_MODELS/audiosep_base_4M_steps.ckpt" \
  --tokenizer-dir "$STEMSCENE_MODELS/roberta" \
  --refiner "$STEMSCENE_REFINER" \
  --query "doorbell ringing" \
  --query "a door opening and closing" \
  --query "keys jangling"
uv run --locked stemscene inspect "$STEMSCENE_PROJECT"
```

Descriptions are requests, not assertions that those targets exist in your recording. The domestic-scene example above demonstrates a workflow; its classes are outside the ESC-10 controlled refiner study. General speech, wind and vehicle quality requires separate evaluation.

Copy the desired persistent layer ID from `inspect`, then edit it:

```sh
STEMSCENE_LAYER="paste-layer-id-here"
uv run --locked stemscene edit "$STEMSCENE_PROJECT" "$STEMSCENE_LAYER" \
  --start 4.65 --end 6.29 --gain 0.25 --fade 0.02
uv run --locked stemscene lock "$STEMSCENE_PROJECT" "$STEMSCENE_LAYER"
```

A gain of `0.25` is 25% linear amplitude; `0` removes the selected estimate, not a guaranteed clean physical source. Fades stay inside the selected span. Choose spans within your own recording; the times here are illustrative.

A new `edit` command adds another multiplicative gain operation. To revise a saved event without stacking another attenuation, copy its event ID from the `edits` list in `inspect` and use `--replace`. Removing an event is also undoable:

```sh
STEMSCENE_EVENT="paste-event-id-here"
uv run --locked stemscene edit "$STEMSCENE_PROJECT" "$STEMSCENE_LAYER" \
  --replace "$STEMSCENE_EVENT" --start 4.65 --end 6.29 --gain 0.50 --fade 0.02
uv run --locked stemscene remove-edit "$STEMSCENE_PROJECT" "$STEMSCENE_EVENT"
uv run --locked stemscene undo "$STEMSCENE_PROJECT"
```

Adding another request jointly refines unlocked estimates while retaining the locked stem's samples:

```sh
uv run --locked stemscene extract "$STEMSCENE_PROJECT" \
  --checkpoint "$STEMSCENE_MODELS/audiosep_base_4M_steps.ckpt" \
  --tokenizer-dir "$STEMSCENE_MODELS/roberta" \
  --refiner "$STEMSCENE_REFINER" \
  --query "clock ticking"
```

A new query can be absent or ambiguous. Review the output; never treat extracted energy or similarity as proof of presence.

## Add anchors and undo

Anchors are seconds in the **mixed original**, not clean references. The command replaces that layer's anchor list. Unlock the stem first when its saved separation anchors will change, then re-estimate:

```sh
uv run --locked stemscene unlock "$STEMSCENE_PROJECT" "$STEMSCENE_LAYER"
uv run --locked stemscene anchors "$STEMSCENE_PROJECT" "$STEMSCENE_LAYER" \
  --present 4.65 6.29 --absent 0 0.50
uv run --locked stemscene extract "$STEMSCENE_PROJECT" \
  --checkpoint "$STEMSCENE_MODELS/audiosep_base_4M_steps.ckpt" \
  --tokenizer-dir "$STEMSCENE_MODELS/roberta" \
  --refiner "$STEMSCENE_REFINER" \
  --query "doorbell ringing"
uv run --locked stemscene undo "$STEMSCENE_PROJECT"
```

Choose anchors by inspecting your own mixed recording. The illustrative spans are not valid for every input. A conflicting lock or overlapping present/absent anchors causes an error rather than silent reassignment. Undo restores one preceding persistent revision, including the prior gains, lock state or layer set. Project history is bounded.

## Export and verify elsewhere

```sh
uv run --locked stemscene render "$STEMSCENE_PROJECT" edited-mix.wav
uv run --locked stemscene export "$STEMSCENE_PROJECT" exported-scene
uv run --locked stemscene replay exported-scene replayed-mix.wav
uv run --locked python examples/recombine.py exported-scene --output independently-replayed.wav
```

The last script imports only NumPy and soundfile, not STEMSCENE or model code. It verifies checksums, sample alignment, source bytes, unity reconstruction, WAV envelopes, JSON gain recipes and the edited mix. It can also read the browser's ZIP download directly.

Default output is DOUBLE WAV. `--subtype FLOAT` explicitly quantizes export arrays. Default peak policy rejects a rendered peak above 1; `--clipping preserve` retains floating peaks and `--clipping hard` clips the final mix subject to the unedited-sample safeguard. There is no hidden loudness normalization. See the [audio contract](audio-contract.md) for tolerances and bounds.

## Device and resource limits

CLI extraction defaults to CPU and four CPU threads. On the tested Apple Silicon Mac, add `--device mps` to extraction or `serve` to use MPS. The `col2im` operation has an observed CPU fallback, so this is mixed CPU/GPU execution, not a promise of full GPU acceleration. No Linux/CUDA or other-hardware speed claim follows from that test.

The current product ceiling is 600 seconds and eight distinct requested classes, subject to 256 MiB source, 512 MiB decoded-array, 2 GiB project-array and inference-time bounds. Long-context processing is offline. Cancellation takes effect between bounded steps and before committing a new project state; it does not instantly interrupt an active native kernel. An admitted maximum is not a validated quality or latency guarantee.
