# Frozen confirmation bundle, V1

This bundle retains the executed study's protocol, frozen decision, per-recipe metrics and source snapshots. It does not rerun, retune or generate data. [Comparison](../../docs/comparisons.md) explains the bounded positive primary mean and material regressions; [summary](summary.md) includes every scene and all 60 request outcomes.

| Evidence | Role |
|---|---|
| [confirmation-protocol-v2.json](confirmation-protocol-v2.json) | Exact pre-confirmation protocol, including failure-capable interpretation |
| [confirmation-freeze-v1.json](confirmation-freeze-v1.json) | Exact final V1 model/method/metric decision |
| [report.json](report.json) | Portable complete aggregate; only its freeze path was changed |
| [per-recipe.jsonl](per-recipe.jsonl) | Byte-exact 20-scene outputs, every metric and request |
| [source-capture/sha256.json](source-capture/sha256.json) | Hashes of exact executed preparation/evaluator/refiner/FUSS snapshots |
| [source-capture/provenance.json](source-capture/provenance.json) | Original/published hashes and explicitly allowed portability edits |
| [training-v1/report.json](training-v1/report.json), [trace](training-v1/training-log.json) | Compact exact training provenance and 12-epoch trace; selected epoch 8 |
| [model-selection-review.md](model-selection-review.md) | Prospective selection context; V2 was not tested as a replacement on confirmation |
| [FUSS](fuss-augmentation.json), [sequential](sequential-augmentation.json) | Actual preparation record hashes and timings |
| [logs](logs/) | Retained startup failure, successful preparation and evaluation logs; path-only redactions documented |

The protocol's historical wording precedes final candidate selection. The later unchanged freeze resolves that decision to V1. Historical relative paths in `selection_evidence` refer to the original prospective task layout; the included model-selection review supplies context. Training-era `training-v1/source/joint.py` and confirmation-runtime `source-capture/src/stemscene/joint.py` have different preserved hashes; do not silently substitute one for the other.

The raw executed report SHA-256 is `51be09afc6917cccd5319b1f7ae05ccd68abe75780b49fb7149fa72733744b12`. The portable report SHA-256 is `c46ee66b67528fb6d2e53c0fe53184a3b225f6821303cee8b1b5f9ae8410ca6b`. Its sole parsed change is `decision_freeze.path` to `confirmation-freeze-v1.json`; every other value, including all numeric results and the embedded freeze, is unchanged. The exact freeze SHA remains `467770cb9889b2b2229c8eda7fb21ef6a3fe096e4e65dc4dd99fe9da95702541` and protocol SHA remains `ef9ba78a9ec0c002902fca4734b796f56b36a5bc1c37ea3e9783594ae6aa509b`.

## Run recipe

These commands document the executed sequence using logical paths, not a newly performed experiment. Run from the repository root with installed main and isolated TensorFlow environments. `WORK/mixtures` must already contain the original frozen materialization from [data preparation](../../docs/data-protocol.md). `MODELS` must already contain the verified local AudioSep checkpoint, RoBERTa tokenizer and FUSS archive/extracted model described in the [quickstart](../../docs/quickstart.md) and [FUSS protocol](../../docs/fuss-comparison.md). Large artifacts are intentionally not bundled. All output directories below must be fresh and separate from originals. Do not use confirmation to select another model, scalar, threshold or edit subset.

The source-capture utilities are the exact executed bytes. The FUSS command imports the already installed setuptools first, matching the successful retained startup wrapper. A later adapter startup repair is not substituted into this capture. The scripts use the installed repository backend and frozen runtime; the preflight below checks relevant hashes before any confirmation processing. MPS may require a native scoped GPU permission on managed macOS. No credentials, billed API or recording upload is involved.

```sh
export WORK=/path/to/fresh-study-work
export MODELS=/path/to/verified-local-models
export REFINER="$PWD/src/stemscene/artifacts/joint-refiner-v1.pt"
PYTHON="$PWD/.venv/bin/python"
FUSS_PYTHON="$PWD/.venv-fuss/bin/python"
CAPTURE=evaluation/v1/source-capture
FREEZE=evaluation/v1/confirmation-freeze-v1.json
MANIFEST=data/esc10-split.json
RECIPE_SHA=bd19a599b3165a0648e1ed9fef17e085e18fd4878f090e25e24945454c3d0728

"$PYTHON" - <<'CHECK'
from pathlib import Path
import hashlib, json, os
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
root = Path.cwd()
bundle = root / 'evaluation/v1'
freeze = json.loads((bundle / 'confirmation-freeze-v1.json').read_text())
assert sha(bundle / 'confirmation-freeze-v1.json') == '467770cb9889b2b2229c8eda7fb21ef6a3fe096e4e65dc4dd99fe9da95702541'
assert sha(bundle / 'confirmation-protocol-v2.json') == freeze['protocol_sha256']
assert sha(root / 'data/esc10-split.json') == freeze['frozen_recipes_sha256']
assert sha(os.environ['REFINER']) == freeze['checkpoint_sha256']
assert sha(root / 'src/stemscene/joint.py') == freeze['refiner_runtime_sha256']
assert sha(root / 'scripts/evaluate_refiner.py') == freeze['evaluator_sha256']
for name, expected in json.loads((bundle / 'source-capture/sha256.json').read_text()).items():
    assert sha(bundle / 'source-capture' / name) == expected
assert sha(Path(os.environ['MODELS']) / 'audiosep_base_4M_steps.ckpt') == 'f8cda01bfd0ebd141eef45d41db7a3ada23a56568465840d3cff04b8010ce82c'
CHECK

PYTORCH_ENABLE_MPS_FALLBACK=1 "$PYTHON" \
  "$CAPTURE/scripts/prepare_proposals.py" \
  "$WORK/mixtures" "$MANIFEST" "$WORK/proposals" \
  --checkpoint "$MODELS/audiosep_base_4M_steps.ckpt" \
  --tokenizer "$MODELS/roberta" --split confirmation \
  --decision-freeze "$FREEZE" --device mps --threads 4

"$FUSS_PYTHON" -c 'import setuptools,runpy; runpy.run_path("evaluation/v1/source-capture/scripts/prepare_fuss_proposals.py",run_name="__main__")' \
  "$WORK/proposals" "$WORK/proposals-fuss" \
  --split confirmation --decision-freeze "$FREEZE" \
  --refiner-checkpoint "$REFINER" --frozen-recipes "$MANIFEST" \
  --expected-recipes-sha256 "$RECIPE_SHA" \
  --model-dir "$MODELS/baseline_dry_model" \
  --model-archive "$MODELS/FUSS_baseline_dry_model.tar.gz" --threads 2

PYTORCH_ENABLE_MPS_FALLBACK=1 "$PYTHON" \
  "$CAPTURE/scripts/prepare_sequential_proposals.py" \
  "$WORK/proposals-fuss" "$WORK/mixtures" "$WORK/proposals-comparison" \
  --split confirmation --decision-freeze "$FREEZE" \
  --refiner-checkpoint "$REFINER" --frozen-recipes "$MANIFEST" \
  --expected-recipes-sha256 "$RECIPE_SHA" \
  --checkpoint "$MODELS/audiosep_base_4M_steps.ckpt" \
  --tokenizer "$MODELS/roberta" --device mps --threads 4

"$PYTHON" "$CAPTURE/scripts/evaluate_refiner.py" \
  "$WORK/proposals-comparison" "$REFINER" "$WORK/evaluation-output" \
  --split confirmation --decision-freeze "$FREEZE" \
  --frozen-recipes "$MANIFEST" --expected-recipes-sha256 "$RECIPE_SHA" \
  --scalar-gain 0.60 --absence-threshold 0.01 --threads 4
```

Keep preparation sidecars and emitted augmentation reports. Check all 20 IDs against the frozen manifest, every NPZ against its sidecar and parent hashes, and exact dtype/value preservation of original arrays across both augmentation stages. Refuse incomplete coverage or changed provenance. A rerun requires actual input/model artifacts and computation; metadata alone cannot reproduce predictions, and platform changes may prevent bit-identical MPS outputs. This recipe is not evidence of an additional run.

The complete retained report was emitted before `/usr/bin/time -l` failed its macOS `sysctl kern.clockrate` query. The evaluator wall timing was captured, evaluator peak memory was unavailable, and the native denial was not bypassed. See [evaluation log](logs/confirmation-evaluation.log). Query-zero legacy fields remain in the exact report for transparency and must not replace the declared all-request primary.
