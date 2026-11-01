# FUSS comparison protocol

FUSS dry TDCN++ is an independently trained universal separator, used here as a general audio reference baseline. It produces four unlabelled outputs from each mixture. It does not read the requested text, identify an absent query or select a physical source for a user. Its comparison must be labelled **present-query evaluator-only oracle assignment**, not runtime text extraction or a state-of-the-art claim.

## Record preparation

`scripts/prepare_fuss_proposals.py` computes fresh local inference on the same mixture arrays used by the AudioSep/refiner records. It reads only the mixture and sample rate before calling `FussSeparator.separate`. Clean references and presence labels never enter that API. After inference it copies all original NPZ arrays without changing their values and appends raw `fuss_all_sources` with shape `(4, samples)`. It performs no source assignment.

Each augmented NPZ has a new SHA256. Its sidecar preserves original provenance, query/source identities, original input hash, recipe hash and AudioSep checkpoint hash, and adds:

- `parent_record_sha256` and `parent_metadata_sha256` identifying the untouched parent record;
- `fuss_checkpoint_sha256` and hashes of the extracted graph/checkpoint members;
- hashes of the exact float64 mixture array supplied and the stored raw float32 FUSS output array;
- `fuss` metadata containing native/record rates, explicit oracle-assignment policy, timing, CPU thread count, cumulative process peak RSS and separate code/weight licenses.

Preparation verifies every parent NPZ against its recorded SHA256, frozen recipe identity, archive expansion bound and finite numeric array contract. It verifies the downloaded FUSS archive and checks the installed model members against that archive. It refuses existing output roots and roots inside the original records. Interrupted output remains evidence and requires a fresh destination, not overwrite. Selection is the default split. Confirmation requires an explicit final checkpoint and a decision record matching both model and recipe hashes before confirmation records are listed or opened.

## Fair scoring and limits

The evaluator alone assigns distinct FUSS outputs one-to-one to **present** requested reference components. This assignment uses clean source references to score an unlabeled separation system. It is unavailable during product use. FUSS absence metrics are omitted, and edit metrics are omitted for records whose edited request is absent. Other FUSS quality metrics use present rows only. Report those denominators and the oracle advantage beside query-conditioned baselines. No zero-valued fabricated output should be presented as FUSS detecting absence.

FUSS receives a compensated polyphase resample at its native **16 kHz mono** rate. The adapter converts its estimates back to the record's 32 kHz rate and exact sample count for alignment. Upsampling does not recover frequencies above the native 8 kHz Nyquist limit. Results against the original 32 kHz references therefore include this bandwidth limitation. This utility does not claim a separate common-16-kHz scoring experiment.

Inference is offline with full-clip context and two CPU threads by default. Load time includes cold TensorFlow graph initialization; per-record inference includes native conversion. Peak RSS is the process's cumulative maximum, not isolated per-record allocation or total system memory. Other jobs can affect observed latency. Alignment/resynthesis checks do not establish isolation quality. These controlled mixtures use genuine licensed source recordings and are not naturally captured multitrack ground truth.

## Model provenance

Code: Google Research's [FUSS inference source](https://github.com/google-research/sound-separation/blob/master/models/dcase2020_fuss_baseline/inference.py), Copyright 2020 Google LLC, Apache-2.0. The existing adapter retains attribution. Weights: [Zenodo deposition 4012661](https://doi.org/10.5281/zenodo.4012661), record-level CC BY 4.0; cite the FUSS paper and the data's FSD50K source as requested in the deposition. Original source licenses/attribution in the frozen mixture manifest remain applicable and are preserved, not replaced by the model license.

Verified dry archive: 102324048 bytes, MD5 `aca2b32125659a420bd5c307e2505e43`, SHA256 `49f49e9a9ce3bf78771418c342b031c770e0f0604cbf580eb474593af0589529`. It contains the dry graph and checkpoint members; no download occurs during preparation.

## Local development command

Use the isolated TensorFlow environment, keeping the main installed engine environment independent:

```sh
uv venv .venv-fuss --python 3.12
uv pip install --python .venv-fuss/bin/python '.[fuss]'
STEMSCENE_STUDY=/absolute/path/to/study
.venv-fuss/bin/python \
  scripts/prepare_fuss_proposals.py \
  "$STEMSCENE_STUDY/proposals" \
  "$STEMSCENE_STUDY/proposals-fuss" \
  --frozen-recipes data/esc10-split.json \
  --expected-recipes-sha256 bd19a599b3165a0648e1ed9fef17e085e18fd4878f090e25e24945454c3d0728 \
  --model-dir "$STEMSCENE_STUDY/fuss/baseline_dry_model" \
  --model-archive "$STEMSCENE_STUDY/fuss/FUSS_baseline_dry_model.tar.gz" \
  --threads 2
```

Run from the repository root. This command computes only selection records. It writes `fuss-augmentation.json` with parent/output hashes and observed timings. Do not treat this command listing as evidence that execution passed; use that emitted report and the evaluator's measured result.

The retained confirmation run initially failed because TensorFlow 2.18 imported
`distutils`, which Python 3.12 removed. Importing its existing setuptools
dependency first resolved startup without changing separator mathematics. The
adapter now performs that import itself. The original failed log, unchanged-run
wrapper and exact pre-repair source hashes remain in the confirmation evidence.

## Executed selection preparation, 2026-10-29

The command above completed all **10 frozen selection mixtures** in its retained proposal root. Each record contains four raw 160000-sample outputs at the original 32 kHz record rate. Native inference remains 16 kHz mono. This selection preparation opened no confirmation records.

Observed cold model load was **42.34 seconds**, total preparation after model verification **74.50 seconds**, and cumulative process peak RSS **547749888 bytes**. Per-record inference, including rate conversion and first-run graph overhead, ranged from **0.689 to 7.925 seconds**. These are shared-machine observations, not a hardware guarantee.

A separate read-back check verified complete frozen selection coverage, every parent and new NPZ SHA256, finite `(4, 160000)` output arrays, and exact dtype/value preservation of all six original arrays in all ten records. The development readback report is retained as `fuss-selection-verification.json`; generated augmentation roots retain `fuss-augmentation.json` with individual record hashes. The public [confirmation summary](../evaluation/v1/summary.md#fuss-execution-verified) records the subsequent full frozen run and its independent readback. This preparation result establishes real aligned inference and provenance, not separation accuracy. Accuracy and edit results must come from the present-only oracle evaluator described above.
