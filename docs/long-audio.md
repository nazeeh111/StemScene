# Actual 48-second long-audio stress validation

**The installed engine completed actual long inference, persistent edits and aligned export. This is operational evidence with mixed accuracy, not proof of correct persistent source identity.** V1 improves average edit error in this one case while making the already better sneeze estimate substantially quieter and less accurate. The clock estimate remains heavily contaminated.

**Near-synonym limitation:** the tested common aliases did not trigger the unchanged prompt-overlap threshold. Exact normalized duplicates group correctly; semantic synonyms are not reliably grouped or warned about.

The fixed development input is 48 seconds, 32 kHz mono, 1,536,000 samples, built before inference from three licensed selection-source ESC-10 recordings: rain `4-161127-A-10.wav`, clock `4-179984-A-38.wav`, and sneezing `4-156844-A-21.wav`. Original WAV bytes, attribution, hashes and verified ESC-10 license are retained in the reproduction workspace. Compact [source provenance](../evaluation/long-v1/provenance.json) is included here; source audio and large arrays are not bundled. Source excerpts repeat with declared gains/fades, overlap, transients near backend 3-second hops and refiner overlap boundaries at 18/20/36/38 seconds, and complete silence at 21.5–26.5 and 46–48 seconds. The sneeze excerpt's strongest peak occurs about 110 ms before its nominal center. This is a deliberately constructed class-level mixture, not natural multitrack or pure physical-source ground truth.

Preparation, inference and evaluation ran in separate processes. `infer` reads only the mixed WAV, requests, public pinned AudioSep weights and frozen V1 checkpoint. References remain in `evaluator-only/` and enter only the later scoring process. The fresh project's three actual MPS proposal calls completed before caching; operational repetitions reuse these verified proposals. No model or evaluator was changed for the stress run, and no held-out audio entered it. Later packaging adds this standalone runner and evidence.

## Accuracy and edits

All edits attenuate to gain 0.25 with 20 ms inside-span fades. Each requested layer is edited separately, then the three errors are averaged. NMSE divides waveform squared error by mean mixture power. Whole-record averaging dilutes short edits; edited-sample-only values are also retained in the JSON.

| Metric | Independent AudioSep | Fixed 0.60 scaling | Frozen V1 |
| --- | ---: | ---: | ---: |
| Edit NMSE, 17–21 s | 0.009575 | 0.010416 | **0.007631** |
| Edit NMSE, 35–39 s | 0.009604 | 0.009859 | **0.007411** |
| Edit NMSE, interior 6.5–8.5 s | 0.004201 | 0.004014 | **0.002980** |
| Whole-record attenuation NMSE | 0.076381 | 0.087309 | **0.067885** |
| Whole-record waveform NMSE | 0.135807 | 0.155237 | **0.120703** |
| Activity power-envelope NMSE | **0.243103** | 0.359269 | 0.274285 |
| Activity-change NMSE | 0.197712 | 0.183872 | **0.171521** |

The scalar is fixed from earlier selection, not fitted to this input. Simultaneously attenuating all three layers at 17–21 seconds gives NMSE 0.024044 independent and 0.005393 V1, but simultaneous errors can cancel and do not establish accuracy of individual layers.

| Requested class | Direct SDR, independent → V1 | SI-SDR, independent → V1 | Gain projection, independent → V1 |
| --- | ---: | ---: | ---: |
| Clock | -8.98 → -5.93 dB | -31.37 → -23.87 dB | 0.0717 → 0.1133 |
| Rain | 5.65 → 5.35 dB | 4.57 → 5.43 dB | 0.6404 → 0.5453 |
| Sneezing | **11.34 → 6.20 dB** | 11.01 → 9.39 dB | **0.9275 → 0.5470** |

The sneeze edit at 17–21 seconds worsens from 0.001561 to 0.005288; rain also worsens from 0.008748 to 0.010319. Better average error is driven by reducing erroneous energy in the clock estimate. None of these automated scores establishes human-perceived quality.

## Boundaries and class association

Error was measured in 100 ms windows centered on actual overlap starts/ends. Interior controls use the same transient excerpt and gain at the same waveform phase relative to the measurement center; surrounding background and model context still differ. These are observational controls, not a causal experiment isolating chunk joins.

| Boundary | Independent NMSE | V1 NMSE | V1 matched interior NMSE |
| --- | ---: | ---: | ---: |
| 18 s | 0.824634 | 0.678232 | 0.263706 |
| 20 s | 0.622854 | 0.510895 | 0.263706 |
| 36 s | 0.803827 | 0.707649 | 0.263706 |
| 38 s | 0.809873 | 0.598360 | 0.675762 |

V1 improves error and derivative error at all four refiner boundaries versus independent proposals. Three boundary windows remain worse than their matched interior controls. Backend-boundary mean error is 0.198011 independent and 0.176560 V1. Different source activity prevents interpreting pooled boundary/interior ratios as introduced join artifacts.

Same-query correlations across both 2-second refiner overlaps exceed 0.99998. However, clock/sneeze cross-correlation is about 0.994 at the second overlap and triggers an ambiguity warning. Evaluator-only two-second source regression identifies **12 incorrect dominant class attributions among 42 active reference-class/window instances**, unchanged by refinement. Stable query indices and nearly identical overlap predictions therefore do not prove correct identity or separation. The [metric record](../evaluation/long-v1/metrics.json) retains coefficients, conditioning, every boundary/control and failure.

## Operational checks and full silence

- Two request batches followed by full canonical-set refinement match simultaneous requests exactly. This uses independent original-mixture proposals, not sequential residual extraction.
- Reversing queries changes samples by at most 5.96e-8.
- Adding the third class preserves a previously locked rain estimate exactly. Its RMS difference from a fully unlocked three-class estimate is 0.003103, so locks preserve the saved result rather than promising the latest unconstrained estimate.
- A negative rain anchor at 22–25 seconds is grounded solely in the observed all-zero mixed signal. Its interior output is exactly zero. This does not test positive-anchor source isolation.
- Cancellation occurs between long refinement windows; saved manifest bytes and revision remain unchanged.
- Samples outside the edit and undo-to-original error are exactly zero. Original source bytes are preserved.
- Actual DOUBLE WAV exports have unity reconstruction error 1.11e-16; independent WAV/envelope replay equals the saved edited mix exactly.

A separate five-second all-zero pass uses actual AudioSep MPS and frozen V1 after a duplicate-query warmup in the same follow-up process. All three proposal/refined energies, maxima, residual energy/maxima and rendered-original delta are exactly zero; no edit records or original-byte changes occur. The measured warm three-query pass takes 6.14 seconds. This verifies numerical full-silence behavior, not semantic absence detection in nonzero mixtures.

Exact normalized duplicates (`rain`, ` RAIN `, repeated `rain`) share one layer; five input requests produce three actual layers and three backend calls. Actual text embeddings do **not** flag the tested near-synonyms at the unchanged 0.98 threshold: `clock tick`/`clock ticking` cosine 0.9568, `rain`/`rainfall` 0.8853, `sneezing`/`sneeze` 0.8081. No tested pair is exposed as uncertain by that prompt threshold. Near-synonym grouping/detection is therefore not established and should remain an explicit limitation.

## Resources and retained artifacts

The installed editable package was used with four CPU threads, actual MPS AudioSep, documented `col2im` CPU fallback, and CPU V1 refinement. The long run loads models in 15.49 seconds, extracts three layers in 81.20 seconds (1.69 times input duration), and completes primary preparation/export/replay work in 126.08 seconds. The full operational-smoke process takes 261.18 seconds. Cumulative RSS reaches 843 MB for primary work and 962 MB including smokes. MPS samples every 50 ms observe 939 MB allocated and 2.45 GB driver memory; these are observed lower bounds, not exact allocator peaks. Another MPS task ran concurrently, so timings are shared-hardware observations.

Retained: mixed input/request, exact source originals/licenses, evaluator references, actual proposal/refined arrays, persistent projects, WAV/envelope export, operation logs, runtime source snapshot and frozen checkpoint, plus machine-readable reports. The portable runner is [scripts/long_stress.py](../scripts/long_stress.py). One repeated three-source development case supplies no confidence interval, general long-scene accuracy, physical identity, stereo or human listening evidence. It does not alter the registered efficacy metric or model selection.


## Reproduce the fixed case

Use Python 3.11 or 3.12 and the project's pinned environment. The research runner additionally requires **`threadpoolctl==3.7.0`**; this is a script requirement, not a new runtime dependency of the installed editor.

```bash
python -m pip install -e . threadpoolctl==3.7.0
python scripts/long_stress.py prepare   --workspace /tmp/stemscene-long   --source-cache /path/to/esc10-cache

OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=4 PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/long_stress.py infer   --workspace /tmp/stemscene-long   --models /path/to/models   --refiner /path/to/frozen-v1.pt

python scripts/long_stress.py evaluate --workspace /tmp/stemscene-long

OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=4 PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/long_stress.py extras   --workspace /tmp/stemscene-long   --models /path/to/models   --refiner /path/to/frozen-v1.pt
```

`--source-cache` is the pinned ESC-10 cache root containing `audio/` and `LICENSE`. The existing repository data manifest selects only selection-source recordings. The script never downloads audio or models. `--models` expects `audiosep_base_4M_steps.ckpt` and `roberta/`; alternatively provide explicit `--checkpoint` and `--tokenizer` paths. `--refiner` must be the frozen V1 artifact whose SHA-256 appears in [hashes.json](../evaluation/long-v1/hashes.json). The repository is detected from the script location. The inference stages require actual MPS support and retain four CPU threads with the documented operator fallback.

`prepare` requires a fresh workspace and preserves originals. `infer` rejects an existing primary project instead of overwriting valid inference. Preparation writes source references into `evaluator-only/`; inference reads `input/` and model assets only. `evaluate` opens references after inference. `extras` runs the independently timed all-zero and duplicate tests on a warmed backend after its own model-load phase.

The portable script parameterizes paths and command-line plumbing; stimulus and scoring arithmetic, model settings and operations remain fixed. Explicit manifest and V1 hash guards prevent path arguments from silently selecting a different experiment. The exact finalized historical runner hash is retained alongside the portable runner hash. The historical file accumulated separate scoring and silence stages while work progressed; it is not represented as one source-file freeze preceding every stage.

**Preparation readback passes numerical equivalence.** A fresh `prepare` reproduces every decoded mixed-WAV sample and every float64 reference value exactly, with identical source entries, event schedule, shape, rate and gain factor. Its whole-file WAV hash initially failed comparison: libsndfile's `PEAK` chunk contains a current-time timestamp. Only bytes 60–61 differ, within the four-byte timestamp field beginning at offset 60; `fmt`, `fact`, `data`, and every other file byte match. Original whole-file SHA-256 is `e17acb13a102364809a4fafb548a36e70abee34311a788649beafd308ab74ebe`; fresh whole-file SHA-256 is `36894002d1eb5546bf29c092e482e6a41542c885f14046c91a54eb08148df622`. The producer versions and full bounded diagnosis are in [preparation-check.json](../evaluation/long-v1/preparation-check.json).

The initial mismatch and fresh output are retained. No timestamp was normalized, no historical output was changed, and inference was not rerun. Numerical reproduction is established; whole-file byte identity of newly generated controlled WAV headers is not claimed or required. Original recording bytes remain preserved exactly.

Compact artifacts: [metrics](../evaluation/long-v1/metrics.json), [provenance](../evaluation/long-v1/provenance.json), [operations](../evaluation/long-v1/operations.json), [runtime](../evaluation/long-v1/runtime.json), and [hashes](../evaluation/long-v1/hashes.json). Large arrays, source audio and persistent workspaces remain external to this evidence bundle.
