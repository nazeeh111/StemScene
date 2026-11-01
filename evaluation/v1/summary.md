# Frozen confirmation results

Status: complete frozen confirmation execution. No training, tuning, implementation changes, scalar changes, or metric/subset changes were made for this run.

## Frozen input verification

Before processing confirmation records, read back SHA-256 of the selected model, manifest, decision freeze, evaluator and refiner runtime. All matched the frozen values:

- Selected V1 model: `ddbc872006da68440ad3f72435c3b89ecacf422f2f7265d68581f1dee879a9a9`
- Frozen recipes: `bd19a599b3165a0648e1ed9fef17e085e18fd4878f090e25e24945454c3d0728`
- Decision freeze: `467770cb9889b2b2229c8eda7fb21ef6a3fe096e4e65dc4dd99fe9da95702541`
- Evaluator: `cdff3c59ddedbd042644de941ad5f614333bb51578bc99e1bfba8fb13147b328`
- Refiner runtime: `2f4ac80775563c924164686429ccd29757b1ea2a05d428f1a3f32d9895a7b40d`

Original AudioSep root `WORK/proposals`: all 20 NPZ plus metadata sidecars complete; exact frozen recipe IDs, 60 query identities, source IDs and manifest provenance match. Every NPZ SHA matches its sidecar; numeric arrays finite. Original parents are read only.

## Failures and retained outputs

First FUSS launch, using `.venv-fuss/bin/python scripts/prepare_fuss_proposals.py`, failed before inference with `ModuleNotFoundError: No module named 'distutils'` during TensorFlow import under Python 3.12. Full log: [retained startup failure](logs/confirmation-fuss-generation.log). The empty output `WORK/failed-fuss-empty-output` is preserved.

An existing-runtime probe established that importing the already installed `setuptools` first supplies its vendored `distutils` and permits TensorFlow 2.18.1 import. The retry uses the unchanged script through `python -c 'import setuptools,runpy; runpy.run_path(...,run_name="__main__")'`. No package or source changes. Retry log: [successful FUSS log](logs/confirmation-fuss-generation-retry1.log); fresh output `WORK/proposals-fuss`.

## Execution scope

FUSS, actual forward/reverse sequential calls, frozen evaluation and independent parent-array/hash checks are complete. The result and tradeoffs are below. A positive mean alone is not a public-release or goal-completion determination.

## FUSS execution verified

20/20 records completed in the retry. Independent readback checked every NPZ and metadata parent hash and verified 120 original arrays, including dtype and exact values, unchanged. The sole added array is aligned finite `fuss_all_sources` with four model outputs.

Cold model construction: 15.592 s; loop elapsed including model load and record writes: 73.511 s; total inference: 24.816 s (mean 1.241, range 0.715–3.265 s). Process cumulative peak RSS: 695730176 bytes. These are shared-machine observations, not isolated benchmark guarantees.

Actual sequential MPS forward/reverse job launched using the unchanged frozen model/manifest and four threads, with `PYTORCH_ENABLE_MPS_FALLBACK=1`; expected 120 model calls. Log [sequential log](logs/confirmation-sequential-generation.log).

## Sequential execution verified

20/20 scenes completed, with 120 actual AudioSep calls (3 queries × 2 orders × 20 scenes) on MPS, with observed `aten::col2im` CPU fallback. Independent readback verified 140 FUSS-parent arrays unchanged in dtype and exact values. The only additions are aligned finite forward/reverse arrays; each computed array SHA matches its metadata. All parent NPZ and metadata SHA values match. Evaluator and refiner runtime SHA still match the freeze before evaluation.

Cold model load: 13.126 s. Summed forward extraction: 104.176 s; reverse: 38.696 s. Text/model warming, execution order, writes and concurrent machine load prevent treating these as controlled forward-versus-reverse timing comparisons. Cumulative process peak RSS: 1,082,294,272 bytes. Forward/reverse waveform RMS differences span 0.00330564 to 0.0741726, establishing observed order dependence.

Full unchanged frozen evaluator launched with scalar 0.60, absence threshold 0.01, four CPU threads into fresh `WORK/evaluation-output`. `/usr/bin/time -l` was requested to capture timing and process memory in [evaluation log](logs/confirmation-evaluation.log); timing succeeded but macOS denied its extended memory query, as documented below.

## Frozen result

Status: complete execution of all 20 frozen confirmation scenes and 60 requests (54 present, 6 absent). Model, manifest, freeze, evaluator and runtime SHA read back unchanged after evaluation. Report SHA-256: `51be09afc6917cccd5319b1f7ae05ccd68abe75780b49fb7149fa72733744b12`; per-recipe JSONL SHA-256: `89c2606f59937de810d14f284344cc56eda8f441cb9ad83b8bdc9c6077dc55e3`.

The smallest all-request mean among the seven comparable non-oracle baselines is `scalar_fixed_0_75`. Trained all-request edit NMSE is 0.044184319, versus 0.047639145; baseline minus trained difference 0.0034548261 (7.2520741%). Positive difference favors trained. This is a bounded descriptive result on these controlled five-second scenes, with the failures and dependencies below; it is not a general efficacy, listening-quality or product-release conclusion.

The declared primary metric averages the error of the same fixed attenuation applied separately to each requested layer, then averages scenes. Present and absent rows preserve the evaluator's within-scene averaging. Simultaneous editing is a separately reported operation that attenuates all requested layers together. Lower NMSE, gain error and duplicate-energy values are better; higher direct SDR is better.


| Method | All-request edit NMSE | Present edit NMSE | Absent edit NMSE | Simultaneous edit NMSE | Direct SDR dB | Gain absolute error | Excess source-duplicate energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| scalar_fixed_0_75 | 0.047639145 | 0.047000033 | 0.048283227 | 0.056373859 | 3.0896551 | 0.41971791 | 0.14091149 |
| sequential_reverse | 0.047698942 | 0.047752797 | 0.042772415 | 0.030308881 | 4.0066069 | 0.44310815 | 0.036033174 |
| scalar_calibrated | 0.047738018 | 0.049269992 | 0.030901265 | 0.051297543 | 2.5604755 | 0.53577433 | 0.090183351 |
| residual_wiener | 0.048428096 | 0.048978166 | 0.043506881 | 0.041520039 | 3.6056461 | 0.48938107 | 0.04381679 |
| unweighted_consistency | 0.049648237 | 0.047829058 | 0.066386583 | 0.038166149 | 2.9725023 | 0.35864168 | 0.14925603 |
| sequential_forward | 0.058659845 | 0.060771022 | 0.053093443 | 0.036181736 | 3.2489564 | 0.46668406 | 0.02517141 |
| independent | 0.05937775 | 0.055841245 | 0.085836848 | 0.12573176 | 3.6059286 | 0.22629857 | 0.25050931 |
| trained | 0.044184319 | 0.044253665 | 0.040444907 | 0.02847828 | 3.1446945 | 0.47243442 | 0.093289404 |

FUSS is an unlabelled four-output universal separator matched one-to-one against clean present references by the evaluator only. It cannot choose a semantic absent target; its unmatched zero rows are not successful absence detection. Its present-only edit NMSE is 0.051226577, direct SDR 2.3974516 dB, gain absolute error 0.60205322. It has no comparable all-request or simultaneous-edit score.


## Wins and regressions

Counts are descriptive, using tolerance 1e-12. A request is not an independent sample. The evaluator's older `failure_flags` use only request index 0; all-request comparisons below recompute from each frozen row.

| Baseline | Scene primary wins / regressions / ties | Request wins / regressions / ties | Present wins / regressions / ties | Absent wins / regressions / ties |
|---|---:|---:|---:|---:|
| scalar_fixed_0_75 | 16 / 4 / 0 | 31 / 29 / 0 | 26 / 28 / 0 | 5 / 1 / 0 |
| sequential_reverse | 8 / 12 / 0 | 26 / 34 / 0 | 23 / 31 / 0 | 3 / 3 / 0 |
| scalar_calibrated | 14 / 6 / 0 | 39 / 21 / 0 | 37 / 17 / 0 | 2 / 4 / 0 |
| residual_wiener | 10 / 10 / 0 | 25 / 35 / 0 | 23 / 31 / 0 | 2 / 4 / 0 |
| unweighted_consistency | 11 / 9 / 0 | 33 / 27 / 0 | 27 / 27 / 0 | 6 / 0 / 0 |
| sequential_forward | 9 / 11 / 0 | 29 / 31 / 0 | 27 / 27 / 0 | 2 / 4 / 0 |
| independent | 16 / 4 / 0 | 28 / 32 / 0 | 23 / 31 / 0 | 5 / 1 / 0 |

### Every scene versus scalar_fixed_0_75

| Scene | Baseline primary | Trained primary | Baseline minus trained | Outcome |
|---|---:|---:|---:|---|
| confirmation-000 | 0.079495287 | 0.069627852 | 0.0098674349 | win |
| confirmation-001 | 0.047803624 | 0.04151625 | 0.0062873743 | win |
| confirmation-002 | 0.022024011 | 0.016550156 | 0.0054738552 | win |
| confirmation-003 | 0.063700957 | 0.055903911 | 0.0077970457 | win |
| confirmation-004 | 0.029387571 | 0.025540048 | 0.0038475232 | win |
| confirmation-005 | 0.078072741 | 0.0655212 | 0.012551542 | win |
| confirmation-006 | 0.017562345 | 0.014100825 | 0.0034615203 | win |
| confirmation-007 | 0.024359764 | 0.022277703 | 0.0020820606 | win |
| confirmation-008 | 0.04004074 | 0.03692771 | 0.0031130294 | win |
| confirmation-009 | 0.0064520994 | 0.0041313827 | 0.0023207167 | win |
| confirmation-010 | 0.10330127 | 0.10548959 | -0.0021883154 | regression |
| confirmation-011 | 0.053817453 | 0.053936822 | -0.00011936925 | regression |
| confirmation-012 | 0.05871073 | 0.058014952 | 0.00069577776 | win |
| confirmation-013 | 0.0094852842 | 0.0065356011 | 0.0029496831 | win |
| confirmation-014 | 0.0076119367 | 0.0078517369 | -0.00023980021 | regression |
| confirmation-015 | 0.056868675 | 0.049975705 | 0.0068929703 | win |
| confirmation-016 | 0.045050003 | 0.043418393 | 0.0016316106 | win |
| confirmation-017 | 0.062567997 | 0.064655678 | -0.0020876814 | regression |
| confirmation-018 | 0.10747333 | 0.1056892 | 0.0017841298 | win |
| confirmation-019 | 0.038997074 | 0.036021659 | 0.0029754145 | win |

### Every request versus scalar_fixed_0_75

| Scene / index | Query | Present | Baseline error | Trained error | Baseline minus trained | Outcome |
|---|---|---|---:|---:|---:|---|
| confirmation-000 / 0 | rooster | True | 0.013613515 | 0.026773616 | -0.013160101 | regression |
| confirmation-000 / 1 | crying baby | True | 0.20565728 | 0.16186048 | 0.043796801 | win |
| confirmation-000 / 2 | sea waves | True | 0.019215062 | 0.020249457 | -0.0010343959 | regression |
| confirmation-001 / 0 | crackling fire | True | 0.040530587 | 0.048469676 | -0.0079390894 | regression |
| confirmation-001 / 1 | crying baby | True | 0.0065560065 | 0.0093161519 | -0.0027601454 | regression |
| confirmation-001 / 2 | rain | True | 0.096324279 | 0.066762922 | 0.029561358 | win |
| confirmation-002 / 0 | crackling fire | True | 0.0075600753 | 0.0044985001 | 0.0030615751 | win |
| confirmation-002 / 1 | rooster | True | 0.035581516 | 0.026237034 | 0.0093444824 | win |
| confirmation-002 / 2 | rain | False | 0.022930441 | 0.018914933 | 0.0040155081 | win |
| confirmation-003 / 0 | rain | True | 0.0084934477 | 0.012216601 | -0.0037231535 | regression |
| confirmation-003 / 1 | dog | True | 0.14347888 | 0.092839288 | 0.050639594 | win |
| confirmation-003 / 2 | crying baby | True | 0.039130541 | 0.062655845 | -0.023525304 | regression |
| confirmation-004 / 0 | clock tick | True | 0.039009453 | 0.024179546 | 0.014829907 | win |
| confirmation-004 / 1 | dog | True | 0.028827522 | 0.02386271 | 0.0049648124 | win |
| confirmation-004 / 2 | rain | True | 0.02032574 | 0.028577889 | -0.0082521497 | regression |
| confirmation-005 / 0 | rain | True | 0.16701019 | 0.11127719 | 0.055733005 | win |
| confirmation-005 / 1 | crackling fire | False | 0.01992859 | 0.0080779685 | 0.011850622 | win |
| confirmation-005 / 2 | sea waves | True | 0.047279441 | 0.077208442 | -0.029929002 | regression |
| confirmation-006 / 0 | sneezing | True | 0.021111914 | 0.013663613 | 0.0074483002 | win |
| confirmation-006 / 1 | dog | True | 0.0043514762 | 0.0084841776 | -0.0041327014 | regression |
| confirmation-006 / 2 | chainsaw | True | 0.027223645 | 0.020154683 | 0.0070689622 | win |
| confirmation-007 / 0 | sea waves | True | 0.025420346 | 0.024945059 | 0.00047528625 | win |
| confirmation-007 / 1 | sneezing | True | 0.0068741295 | 0.0071197831 | -0.00024565361 | regression |
| confirmation-007 / 2 | chainsaw | True | 0.040784817 | 0.034768267 | 0.0060165492 | win |
| confirmation-008 / 0 | rain | True | 0.041498565 | 0.043390943 | -0.0018923787 | regression |
| confirmation-008 / 1 | rooster | True | 0.020317742 | 0.029372361 | -0.0090546192 | regression |
| confirmation-008 / 2 | helicopter | False | 0.058305912 | 0.038019826 | 0.020286086 | win |
| confirmation-009 / 0 | rooster | True | 0.017937415 | 0.011285186 | 0.0066522289 | win |
| confirmation-009 / 1 | crackling fire | True | 0.00062353009 | 0.00056826833 | 5.526176e-05 | win |
| confirmation-009 / 2 | clock tick | True | 0.00079535343 | 0.00054069391 | 0.00025465952 | win |
| confirmation-010 / 0 | chainsaw | True | 0.013284896 | 0.015362157 | -0.0020772613 | regression |
| confirmation-010 / 1 | helicopter | True | 0.18982269 | 0.18952758 | 0.00029510644 | win |
| confirmation-010 / 2 | crackling fire | True | 0.10679624 | 0.11157903 | -0.0047827913 | regression |
| confirmation-011 / 0 | chainsaw | True | 0.057358586 | 0.06173314 | -0.0043745539 | regression |
| confirmation-011 / 1 | crying baby | False | 0.016547572 | 0.0093600239 | 0.0071875479 | win |
| confirmation-011 / 2 | rain | True | 0.0875462 | 0.090717302 | -0.0031711017 | regression |
| confirmation-012 / 0 | crackling fire | True | 0.048064916 | 0.051660074 | -0.0035951585 | regression |
| confirmation-012 / 1 | sea waves | True | 0.057855461 | 0.048326999 | 0.0095284614 | win |
| confirmation-012 / 2 | chainsaw | True | 0.070211814 | 0.074057784 | -0.0038459697 | regression |
| confirmation-013 / 0 | crackling fire | True | 0.019660878 | 0.012227921 | 0.0074329568 | win |
| confirmation-013 / 1 | dog | True | 0.0013552108 | 0.0028277825 | -0.0014725718 | regression |
| confirmation-013 / 2 | sneezing | True | 0.0074397638 | 0.0045510996 | 0.0028886642 | win |
| confirmation-014 / 0 | crackling fire | False | 0.00028965536 | 0.0023875366 | -0.0020978812 | regression |
| confirmation-014 / 1 | clock tick | True | 0.0031118198 | 0.0041587326 | -0.0010469128 | regression |
| confirmation-014 / 2 | crying baby | True | 0.019434335 | 0.017008942 | 0.0024253933 | win |
| confirmation-015 / 0 | rain | True | 0.038655359 | 0.035535038 | 0.0031203204 | win |
| confirmation-015 / 1 | sneezing | True | 0.010784882 | 0.01410354 | -0.0033186584 | regression |
| confirmation-015 / 2 | chainsaw | True | 0.12116578 | 0.10028854 | 0.020877249 | win |
| confirmation-016 / 0 | chainsaw | True | 0.061548976 | 0.057018204 | 0.0045307712 | win |
| confirmation-016 / 1 | crying baby | True | 0.0212214 | 0.022153636 | -0.00093223587 | regression |
| confirmation-016 / 2 | rain | True | 0.052379635 | 0.051083338 | 0.0012962965 | win |
| confirmation-017 / 0 | rooster | False | 0.17169719 | 0.16590915 | 0.0057880397 | win |
| confirmation-017 / 1 | crying baby | True | 0.011091385 | 0.022018374 | -0.010926989 | regression |
| confirmation-017 / 2 | crackling fire | True | 0.0049154152 | 0.0060395097 | -0.0011240945 | regression |
| confirmation-018 / 0 | sneezing | True | 0.069329548 | 0.082667567 | -0.013338019 | regression |
| confirmation-018 / 1 | helicopter | True | 0.22970694 | 0.19476786 | 0.034939076 | win |
| confirmation-018 / 2 | chainsaw | True | 0.023383497 | 0.039632164 | -0.016248667 | regression |
| confirmation-019 / 0 | sea waves | True | 0.034701409 | 0.046174992 | -0.011473584 | regression |
| confirmation-019 / 1 | crackling fire | True | 0.082285724 | 0.060204212 | 0.022081512 | win |
| confirmation-019 / 2 | sneezing | True | 4.0883377e-06 | 0.0016857732 | -0.0016816849 | regression |

## Absent-query behavior

False extraction means extracted energy divided by mixture energy exceeds the frozen 0.01 threshold. These six semantic absences have exactly zero reference waves. A reduced energy ratio is not successful absence handling if it remains above threshold.

| Method | False extractions / 6 | Mean absent energy / mixture | Mean absent edit NMSE |
|---|---:|---:|---:|
| scalar_fixed_0_75 | 5 / 6 | 0.13787575 | 0.048283227 |
| sequential_reverse | 4 / 6 | 0.14534742 | 0.042772415 |
| scalar_calibrated | 5 / 6 | 0.08824048 | 0.030901265 |
| residual_wiener | 4 / 6 | 0.13156035 | 0.043506881 |
| unweighted_consistency | 6 / 6 | 0.18741255 | 0.066386583 |
| sequential_forward | 2 / 6 | 0.14614315 | 0.053093443 |
| independent | 5 / 6 | 0.24511244 | 0.085836848 |
| trained | 5 / 6 | 0.11700219 | 0.040444907 |

| Scene | Absent query | Independent energy ratio | scalar_fixed_0_75 energy ratio | Trained energy ratio | Trained false extraction |
|---|---|---:|---:|---:|---|
| confirmation-002 | rain | 0.07520861 | 0.042304843 | 0.035262441 | True |
| confirmation-005 | crackling fire | 0.11283457 | 0.063469444 | 0.027543973 | True |
| confirmation-008 | helicopter | 0.38858368 | 0.21857832 | 0.14549443 | True |
| confirmation-011 | crying baby | 0.053705143 | 0.030209143 | 0.016900408 | True |
| confirmation-014 | crackling fire | 0.0020168649 | 0.0011344865 | 0.0052763982 | False |
| confirmation-017 | rooster | 0.83832579 | 0.47155826 | 0.47153552 | True |

## Reconstruction, permutation and order

The maximum trained permutation restoration error across actual six-permutation checks per scene is 8.9406967e-08. Each scene has three queries; identity plus five reordered executions were evaluated, not a synthetic output shuffle. Maximum trained resynthesis error is 0; maximum outside-edit change is 0.


## Source reuse and descriptive sensitivity

The 60 source slots use 45 distinct files from 35 original recordings and 33 uploaders. 13 files recur, at most 4 times. Recipes and requests are dependent. Weak class labels do not establish physical-source purity; grouping by uploader is not certified speaker/session grouping; base-separator pretraining overlap is unknown. This is controlled recombination, not natural multitrack truth.

Leaving out every scene involving one filename group at a time gives baseline-minus-trained primary mean differences from 0.002593197 to 0.0039669004; 45/45 omissions remain positive. This sensitivity is supplementary and is not a confidence interval.

Leaving out every scene involving one src_file group at a time gives baseline-minus-trained primary mean differences from 0.0025626186 to 0.0040777733; 35/35 omissions remain positive. This sensitivity is supplementary and is not a confidence interval.

Leaving out every scene involving one uploader group at a time gives baseline-minus-trained primary mean differences from 0.0025626186 to 0.0040777733; 33/33 omissions remain positive. This sensitivity is supplementary and is not a confidence interval.

Source-file sharing produces 5 connected scene groups of sizes 10, 7, 1, 1, 1. Their primary baseline-minus-trained group means are 0.0046658007, 0.0019780503, 0.0038475232, 0.0031130294, 0.0016316106. No source-cluster statistical interval was implemented.


## Timing and memory

Original independent AudioSep recorded total 125.562 s, mean 6.278 s, range 1.575–24.369 s per five-second/three-request scene; cold backend load 20.233 s. Recorded maximum cumulative process RSS across originals: 789020672 bytes.

Trained refinement first executions total 57.656 s, mean 2.883 s, range 1.933–5.752 s. This excludes the additional permutation executions and metric calculations. Full evaluator wall/user/system timing appears below, as reported by macOS `/usr/bin/time -l`; it includes loading, all baseline metrics and permutation checks. The evaluator emitted all 20 rows and the complete validated report. The outer timing wrapper then returned exit 1 because macOS denied `sysctl kern.clockrate`; evaluator peak RSS is unavailable. No permission boundary was bypassed and the frozen run was not rerun for this measurement.

```text
      389.80 real        58.90 user        10.06 sys
time: sysctl kern.clockrate: Operation not permitted
```

Observations share hardware with other authorized product work. Process cumulative RSS is not per-record incremental memory and does not report total GPU allocation. No boundary metric is claimed without observed chunk-boundary metadata. No listening panel, natural-reference accuracy, long/stereo/locks/undo product conclusion follows from this source-reference run.


## Execution source snapshots

Before any later packaging-only FUSS startup repair, exact source bytes were copied and read back under `source-capture/`, with `source/sha256.json`. The retained sources include primary/FUSS/sequential preparation, evaluator, FUSS adapter and refiner runtime. The FUSS adapter used here has SHA-256 `18445b3c171a02d4a294458b472af17f3b30c358e9ca9bd90ebb215dd0f7c06c`; FUSS preparation SHA-256 `884e61e720457792759a37cef75ac9a925e371368f298008c62f47169011320b`.

All 20 source-attribution bases passed the frozen condition bound, with condition numbers from 1.3465316 to 879.83823; full coefficients are retained in per-recipe JSONL. They measure attribution within the weakly labelled reference basis and do not establish physical-source purity.

The largest request regressions against fixed scalar 0.75 (increase in edit NMSE) are:

- `confirmation-005`, sea waves, present=True: 0.047279441 → 0.077208442 (increase 0.029929002).
- `confirmation-003`, crying baby, present=True: 0.039130541 → 0.062655845 (increase 0.023525304).
- `confirmation-018`, chainsaw, present=True: 0.023383497 → 0.039632164 (increase 0.016248667).
- `confirmation-018`, sneezing, present=True: 0.069329548 → 0.082667567 (increase 0.013338019).
- `confirmation-000`, rooster, present=True: 0.013613515 → 0.026773616 (increase 0.013160101).

## Portable evidence

[Published report](report.json), [all per-recipe metrics](per-recipe.jsonl), [protocol](confirmation-protocol-v2.json), [freeze](confirmation-freeze-v1.json), [source/provenance](source-capture/provenance.json), [training trace](training-v1/training-log.json), [prospective model-selection review](model-selection-review.md), and [run recipe](README.md#run-recipe). The report SHA quoted above identifies the raw executed report; the portable report changes only its freeze path. Both hashes are recorded in provenance. `WORK` and `REPO` are logical destinations, not private machine paths.

## Independent comparison qualifications

The favorable source-omission ranges above compare only against fixed scalar 0.75. Against actual sequential reverse, trained V1 wins 8/20 scenes and has paired median baseline-minus-trained difference −0.002331803477704614. Omitting every scene involving original recording ID 215658 (scenes 001, 010, 012) leaves mean difference −0.002926769672685104; ID 216370 (007, 010, 011) leaves −0.0006194004813556794; ID 234879 (000, 008, 009) leaves −0.0009456894515142255. Each omission retains 17 scenes. These overlapping-group summaries are descriptive sensitivity, not confidence intervals or estimates of independent sample size. They show that fixed-control robustness must not be generalized to every baseline.

Gain projection absolute error worsens on 53/54 present requests versus independent extraction. The fixed-scalar-0.75 paired scene median improvement is +0.002962548759598503. These arithmetic readbacks add distributional qualifications without changing the frozen report, model, metric or subset.
