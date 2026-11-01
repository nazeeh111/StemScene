# Independent prospective model-selection review

**Recommendation: freeze V1 as the single primary efficacy candidate if proceeding under the registered all-request edit metric. Neither candidate has earned the full product or public novelty claim.** V1 has the lower primary mean and lower present-request edit error. V2 is a credible, more interpretable peer-repair experiment with better average gain accuracy, but its average advantage over simple scalar calibration comes entirely from absent requests. That does not justify overriding the declared edit-integrity priority. Retain V2 and its failures as selection evidence, not a second candidate to choose after held-out results.

V1 artifact: `training-v1/selected.pt`, SHA-256 `ddbc872006da68440ad3f72435c3b89ecacf422f2f7265d68581f1dee879a9a9`. V2 artifact: `training-v2/selected.pt`, SHA-256 `ef631022391812eae3b1f24af3ce780e0284a2b43a7d67feaaf92c645835ebf8`. The current V2 trainer/module hashes match its provenance. The fixed twelve-epoch pilot selected epoch eleven. No candidates, evaluator, protocol, manifest or repository files were changed by this review.

## Candidate evidence

These are selection means over ten dependent scenes, not held-out results. Primary scoring edits every request separately, including absent requests, using the same frozen span/gain/fade. Present-only scores below were calculated directly from the raw per-query edit errors, averaging present requests within each scene and then scenes. This gives a matched denominator for FUSS.

| Method | All-request edit NMSE | Present-only edit NMSE | Gain error | Direct SDR, dB | SI-SDR, dB | Absent energy / mixture |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Independent | 0.061753 | 0.053916 | 0.2131 | 2.715 | 0.766 | 0.55774 |
| Fixed 0.60 scalar | 0.042473 | 0.041272 | 0.5279 | 1.145 | 0.766 | 0.20079 |
| Analytic selection scalar 0.597682 | 0.042472 | 0.041310 | 0.5297 | 1.135 | 0.766 | 0.19924 |
| V1 | **0.039221** | **0.040850** | 0.4756 | 1.442 | 0.361 | 0.14533 |
| V2 | 0.040709 | 0.044754 | **0.4245** | **1.909** | **0.633** | **0.12082** |
| FUSS present-only oracle | not comparable | 0.040624 | 0.5293 | 3.291 | -14.838 | not evaluated |

V2's primary error is 3.79% higher than V1's. This is not dominance by V1: V2 wins six of ten scenes and 22 of 30 request edits versus V1, improves scene-level gain error in nine scenes and direct SDR in eight. V1's mean lead disappears when `selection-008` is omitted. Nevertheless, V2's better whole-clip loudness summaries do not translate into better selected-span edit integrity. Both candidates still worsen average gain error and direct SDR relative to independent extraction.

V2 beats independent on all ten scene means, but improves only 13 of 27 present request edits and worsens 14. V1 improves only 11 of 27 present request edits. Thus neither scene-average result means users can expect each extracted layer to improve. Severe V2 failures include dog in `selection-005` (edit NMSE 0.009570 independent, 0.071127 V1, **0.091457 V2**) and rain in `selection-008` (0.085477 scalar 0.60, 0.147284 V1, **0.216803 V2**). These must survive into reporting.

V2 improves over the analytic global scalar by 0.001764 on the primary metric and wins eight scenes. However, across the 30 equal-weight request slots, its three absent requests contribute total improvement **+0.059757**, while the 27 present requests contribute **-0.006847**. Matched present-only scene-average error is 0.003444 worse than the scalar. V1's matched present-only advantage over that scalar is only 0.000460, and FUSS is slightly better than V1 on that matched scope. These margins support a bounded candidate test, not broad layer-quality superiority.

All three V2 absent outputs remain above the fixed 0.01 energy threshold: 0.19237, 0.09289 and 0.07720. Absence energy improves over V1, but absence detection still fails 3/3. Residual Wiener and forward sequential extraction have lower average absence energy than either learned candidate.

## What the peer ablations establish

| V2 variant | All-request error | Present-only error |
| --- | ---: | ---: |
| Full model | **0.040709** | 0.044754 |
| Gain-only controls | 0.048644 | 0.045898 |
| Peer subtraction control disabled | 0.045560 | **0.044368** |

The full model beats both ablations on every scene's all-request mean. This is evidence that the trained correction terms matter, not merely fixed shrinkage. But **91.1%** of the summed 30-slot improvement over disabling peer subtraction comes from the three absent slots. The matched present-only scene mean slightly worsens with peer subtraction, and only 13/27 present request edits improve. The ablation retains peer-derived features and collision gating; it disables the explicit peer-subtraction output channel, not all query-set conditioning.

The source keeps clean references and presence out of inference. V2 suppresses correction without a distinct peer, uses coherent proposal overlap to gate bounded corrections, and preserves ambiguous-description identity behavior. Those structural choices are sensible, but coherent overlap can represent legitimate shared sound. The scores do not establish physical-source identity, long-term tracking, anchor usefulness, or clean removal of duplicated sources. Maximum measured permutation discrepancy is 8.94e-8, a numerical symmetry result.

## Failure-capable interpretation before held-out access

Use the already registered primary metric and fixed **0.60 scalar**, not a scalar fitted to new references. Freeze V1's hash and the executable evaluation version before any held-out inference. A positive mean improvement must be compared with the strongest comparable non-oracle baseline in the registered set, including both real sequential orders. If it is nonpositive, the joint-refinement efficacy claim fails for this study. V2 cannot replace V1 afterward because its held-out result looks better.

A positive mean with a negative median, few scene wins, source-reuse sensitivity, or present-layer deterioration supports only an explicitly qualified aggregate result. If the gain comes chiefly from absent targets, say so; it does not establish improved fidelity of present layers. Report matched present-only comparisons against FUSS, all per-scene/request failures, direct SDR and SI-SDR separately, gain errors, absence threshold failures and activity limits. Do not turn a favorable secondary subset into a new primary after seeing results.

Neither five-second selection study tests persistence across real chunk boundaries. ESC class-labeled reference recordings can contain background sound and are not guaranteed pure physical sources. Ten selection scenes reuse sources, and the planned twenty scenes remain a small controlled study. No natural multitrack, human listening, certified speaker/session, foundation-pretraining isolation, stereo or broad novelty claim follows. Passing this bounded efficacy test would leave those product requirements separate.

## Reviewed sources

Raw `selection-v2-comparison/report.json` and `per-recipe.json`; `training-v2/report.json`; current `src/stemscene/joint_v2.py`, `scripts/train_refiner_v2.py` and task-local `compare-v2-selection.py`; the original goal and `evaluation-review.md`; and `protocols/confirmation-protocol-v2.json` (SHA-256 `ef9ba78a9ec0c002902fca4734b796f56b36a5bc1c37ea3e9783594ae6aa509b`). Only authorized selection metadata was read to identify present/absent request slots. No held-out recordings, references or results were listed or opened. No new audio experiment was run.
