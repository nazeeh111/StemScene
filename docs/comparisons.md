# Frozen V1 comparison

On 20 controlled five-second scenes, trained refinement reduced the declared all-request edit error by **7.25%** against the strongest comparable baseline, fixed scalar 0.75: **0.0441843 versus 0.0476391**. It won 16/20 scene means, but regressed on **29/60 individual requests**, including 28/54 present requests. This supports a bounded aggregate result on this study. It does not establish that a typical requested event improves.

The [protocol](../evaluation/v1/confirmation-protocol-v2.json), [final decision freeze](../evaluation/v1/confirmation-freeze-v1.json), selected model, scalar controls, metric and threshold were fixed before confirmation. V1 was selected at epoch 8 of 12; V2 remained a selection-only pilot. There was no confirmation tuning or candidate switch. See the [prospective selection review](../evaluation/v1/model-selection-review.md), [training trace](../evaluation/v1/training-v1/training-log.json), [portable report](../evaluation/v1/report.json), and [all 60 request outcomes](../evaluation/v1/summary.md#every-request-versus-scalar_fixed_0_75).

**Read the primary field carefully:** `all_request_edit_nmse` edits every requested layer separately with the declared 1–4 second attenuation to gain 0.25 and 20 ms fades, averages requests within a scene, then scenes. NMSE is squared waveform error divided by mixture power. The older `edit_nmse`, top-level paired-improvement fields and `failure_flags` score only original request index zero. They are secondary diagnostics, not this primary result. `simultaneous_edit_nmse` is a separate operation that attenuates all requested layers together.

## Matched baselines

Every non-oracle method uses the same scenes, queries and edit. Lower error is better. Present/absent strata average within each scene first; there are 54 present requests across 20 scenes and six absent requests across six scenes.

| Method | All-request edit NMSE | Present edit NMSE | Absent edit NMSE | Simultaneous edit NMSE |
|---|---:|---:|---:|---:|
| Independent AudioSep | 0.0593777 | 0.0558412 | 0.0858368 | 0.125732 |
| Unweighted consistency | 0.0496482 | 0.0478291 | 0.0663866 | 0.0381661 |
| Residual Wiener | 0.0484281 | 0.0489782 | 0.0435069 | 0.04152 |
| Fixed scalar 0.75 | 0.0476391 | 0.047 | 0.0482832 | 0.0563739 |
| Selection scalar 0.60 | 0.047738 | 0.04927 | 0.0309013 | 0.0512975 |
| Sequential forward | 0.0586598 | 0.060771 | 0.0530934 | 0.0361817 |
| Sequential reverse | 0.0476989 | 0.0477528 | 0.0427724 | 0.0303089 |
| Trained V1 | 0.0441843 | 0.0442537 | 0.0404449 | 0.0284783 |

FUSS is a separately trained, four-output universal separator. The evaluator alone uses clean references for one-to-one assignment to **present requests**. Its matched present-only edit NMSE is **0.0512266**, versus trained **0.0442537** on the same present-only scene averaging. FUSS has no semantic query/absence detector; its absent placeholders are not successful detections, and it has no comparable all-request or simultaneous score. Its native 16 kHz mono output is resampled to 32 kHz for alignment without restoring high frequencies. This is not a common-16-kHz scoring experiment or a runtime text-extraction comparison.

## Gain, activity and absence tradeoffs

Direct SDR measures waveform error including wrong level; higher is better. SI-SDR removes fitted scale and is reported separately. Gain error is absolute deviation of the estimate's projection onto the target from unity; lower is better. Activity error compares 100 ms power envelopes, not human-perceived identity. Absence false extraction means estimate energy divided by mixture energy exceeds the frozen **0.01** threshold.

| Method | Direct SDR dB | SI-SDR dB | Gain error | Activity NMSE | Activity-change NMSE | False extraction / 6 |
|---|---:|---:|---:|---:|---:|---:|
| Independent AudioSep | 3.60593 | 1.44257 | 0.226299 | 0.12219 | 0.0784718 | 5 |
| Unweighted consistency | 2.9725 | 0.714646 | 0.358642 | 0.0957684 | 0.0578171 | 6 |
| Residual Wiener | 3.60565 | 0.787061 | 0.489381 | 0.139067 | 0.0784519 | 4 |
| Fixed scalar 0.75 | 3.08966 | 1.44257 | 0.419718 | 0.129541 | 0.0771251 | 5 |
| Selection scalar 0.60 | 2.56048 | 1.44257 | 0.535774 | 0.18756 | 0.107207 | 5 |
| Sequential forward | 3.24896 | -0.99927 | 0.466684 | 0.153127 | 0.0835384 | 2 |
| Sequential reverse | 4.00661 | -0.169435 | 0.443108 | 0.106154 | 0.0479764 | 4 |
| Trained V1 | 3.14469 | 1.50604 | 0.472434 | 0.131065 | 0.0752269 | 5 |
| FUSS present-only oracle | 2.39745 | -14.3736 | 0.602053 | 0.104766 | 0.0770599 | n/a |

Trained V1 worsened mean direct SDR and gain accuracy versus independent extraction; gain error increased for **53/54 present requests**. Its mean activity error also worsened, despite lower activity-change error. Five of six absent targets still crossed the false-extraction threshold. The absent rooster in `confirmation-017` retained energy equal to **0.471536** of mixture power. Reduced absent energy is not reliable absence handling.

The following diagnostics explain error allocation; they are not proofs of pure source ownership. Retained-audio projection is the squared fitted coefficient of edit error along `mixture − target`, averaged across requests. It is not projected-error energy divided by mixture power, despite the legacy JSON field's `_nmse` suffix. The target/orthogonal split projects onto the true edited target change; its orthogonal remainder is not necessarily a pure non-target source. The duplicate measure fits estimates against all clean reference recordings and subtracts true shared attribution. All 20 bases passed the declared condition bound; weak clip labels still limit physical interpretation. Lower values are better.

| Method | Retained-audio coefficient² | Target-projection edit NMSE | Orthogonal edit NMSE | Excess source-duplicate energy |
|---|---:|---:|---:|---:|
| Independent AudioSep | 0.0293935 | 0.0111994 | 0.0481784 | 0.250509 |
| Unweighted consistency | 0.0184327 | 0.014776 | 0.0348723 | 0.149256 |
| Residual Wiener | 0.00953575 | 0.0265018 | 0.0219263 | 0.0438168 |
| Fixed scalar 0.75 | 0.016542 | 0.0205388 | 0.0271003 | 0.140911 |
| Selection scalar 0.60 | 0.010593 | 0.0303938 | 0.0173442 | 0.0901834 |
| Sequential forward | 0.0161189 | 0.0302344 | 0.0284255 | 0.0251714 |
| Sequential reverse | 0.0119905 | 0.0240812 | 0.0236178 | 0.0360332 |
| Trained V1 | 0.0119783 | 0.0226926 | 0.0214917 | 0.0932894 |
| FUSS present-only oracle | 0.0252446 | 0.0259078 | 0.0253187 | 0.0271113 |

FUSS diagnostics in this table use present requests only. V1 reduces attributed duplicate energy versus independent extraction, but both sequential methods and residual Wiener score better on this diagnostic. Scalar 0.60 also slightly outperforms V1. See [metric definitions](evaluation-metrics.md) and exact per-query metrics in [per-recipe JSONL](../evaluation/v1/per-recipe.jsonl).

## Individual failures and dependence

Against scalar 0.75, trained V1's paired scene median improvement is **+0.00296255**, but its request edit error increased for sea waves in `confirmation-005` (**0.0472794 → 0.0772084**), crying baby in `confirmation-003` (**0.0391305 → 0.0626558**), and chainsaw in `confirmation-018` (**0.0233835 → 0.0396322**). These are real present-target regressions. The complete [scene/request tables](../evaluation/v1/summary.md#wins-and-regressions) retain failures rather than selecting favorable examples.

Against sequential reverse, V1 wins only **8/20** scene means and its paired median improvement is **−0.00233180**, although its aggregate mean is lower. Omitting every scene containing any take of original recording **215658**, **216370**, or **234879** reverses that mean advantage; remaining mean improvements are respectively **−0.00292677**, **−0.000619400**, and **−0.000945689**. These overlapping-group omissions are descriptive sensitivity, not confidence intervals. Favorable source-omission sensitivity against fixed scalar 0.75 does not establish robustness against every baseline.

The 60 source slots reuse 45 files from 35 original recordings and 33 uploaders. Thirteen files recur, up to four times; requests and scenes are dependent. Source-file sharing connects scenes in groups of sizes 10, 7, 1, 1 and 1. These counts do not estimate independent sample size or statistical certainty. No source-cluster confidence interval is implemented. Weak class labels do not guarantee pure physical sources; uploader separation does not certify speaker/session isolation; base-separator pretraining overlap is unknown. Controlled recombinations are not natural multitrack truth.

## Runtime and interpretation boundary

The run used an Apple Silicon Mac with 32 GiB RAM, AudioSep MPS execution, four CPU threads and observed `aten::col2im` CPU fallback; FUSS used two CPU threads. Independent three-query extraction averaged **6.278 s** per five-second scene; refinement alone averaged **2.883 s**. Cold/warm query caching and concurrent work affect timing. Cumulative maximum process RSS was **789,020,672 bytes** for independent proposals, **695,730,176** for FUSS and **1,082,294,272** for sequential preparation. These are not isolated per-record allocations or total GPU memory. The full evaluation took **389.80 s**; macOS denied the timing wrapper's extended memory query after the complete report was written, so evaluator peak RSS is unavailable.

All 20 scenes completed actual FUSS inference plus 120 actual sequential extraction calls. Independent readback verified parent hashes and exact preservation of original arrays. Actual six-permutation refiner execution per scene gave maximum restored waveform discrepancy **8.94e−8**. Reconstruction and outside-edit sample error were **0**. Those integrity results do not establish separation quality. This five-second source-reference run does not establish long-clip boundaries, stereo quality, natural-scene reference accuracy, listening-panel quality, locks/undo behavior or hardware latency guarantees; separate product checks have separate evidence.

The frozen failure gate states that a nonpositive primary mean improvement over the strongest comparable non-oracle baseline would not establish efficacy. The observed positive mean passes that narrow gate while retaining the distribution, gain, absence and source-dependence limits above. No broad, typical-event or state-of-the-art claim follows. No secondary subset replaces the frozen primary.

See the [portable bundle and exact run recipe](../evaluation/v1/README.md), [full numerical summary](../evaluation/v1/summary.md), and [source capture/provenance](../evaluation/v1/source-capture/provenance.json). Heavy source audio, prediction archives, base checkpoints and tokenizer are excluded from this metadata bundle.
