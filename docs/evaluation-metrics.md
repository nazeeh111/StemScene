# Source-reference evaluation

`scripts/evaluate_refiner.py` evaluates real backend proposal records. It does not create proposals or train a separator. Analytic sine fixtures in tests establish metric correctness only. A passing unit test is not empirical audio-quality evidence.

Each NPZ needs `mixture[N]`, query-aligned `proposals[K,N]`, `references[K,N]`, `embeddings[K,512]`, and scalar `sample_rate`. Its sibling JSON needs `split`, `recipe_id`, `source_ids` (all original clip filenames), `backend_checkpoint_sha256`, `frozen_recipes_sha256`, `record_sha256`, `queries`, `presence_by_request` and the frozen `edit`. Observed timing is preserved from `inference`, memory from `peak_rss_bytes` plus `peak_memory_note`, and settings from `backend_settings`. The macOS memory observation is a cumulative process maximum, not an isolated per-record peak. Absent requests have zero references. Only the mixture, proposals and embeddings are supplied to `JointRefiner.refine`; labels/references stay in this evaluator. Optional `sequential_forward` and `sequential_reverse` are actual backend extraction results mapped back to the original query axis. Optional `fuss_all_sources[4,N]` need `fuss_checkpoint_sha256` metadata and actual raw universal-separator outputs, with explicitly evaluator-only optimal reference assignment.

The evaluator checks the committed frozen-recipe SHA-256 and training checkpoint provenance. Every record's source filenames, queries, presence and edit must match its frozen recipe, and duration must match after resampling. Confirmation refuses before opening proposal data unless `--decision-freeze` supplies a JSON record with `frozen: true`, `selection_finished: true`, matching `checkpoint_sha256` and matching `frozen_recipes_sha256`. Freeze all model/threshold/epoch decisions before creating that record. The output directory must be fresh; failures retain already written per-recipe results. Complete recipe coverage is required before emitting an aggregate, except when `--allow-partial` explicitly labels an incomplete pilot. Duplicate recipe IDs always fail.

Methods are independent backend proposals, known unweighted mixture-consistency projection, a stronger known residual-aware Wiener baseline, trained refinement, and supplied sequential/FUSS outputs. The unweighted projection assigns all mixture energy to requested stems, so it is weak for incomplete query sets. The Wiener method computes proposal STFT powers plus the power of `mixture - sum(proposals)`, uses their relative powers as mixture masks and retains the remainder as residual. It uses periodic Hann 1024-point STFT, hop256, centered constant padding and exact requested output length. These baselines are not claimed as original inventions.

Gain-sensitive direct SDR is `10 log10(sum(reference²)/sum((estimate-reference)²))`, with a relative error floor of `1e-12` and scores capped to ±120 dB. It preserves gain/DC error. SI-SDR mean-centers both signals, projects the estimate onto the reference, and compares that projection with its remaining error. Wrong scale can receive a high SI-SDR, so report direct SDR and target gain together. Reference mean power ≤`1e-12` is numerical silence: SDR is unavailable. A non-silent reference with silent estimate receives SI-SDR −120 dB.

The original single-query edit diagnostic compares `mixture + (envelope-1)*estimate` with the same expression using the clean target. The envelope follows the product's linear amplitude fades inside a half-open selected span. `edit_nmse` normalizes whole-edit MSE by mixture mean power. It therefore penalizes both wrong target attenuation and contamination removed from other sources. `retained_source_projection_nmse` is the squared coefficient `dot(error, retained) / dot(retained, retained)`, where `retained = mixture-clean_target`. The all-request version averages those squared coefficients. Despite the legacy `_nmse` names, they are not projected-error power normalized by mixture power; they can miss correlated/orthogonal distortion. `outside_edit_max_delta` checks exact untouched samples separately.

The frozen confirmation primary is **`all_request_edit_nmse`**, under protocol
`all-request-separate-and-simultaneous-v2`. Apply the same 1–4 second, 25% gain,
20 ms fade separately to each of the three requested estimates, including absent
requests; average their normalized squared errors within each scene, then
average scenes. Present/absent strata retain scene-first averaging and report
their measured denominators. The older `edit_nmse`, paired-improvement fields
and `failure_flags` retain the original request-index-zero diagnostic and must
not be mistaken for the all-request primary. Per-query errors remain available.

The separately declared `simultaneous_edit_nmse` attenuates all requested layers
together while leaving the residual unchanged. Its error is
`(envelope-1) * sum(estimate-reference)`. It is a different operation, not extra
independent samples. Fixed 0.75 scaling and selection-calibrated 0.60 scaling
are additional unlearned controls, frozen before confirmation.

The evaluator also splits edit error into a projection onto the true edited
target change and its orthogonal remainder. This distinguishes wrong target
level from other distortion without calling the remainder a pure non-target
source. A supplementary source-conditioned diagnostic projects estimates onto
all clean references using a Gram-matrix least-squares solution, subtracts true
shared attribution and measures duplicate attributed energy. Ill-conditioned
bases (condition number at least `1e8`) return unavailable attribution. References
remain evaluator-only; these diagnostics do not establish physical-source purity.

Absent-target extracted energy is estimate power divided by mixture power. False extraction uses a fixed threshold, default0.01; choose any alternative on selection and freeze it before confirmation. Labels indicate source membership, while numeric silence is reported separately. Excess duplicate attribution compares absolute off-diagonal waveform covariance of estimates with true references, normalized by mixture power². It subtracts natural reference overlap but remains a correlation diagnostic, not proof of exclusive physical identity.

Activity metrics compare 100 ms mean-power envelopes and their first differences, normalized by mixture power². These quantify envelope errors rather than human-perceived identity. Boundary error is measured only when observed `chunk_boundaries_samples` metadata exists, using ±20 ms join windows and separate non-boundary windows. Five-second records without joins cannot establish long-clip continuity.

Resynthesis error recombines predicted layers and the explicit algebraic residual. Exact reconstruction does not establish correct isolation. The evaluator checks all learned-query permutations for K≤3, otherwise reversal, using a mapped maximum waveform difference. Sequential-order RMS compares supplied real forward/reverse residual extractions. Optional FUSS assignment uses clean references only for one-to-one matching to **present** requested sources. This method is reported as `fuss_present_evaluator_oracle`; it has no semantic query assignment or absence detector. Absence scores and any edit selecting an absent request are unavailable, and other source-quality metrics score present requests only. Its denominators can therefore differ from the text-conditioned methods. Native FUSS is 16 kHz and its outputs may be resampled to the record rate: a 32 kHz comparison does not restore high-frequency information or establish a matched common-16 kHz comparison. Preserve this limitation beside results.

Reports retain per-query/per-recipe values, paired trained-minus-baseline edit results and every recipe where refinement worsens edit integrity versus independent or Wiener estimates. Aggregate means, medians, minima and maxima include measured denominators. No source-cluster confidence intervals are implemented, so small pilots must not imply statistical certainty. Inference timing/memory are reported only from actual record metadata; evaluator refiner timing is measured separately and does not include base separation. Missing sequential/FUSS/long/stereo/natural reference, lock/undo or listening-panel evidence remains an explicit limitation.
