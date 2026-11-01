# Joint proposal refinement

This is a compact experimental refinement method over real pretrained separator
outputs. It does not replace the pretrained separator, generate speech, identify
speakers, or establish scientific novelty by itself. Release claims must follow
the frozen confirmation comparison, including negative results.

## Computation

The engine supplies a mono mixture, one aligned proposal per requested sound
class, and the corresponding 512-dimensional text embeddings. A shared 579
parameter network predicts a smooth proposal gain and two components of a
bounded complex correction coefficient. There are no learned query slots.

The network sees 15 features per query and time-frequency region: proposal versus
mixture magnitude, mixture-compatible phase, symmetric peer mean/sum/maximum,
prompt-similarity-weighted peer magnitude, the proposal's whole-clip spectral
profile, temporal activity, mixture magnitude, local activity change, normalized
frequency, signed user-anchor spans, mixed-audio positive-anchor spectrum,
maximum other-prompt similarity, and query count. Frequency/time average pooling
reduces cost; interpolated controls vary smoothly. Input analysis uses a 1024
sample periodic Hann window, hop 256, centered constant padding, and no delay
correction. At 32 kHz those are 32 ms windows and 8 ms hops.

For proposal spectrum P_i and mixture spectrum X, the correction is:

```text
R = X - sum_i(P_i)
gain_i = 1 + 0.75 tanh(a_i)
coefficient_i = tanh(b_i) + j tanh(c_i)
Y_i = gain_i P_i + 0.25 / sqrt(query_count) coefficient_i R
```

The corrected estimates are inverse-transformed to the exact input sample
length. The explicit residual is computed afterward in float64 as original
decoded mixture minus the sum of corrected estimates. That algebraic
reconstruction property is separate from separation accuracy. No normalization,
target reference, presence label, gain adjustment from a clean target, or delay
matching to a clean source is used during inference.

Symmetric query aggregation and shared weights make the method equivariant to
query permutations: permuting input queries permutes their outputs, up to
floating-point summation error. The query set still matters. Adding a query can
change unlocked refinements; the persistent Project layer must enforce user
locks and report incompatible updates. For simultaneous-versus-sequential
comparison, generate the same proposals from the full original mixture and
refine the complete canonical set. Sequential *residual extraction* is a
separate baseline and is expected to depend on extraction order.

## Identity, anchors, and ambiguity

The full-clip spectral profile and a fixed query embedding provide a persistent
class context through the pretrained backend's chunks. They do not establish an
identifiable physical source. A request for "dog barking" can include multiple
dogs. Similar or duplicate descriptions must be grouped by the Project layer,
given a useful user anchor, or reported as uncertain. This network deliberately
does not impose an exclusive softmax over queries. Prompt cosine similarity
above 0.98 is exposed as an overlap warning; it is not confidence in isolation.

Anchors are intervals in the mixed recording. Positive-anchor spectra come from
the mixed audio itself and can contain other sounds. Present anchors add context;
they are not proof that a physical source has been disambiguated. Absent anchors
apply deterministic suppression inside the requested span, with 10 ms internal
fades. Contradictory overlapping present/absent anchors are rejected. Anchor
conditioning has not been established by passing structural tests; evaluate it
on real proposals and user-grounded spans before claiming an improvement.

This implementation uses whole-clip context and is offline, not live or causal.
The direct `refine` resource bound is eight queries and 960,000 mono samples,
which is 30 seconds at 32 kHz. `refine_long` delegates to that same direct method
for short inputs. For longer inputs, it accumulates the same class spectral
profiles in bounded centered-STFT frame batches across the entire mixed audio
and proposals. Positive-anchor mixed spectra are accumulated in the same pass.
No full-recording spectrogram or source reference is required.

Long refinement then uses overlapping windows with those shared global profiles,
global mixture magnitude scale, stable query embeddings, and overlap addition.
Defaults are 20-second windows and 2-second overlaps, with analysis-hop-aligned
starts. The per-window sample count remains bounded by 960,000, so the actual
window duration decreases at higher sample rates. Negative-anchor envelopes are
applied once in global sample coordinates after overlap addition; a chunk edge
cannot restart their fades. The original sample rate determines all durations.

Class association is explicitly the persistent query index. At every window
overlap, the engine reports same-query zero-lag waveform correlation and the
cross-query correlation matrix. High competing correlation is marked ambiguous;
it never causes an order-dependent reassignment. Correlation is diagnostic, not
proof of physical source identity or correct isolation. Identical descriptions
can remain indistinguishable. Query permutations also permute the association
diagnostics rather than changing a matching decision.

The long path rejects clips over 600 seconds and audio-array requirements over
4 GB, conservatively counting caller mixture/proposals, output, residual, overlap
weights, input conversion and sum temporaries, before allocating its output. A
bounded per-window model workspace is additional.
High-rate recordings with many queries can hit the byte limit before 600 seconds.
These are admission bounds, not a measured peak-memory guarantee. Structural
tests verify the streaming profile matches a full STFT, long-path permutation
symmetry, overlap coverage, one global anchor envelope and residual reconstruction.
The subsequent [48-second real-proposal stress test](long-audio.md) measured
latency, memory, activity, boundaries and attribution failures. It is one
development case and does not establish general long-recording quality.

## Training record contract

`scripts/train_refiner.py RECORD_ROOT NEW_OUTPUT_DIR` reads only
`RECORD_ROOT/train/*.npz` and `RECORD_ROOT/selection/*.npz`. It never opens the
confirmation directory. Each record contains:

| Array | Shape | Meaning |
| --- | --- | --- |
| `mixture` | N or N×1 | Mixed audio presented to the separator |
| `proposals` | K×N or K×N×1 | Actual backend inference results |
| `references` | K×N or K×N×1 | Training/evaluator-only source references, zero for absent requests |
| `embeddings` | K×512 | Text embeddings, with no presence labels |
| `sample_rate` | scalar | Declared sampling rate |

A same-stem JSON sidecar is mandatory with `split`, `recipe_id`, `source_ids`,
`backend_checkpoint_sha256`, `frozen_recipes_sha256`, and `record_sha256`. The
trainer verifies each proposal archive's recorded SHA-256. Optional provenance can
include exact backend settings, model code revision, audio hashes, and source
license attribution. The script rejects overlapping original recording IDs
between training and selection, duplicate recipe IDs, mixed recipe-freeze
hashes, and mixed backend checkpoint hashes. The separate data preparation and
confirmation evaluation must enforce uploader/session/recipe isolation as
appropriate; the trainer does not infer those properties from an audio tensor.

Records must use genuine source audio and real backend proposals. A controlled
mix of genuine recordings is not naturally captured multitrack ground truth.
Synthetic/random proposals are useful only for correctness tests and are not
valid evidence for the trained method. Pretrained backend exposure to ESC-10 is
unknown, so these splits establish refiner evaluation isolation, not foundation
model unseen-source generalization.

The v1 training loss is gain-sensitive waveform error plus a small projection of
error onto other requested references, excess pairwise waveform attribution, and
envelope-change error. Completely attenuating an estimate introduces exactly its
waveform estimation error into the edited original, so gain-sensitive error is a
direct edit-integrity objective. Absence is represented only by a zero reference
in the loss. No target presence flag is passed to `JointRefiner.forward`.

The v1 auxiliary contamination term omits sources present in the mixture but
absent from the request set, because it uses the sum of requested references.
The primary waveform error still penalizes their leakage into each estimate.
The independent edit evaluator uses the complete retained audio
`mixture - target_reference`. This auxiliary-loss limitation must remain visible
in v1 results; correcting it is a separate, documented experiment, not a silent
change to a frozen training run.

The initial network passes proposals through. Only training and selection data
choose the checkpoint; query order is randomly permuted during training. The
untrained identity state remains an eligible candidate. `selected_epoch=-1` and
`selection_improved=false` mean the experiment failed to improve selection and
must not be presented as a successful trained refiner. The last genuinely
optimized checkpoint is retained separately for reporting. Training writes the
seed, hyperparameters, trainer/module hashes, proposal file hashes, recipe-freeze
hash, selection trace and selected epoch. It also preserves the initial
`identity.pt` artifact separately from `selected.pt` and `last.pt`.

## Comparisons and evidence

Compare independent extractions, sequential residual extraction, unlearned
mixture-consistency projection, and this trained method using identical backend
weights, mixtures and query sets. The implemented unlearned baseline is the
known unweighted Euclidean projection:

```text
Y_i = P_i + (X - sum_j(P_j)) / query_count
```

It assigns all mixture energy to the requested layers; its explicitly exported
residual is numerically zero. Report its limitation for incomplete query sets.
Mixture consistency is established prior work, not this project's invention;
see [Differentiable Consistency Constraints for Improved Deep Speech Enhancement](https://arxiv.org/abs/1811.08521).

Structural tests exercise permutation symmetry after nonzero learned controls,
gradients, silence, anchor conflicts, exact output length, residual reconstruction,
resource bounds, safe weight-only loading, and the projection identity. These
tests do not establish audio quality, anchor usefulness, physical identity
tracking, training improvement, or human listening judgments. Real confirmation
results and failure examples must be recorded separately before release.

## Executed v1 selection result

The first genuine training run used the frozen 30 training and 10 selection
recipes, actual AudioSep proposals, 12 epochs, learning rate 0.002, seed 1729 and
four CPU threads. It trained 579 parameters in 653.25 seconds on a shared Mac.
Selection chose epoch 8: the training objective on selection fell from 0.234282
to 0.138205. The chosen artifact SHA-256 is
`ddbc872006da68440ad3f72435c3b89ecacf422f2f7265d68581f1dee879a9a9`.
These are selection results, not untouched confirmation results.

| Selection metric | Independent AudioSep | Residual Wiener | Trained v1 |
| --- | ---: | ---: | ---: |
| Mean edit normalized squared error, lower better | 0.040206 | 0.039867 | 0.033968 |
| Median edit normalized squared error, lower better | 0.010482 | 0.031354 | 0.024716 |
| Mean present-target gain-sensitive SDR, higher better | 2.715 dB | 1.664 dB | 1.442 dB |
| Mean present-target SI-SDR, higher better | 0.766 dB | 0.163 dB | 0.361 dB |
| Mean absolute target gain error, lower better | 0.2131 | 0.5091 | 0.4756 |
| Mean absent-request energy / mixture energy, lower better | 0.55774 | 0.04590 | 0.14533 |
| Absent-request false extraction rate at 0.01 energy threshold | 100% | 100% | 100% |
| Mean excess pairwise attribution proxy, lower better | 0.09216 | 0.000817 | 0.006318 |

The lower mean edit error is not broad success. Trained v1 worsened edit error
on eight of ten recipes compared with independent extraction and five of ten
compared with Wiener masking. The median paired improvement was negative
(-0.006757). Both improved recipes edited sneezing, and one difficult recipe
dominated the aggregate gain. Present-target loudness accuracy deteriorated and
all measured absent requests still crossed the false-extraction threshold.
This is a suppression tradeoff and a limited negative result, not evidence of
robust separation quality or a proven novelty claim. Query permutation error
was at most 8.94e-8 in the selected model; that verifies symmetry, not accuracy.

The identity, last optimized and selected artifacts, all training/selection
record hashes, executed source snapshots, complete trace and per-recipe failures
were retained. The original `last.pt` records the shared experiment's best
selection loss alongside an overridden last-epoch number; use the trace's epoch
12 value (0.141374) when discussing that last optimized state. The primary
`selected.pt` provenance correctly records epoch 8 and its selection loss.
No confirmation data was read for this training run. The separately documented
V2 selection pilot followed; V1 was subsequently frozen and evaluated on
[all requested confirmation edits](comparisons.md). The historical table above
uses the original request-index-zero diagnostic and is not that final primary.
