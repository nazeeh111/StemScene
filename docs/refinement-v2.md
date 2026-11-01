# Coherent peer repair, v2 experiment

This is a separate experiment. It does not replace v1, and it is not an achieved
quality claim. This selection pilot did not open confirmation; V1 was subsequently
frozen before confirmation. The first v1 evaluation's
single selected query was confounded by gain shrinkage; all-request editing is
the relevant broader job. The independent review found v1 stronger on that
broader score, so v1 remains a candidate rather than being discarded.

## Fixed mechanism before training

The network has shared per-query weights and symmetric peer accumulation. For
each distinct pair of requested classes it computes nonnegative local complex
coherence, bounded amplitude overlap, and a bounded coherent peer component.
Similar descriptions with text cosine >= 0.98 do not compete. A single query or
a fully indistinguishable query set is passed through without arbitrary ownership.

Nineteen mixed/proposal features feed a 19→16→16→4 shared MLP. Controls are
pooled and interpolated using v1's STFT geometry. The output is:

```text
R = mixture - sum(proposals)
Y_i = P_i + collision_i * (
       0.25 tanh(a_i) P_i
     - 0.50 tanh(b_i) coherent_peer_i
     + 0.25 [tanh(c_i) + j tanh(d_i)] R)
```

The final layer starts at zero, hence identity. There is no exclusive softmax,
presence flag, target waveform, or source-reference delay correction at inference.
The peer correction can remove correlated attribution without requiring all
audio in the proposal to be attenuated. Whether it learns that useful behavior
is an empirical question, not established by this equation.

The base class supplies explicit float64 residuals, mixed-recording anchors,
bounded whole-recording context and overlapping long refinement. Query classes
retain fixed indices; physical-source identity is not inferred. Positive-anchor
usefulness and real long-clip quality need separate evidence.

## Fixed training and selection protocol

Before optimization: 12 epochs, learning rate 0.002, AdamW weight decay 0.0001,
seed 1729, four CPU threads, the existing 30 actual AudioSep training records and
10 selection records. Query order is permuted during training. No extra data,
confirmation reference, synthetic proposal or foundation-model update is used.

The loss on every requested query is the mixture-energy-normalized waveform
error plus 0.10 complete-retained-signal error projection, 0.10 target-gain
projection error, 0.02 excess pairwise attribution, and 0.01 envelope-change error.
Retained audio is **mixture minus that target reference**, including sources not
requested. Target references and numerical presence occur only in the loss.

Checkpoint selection uses mean **all-request edit NMSE**: independently apply the
same frozen 1–4 second, 25% gain, 20 ms inside-span fade to each of the three
requested estimates, average within each recipe, then average recipes. Identity
is retained as a candidate. Gain-sensitive SDR, gain error, single-query edits,
unlearned 0.75 gain, v1 and other baselines are reported alongside this primary
selection score. Lower edit error does not excuse hidden loudness loss or prove
subjective quality. No new hard acceptance threshold is invented for this pilot.

The script preserves the initial identity, last optimized and selected states,
source/record hashes, fixed protocol, trace and negative outcomes. It never opens
the confirmation folder. Any revision after this fixed pilot must be a separately
documented experiment. Confirmation can only begin after the owner freezes the
chosen model, methods and reporting protocol.

## Executed selection pilot, 2026-10-29

The fixed run completed all 12 epochs and selected epoch 11. It trained 660
parameters on the existing 30 real-proposal training records and selected on the
10 reserved selection recipes. Its recorded training/selection loop took
1,560.29 seconds on the shared Mac with four CPU threads, excluding imports and
record preparation. Initial identity, last optimized and selected states, the
pre-training protocol, all record hashes and exact four-file source snapshots
were retained. The inherited parent `joint.py` snapshot has SHA-256
`a252f5f2c62a036381a2adabbcb75e83539b99586cad80b87e45ecc3e3938c12`.

Selected v2 artifact SHA-256:
`ef631022391812eae3b1f24af3ce780e0284a2b43a7d67feaaf92c645835ebf8`.
No confirmation record was opened by training or this comparison.

All-request scoring independently edits every requested layer using the same
span, level and fades, then averages within a recipe. This differs from the
original v1 single-query headline. On these ten five-second, three-query mono
selection recipes:

| Method | Mean all-request edit NMSE, lower better | Mean target gain error, lower better | Mean target SDR, higher better |
| --- | ---: | ---: | ---: |
| Independent proposals | 0.061753 | 0.2131 | 2.715 dB |
| Fixed 0.75 gain | 0.045236 | 0.4098 | 1.880 dB |
| Fixed 0.60 gain | 0.042473 | 0.5279 | 1.145 dB |
| Selection-calibrated scalar 0.597682 | 0.042472 | 0.5297 | 1.135 dB |
| Residual Wiener | 0.043862 | 0.5091 | 1.664 dB |
| Unweighted consistency | 0.044851 | 0.3684 | 0.931 dB |
| Trained v1 | **0.039221** | 0.4756 | 1.442 dB |
| Trained v2 | 0.040709 | 0.4245 | 1.909 dB |
| V2 gain-only ablation | 0.048644 | 0.3086 | 2.382 dB |
| V2 without coherent-peer term | 0.045560 | 0.3334 | 2.181 dB |
| Sequential forward | 0.051261 | 0.3907 | 4.414 dB |
| Sequential reverse | 0.054016 | 0.4452 | 2.927 dB |

The scalar is fitted only to selection, not a confirmation oracle. FUSS's raw
four-source output was also scored with evaluator-only optimal matching on
present requests: mean edit NMSE 0.040624, target SDR 3.291 dB. Its denominator
excludes absent requests, it supplies no runtime semantic assignment, and its
native 16 kHz bandwidth differs, so that number is not a full-query equivalent
of the table's methods.

V2 reduced mean all-request edit error by 34.1% versus independent proposals and
4.15% versus the selection-calibrated scalar. It improved the recipe-level
all-request mean in 10/10 cases versus independent and 8/10 versus that scalar;
all leave-one-recipe-out mean differences against the scalar remained positive.
This is a small selection diagnostic with reused sources, not a source-cluster
confidence interval or confirmation claim.

V2 does **not** dominate v1. Its mean edit error is 3.79% higher, although it has
lower gain error, higher SDR/SI-SDR and lower excess-attribution and absence-energy
proxies. It wins 6/10 recipe averages and 22/30 individual requested-layer edits
against v1. Against independent proposals, only 16/30 individual layer edits
improve. A serious failure is the dog request in `selection-005`: edit NMSE rises
from 0.00957 to 0.09146. Aggregate improvements do not make every selected event
safe or correctly isolated.

All three absent requests still exceeded the fixed 0.01 mixture-energy threshold
(100% false extraction). Their mean energy ratio fell from 0.55774 independent
to 0.12082 v2, but this is not successful semantic absence detection. V2's mean
SI-SDR is 0.633 dB versus 0.766 dB independent and 0.361 dB v1. Loudness and source
fidelity remain imperfect; these automatic metrics are not human listening
judgments.

Removing only the coherent-peer term worsened all-request edit error in all ten
recipes, as did retaining only the learned gain term. Those selection ablations
support a contribution from the joint correction on these records, rather than
only a global level change. They do not establish broad scientific novelty.
Actual reverse-order tests on all ten records and all six permutations on the
first three produced maximum absolute waveform difference 8.94e-8.

The comparison process's cumulative peak CPU RSS was 598,114,304 bytes with both
models and evaluator arrays resident. Observed median CPU refinement time per
five-second record was 1.381 seconds for v1 and 2.074 seconds for v2. These are
shared-machine observations, not isolated per-model memory measurements or live
streaming claims. Real long-recording quality, natural-source accuracy and
positive-anchor usefulness were not measured by this selection pilot.

Task-local evidence is `evidence/training-v2/` and
`evidence/selection-v2-comparison/{report.json,per-recipe.json}`. Eight structural
checks passed across the initial seven-test run and a separate silence/bounds
check. The owner must choose the candidate and freeze the final protocol before
opening confirmation. This pilot does not authorize choosing a model after
seeing confirmation outcomes.

Independent review qualified the peer-term ablation: 91.1% of the summed
request-slot improvement over disabling that term came from the three absent
requests. With a matched present-request denominator averaged by scene, full v2
had edit NMSE 0.044754 versus 0.044368 without the peer term, a slight regression.
The positive all-request ablation therefore chiefly supports reduced false
attribution to absent requests on this selection set; it does not establish
better present-source isolation. Both denominators and this negative result
must accompany the ablation claim.
