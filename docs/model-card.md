# V1 refiner model card

The bundled `joint-refiner-v1.pt` is a **579-parameter experimental joint refiner**
for real AudioSep proposals. It improves a declared mean edit-integrity measure
on a small frozen controlled study, while worsening many individual estimates
and target loudness. It is suitable for reviewed sound-design experiments with
the original recording retained. It does not establish clean source isolation,
physical identity, successful absence detection or general scientific novelty.

## Artifact and training

| Field | Value |
| --- | --- |
| Architecture | `joint-mask-residual-v1`, shared controls and symmetric query aggregation |
| Artifact SHA-256 | `ddbc872006da68440ad3f72435c3b89ecacf422f2f7265d68581f1dee879a9a9` |
| Trainable parameters | 579; pretrained AudioSep/CLAP weights remain frozen |
| Training | 12 epochs, seed 1729, AdamW, learning rate 0.002, weight decay 0.0001 |
| Selected checkpoint | Epoch 8, selected before confirmation |
| Training/selection objective | Selection loss 0.234282 at identity to 0.138205 selected |
| Source data | Licensed ESC-10 recordings; 30 training, 10 selection and 20 confirmation recipes |
| Recipe manifest SHA-256 | `bd19a599b3165a0648e1ed9fef17e085e18fd4878f090e25e24945454c3d0728` |
| AudioSep checkpoint SHA-256 | `f8cda01bfd0ebd141eef45d41db7a3ada23a56568465840d3cff04b8010ce82c` |

These are controlled mixtures of genuine recordings, with original
recording/uploader/hash separation between splits. Within a split, sources recur.
Weak class labels do not guarantee a pure physical source. Speaker/session
isolation is not certified, and AudioSep's possible pretraining exposure is
unknown. Clean references and presence labels enter training loss and evaluation
only. The inference API receives mixed audio, proposals, text embeddings and
optional anchors in the mixed recording. It never receives reference waveforms
or presence labels.

The V1 auxiliary contamination loss uses other requested references and omits
unrequested sources; its primary waveform loss still penalizes their leakage.
The separate V2 pilot corrected that term but had worse selection edit error,
so V1 remained the frozen candidate. Neither model was switched or retuned on
confirmation. See [the method](refinement-method.md),
[V2 pilot](refinement-v2.md) and [frozen study bundle](../evaluation/v1/README.md).

## Frozen confirmation

Each of three requested layers receives the same 1–4 second attenuation to 25%
gain, with 20 ms internal fades, separately. Normalized squared error against
the corresponding clean-reference edit is averaged within each scene, then
across 20 scenes. Requests include 54 present and six absent targets.

Trained mean error is **0.0441843**, versus **0.0476391** for fixed 0.75 proposal
scaling, the strongest comparable non-oracle mean among the predeclared
baselines. That is **7.25% lower**, with 16/20 scene wins and 31/60 request wins.
The remaining **29 individual edits regress**; only 26/54 present requests
improve against that scalar control. Sequential reverse extraction wins in
12/20 scenes despite a worse overall mean.

Mean direct SDR falls from 3.606 to 3.145 dB versus independent extraction.
Mean target gain error increases from 0.226 to 0.472 and worsens for 53/54 present
requests. Five of six absent requests still cross the fixed 0.01 extracted-energy
threshold. Reduced error or energy does not make those targets correctly absent.
FUSS uses evaluator-only present-reference assignment and has a different scope;
it is not a runtime semantic selector or absence detector.

The 60 source slots use 45 files from 35 original recordings and 33 uploaders,
with only five connected source-file scene groups. Group-omission diagnostics
against the scalar remain positive, but comparisons with sequential reverse are
less robust. There is no statistical confidence interval or listening panel.
Read [all methods, denominators, failures and resource measurements](comparisons.md).

## Runtime and user controls

The learned path accepts mono recordings, resamples to 32 kHz, and restores the
original declared sample count/rate for storage and editing. The local core keeps
source bytes, float64 arrays, stable class IDs, locks, explicit residuals and
undoable selected-span gains. Unity reconstruction is an algebraic contract,
not evidence of separation accuracy.

Long refinement uses full-recording context and overlapping bounded windows.
It is offline. A 48-second development case ran locally and exposed substantial
clock contamination, incorrect dominant class attribution and a quieter sneeze
estimate despite better average edit error. Stable query indices and smooth
overlaps do not establish correct identity. See [long-audio evidence](long-audio.md).

Exact normalized duplicate requests group. Automatic overlap warnings use an
unchanged 0.98 prompt-cosine threshold and missed tested near-synonyms such as
`clock tick`/`clock ticking`. Users should group similar descriptions, inspect
estimates and use useful mixed-recording anchors. Negative anchors can suppress
a marked span; positive-anchor isolation improvement has not been established.
Locked estimates retain their saved samples when requests are added; incompatible
locks and changed anchors are reported.

CPU inference is verified. Optional Apple MPS AudioSep inference is verified with
some CPU fallback, while refinement remains on CPU. An independent wheel run on
a different 15.70-second natural recording took 328.28 seconds including loading
and reached 3.20 GiB process RSS. The tested Mac had 32 GiB RAM; minimum hardware
is not established. The 600-second admission ceiling is not a quality guarantee.

## Distribution and limits

The newly trained refiner artifact is distributed under the project's MIT
license, with training attribution retained. This does not relicense AudioSep,
FUSS, tokenizer assets or source recordings. Code and weight-host terms are
recorded separately in [third-party notices](../THIRD_PARTY_NOTICES.md).

Normal extraction uses explicit local files and does not upload recordings.
Only the user-invoked public asset downloader accesses model hosts. There is no
voice cloning, speech regeneration, hidden normalization, causal streaming or
evaluated stereo separation. Review each estimate before applying it to work
where wrong attribution or level would matter.
