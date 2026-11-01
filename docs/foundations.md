# Foundations and scope

Primary sources and live repository revisions were inspected on 2026-10-29.
Author-reported performance is not a local replication. Local executed comparisons
and their failures are recorded separately in [the confirmation report](comparisons.md).

## Backend choices and prior work

[AudioSep](https://github.com/Audio-AGI/AudioSep) already provides text-conditioned
extraction and chunking. STEMSCENE retains its pretrained separator architecture
and frozen CLAP text representation. The selected source revision is
`944583f18b84589dc965de3ad77525c945334252`. Its Linux/CUDA environment file was not
used as an Apple installation recipe. An explicit local loader avoids eager
encoder initialization and network access; a separately verified
[Torchlibrosa compatibility adaptation](stft-adaptation.md) removes an initialization
bottleneck without changing convolution state or operations. Actual CPU and
optional MPS execution were tested, including correctly aligned exports.

AudioSep code is MIT. Separately, the official
[weight-host metadata](https://huggingface.co/spaces/Audio-AGI/AudioSep/raw/main/README.md)
declares MIT; no standalone weight-specific text was found. This is reported as
host metadata, rather than an invented separate agreement. The pinned checkpoint
already includes the required frozen text weights. Tokenizer assets are downloaded
explicitly from the pinned public source. See [notices](../THIRD_PARTY_NOTICES.md).

[SAM Audio](https://github.com/facebookresearch/sam-audio) already supports text,
visual and positive/negative temporal prompts, optional span prediction and
candidate reranking. Its documented path recommends CUDA and its official weights
require granted access. That access was not assumed or bypassed. SAM Audio is
optional prior art, not a required runtime or an executed comparator here.

[Unified Separation and Extraction, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40635)
jointly models universal separation and clue-driven extraction using attractors,
multimodal clues and cross-task consistency. It directly limits broad novelty
claims. The inspected author demo was not treated as a locally runnable checkpoint.

[FUSS TDCN++](https://github.com/google-research/sound-separation) already uses
weighted mixture consistency. Its accessible dry checkpoint was executed locally
as a second, independently trained universal separator. It produces four unlabeled
16 kHz mono sources. Reference assignment is evaluator-only, present-only and
unavailable to the product; it is not semantic absence detection. Code is
Apache-2.0 and the [deposited checkpoint](https://doi.org/10.5281/zenodo.4012661)
has separate CC BY 4.0 terms.

The bounded alternative search also inspected CLAPSep, CodecSep, OmniSep and a
speech-only MetricGAN+ option. Missing runnable assets, unresolved separate
weight/code terms, extra dependencies or task mismatch prevented treating them
as tested drop-in alternatives. Listings and author demos were not counted as
executed capabilities.

Text prompting, chunking, temporal anchors, reranking and mixture consistency
are established methods. STEMSCENE's experimentally assessed question is whether
a compact query-set-aware trained correction improves multi-layer edit integrity
within a persistent class-layer editor. Its small frozen study supports a narrow
mean improvement with substantial failures, not a field-wide novelty conclusion.

## Existing portfolio projects

| Checked project | Inspected main revision | Different input and job |
| --- | --- | --- |
| [EventizedAudio](https://github.com/nazeeh111/EventizedAudio) | `dab02fc3281fcd1ccdb3847981be1d18a136d9f0` | Event-camera optical vibration reconstruction, including supplied-reference alignment |
| [MotionizedAudio](https://github.com/nazeeh111/MotionizedAudio) | `69b38f7722137c204db485baf2149419fd037abc` | High-speed video surface vibration to audio |
| [SoundMap](https://github.com/nazeeh111/SoundMap) | `c287d8cd22947b9695bb46241b1df32ad047e93b` | Microphone-array source localization and strength maps |

Those source trees and relevant notices were read. Their optical/sensor inputs
and reconstruction/localization jobs do not supply the ordinary-recording,
persistent requested-layer editor built here. No source, history, visibility or
attribution in those repositories was changed, and none of their scientific
pipelines was copied into STEMSCENE. This source-scope distinction does not
establish scientific novelty across other projects.

The requested GitHub name returned 404 on the checked account before creation.
The exact PyPI normalized name returned 404 during initial research; no package
publication or trademark clearance is claimed. Namespace availability is a dated
observation, not a reservation.
