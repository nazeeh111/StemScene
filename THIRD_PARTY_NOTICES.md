# Attribution and separately licensed assets

STEMSCENE's original code is MIT. That does not relicense model weights, datasets or the components below.

The newly trained compact `joint-refiner-v1.pt` artifact is distributed under
STEMSCENE's MIT license. Its ESC-10 training attribution and separate pretrained
backend terms remain applicable and are recorded in [the model card](docs/model-card.md).

## AudioSep

The retained `src/stemscene/_vendor/audiosep/{base,resunet}.py` derives from [Audio-AGI/AudioSep](https://github.com/Audio-AGI/AudioSep), commit `944583f18b84589dc965de3ad77525c945334252`, Copyright Xubo Liu, MIT. Its full notice is adjacent in `LICENSE`. Imports were adapted; the trained separation architecture and weights are attributed to AudioSep, not invented here. The text inference implementation follows the checkpoint's frozen CLAP RoBERTa text branch, projection and normalization.

The official checkpoint is downloaded separately. Its hosting [Hugging Face Space README](https://huggingface.co/spaces/Audio-AGI/AudioSep/raw/main/README.md) declares `license: mit`. No separate weight-specific license text was found. This is repository-level weight-host metadata, distinct from inspection of the code's MIT license. The downloader retains both pieces of evidence. Checkpoint SHA-256: `f8cda01bfd0ebd141eef45d41db7a3ada23a56568465840d3cff04b8010ce82c`.

The checkpoint includes CLAP, described by Yusong Wu and colleagues in *Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation*. RoBERTa tokenizer assets come from [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base), whose model card declares MIT. No separately downloaded RoBERTa weights are required.

## Torchlibrosa

`src/stemscene/_vendor/audiosep/stft.py` adapts Torchlibrosa 0.1.0's convolution STFT/ISTFT implementation, Copyright 2018–2021 Qiuqiang Kong, MIT. Full notice: adjacent `TORCHLIBROSA_LICENSE`. SciPy/NumPy replace initialization helpers that otherwise trigger Librosa/Numba compilation. The convolution state and outputs were compared against the original algorithms. See `docs/stft-adaptation.md`.

## FUSS optional reference comparator

`fuss.py` adapts the tensor contract and restoration approach from Google Research's [sound-separation inference.py](https://github.com/google-research/sound-separation), Copyright 2020 Google LLC, Apache License 2.0. The separate FUSS dry checkpoint is from [Zenodo 4012661](https://doi.org/10.5281/zenodo.4012661), whose deposit metadata declares CC BY 4.0 and explicitly includes the model archive. It is not licensed as MIT by STEMSCENE. FUSS produces unlabelled sources, not text-selected stems. Its source assignment in the reference evaluation is evaluator-only and is not used in the production engine.

## Data and demonstrations

The controlled study uses only the official ESC-10 subset of [ESC-50](https://github.com/karolpiczak/ESC-50), prepared by Karol J. Piczak, under CC BY 3.0. Per-recording original credits and license links are retained in `data/esc10-split.json` and the preparation protocol. The full ESC-50 collection has different terms; do not generalize the subset's license to it. Sources are genuine recordings; their generated mixtures are controlled mixtures, not naturally captured multitrack ground truth.

Natural examples come from [RealDESED](https://github.com/fschmid56/RealDESED), [Zenodo 20056072](https://zenodo.org/records/20056072). Only individually verified CC0 recordings are selected. Preserve the supplied per-file attribution, reviewed temporal labels and provenance with any redistributed example. They have no clean stems, so they support usability demonstrations rather than source-reference accuracy scores.

Other dependencies retain their upstream licenses. No model or dataset attribution establishes that STEMSCENE's learned refinement improves quality; that requires the executed comparisons.
