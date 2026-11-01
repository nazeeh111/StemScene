# AudioSep STFT compatibility adaptation

The separator retains the convolution-based analysis and synthesis used by the released AudioSep weights. It does not replace them with a different FFT backend. The internal module `stemscene._vendor.audiosep.stft` is adapted from Torchlibrosa 0.1.0 by Qiuqiang Kong, under its MIT license. The complete notice is retained beside that module as `TORCHLIBROSA_LICENSE`.

Original sources: [STFT/ISTFT](https://github.com/qiuqiangkong/torchlibrosa/blob/master/torchlibrosa/stft.py), [license](https://github.com/qiuqiangkong/torchlibrosa/blob/master/LICENSE.md). The version actually adapted was the installed 0.1.0 distribution, whose `stft.py` SHA256 is `c5dd1b1043ea74725c79b5e89efaaef760558b5f0abbf311efc075eb9ed653c6`. Current upstream master can differ.

Changes replace window construction and padding helpers with SciPy and NumPy equivalents, removing librosa's import-triggered Numba compilation from this initialization path. Convolution module names, tensor shapes, DFT matrix construction, frozen state keys, overlap-add, denominator clamping, phase handling and trimming remain unchanged. The AudioSep configuration uses a 2048-sample periodic Hann window, hop 320, centered reflection padding and full-length windows. Other configurations and ONNX export are not supported product claims.

Verification executed on the local Mac on 2026-10-29:

| Check, actual 2048/320 configuration | Maximum absolute difference |
|---|---:|
| Initial state tensors versus original source algorithms | 0.0 |
| Analysis output versus original source algorithms | 0.0 |
| Synthesis output versus original source algorithms | 0.0 |
| Analysis versus independent `torch.stft` | 1.52587890625e-5 |
| Reconstructed waveform versus `torch.istft` | 3.5762786865234375e-7 |
| Random waveform reconstructed with strict-loaded released checkpoint state | 3.5762786865234375e-7 |

The independent FFT differences are ordinary float32 differences between convolution and FFT calculation; the adaptation's outputs matched the original convolution algorithms exactly on these inputs. The test also checked a 64/10 configuration, exact state-key agreement and that neither librosa nor Numba was imported. It extracted original class/helper definitions directly from the installed source with import-time cache/JIT decorators omitted, allowing a comparison of the original numerical algorithms without repeating the initialization bottleneck. This is not a claim that the full original dependency stack completed initialization.

The deterministic actual-size test used 8329 random samples, seed 101, amplitude scale 0.1. Initialization of the adapted STFT and ISTFT pair took 20.88 seconds in that run, excluding cold PyTorch import. Other tasks were active, so this is an observed shared-machine timing. Subsequent actual CPU/MPS checkpoint execution, frozen comparisons and independent installed-wheel inference are recorded in [verification](verification.md) and [comparisons](comparisons.md). Analysis/synthesis equivalence alone does not establish source-separation quality.

The development helper `check-stft-equivalence.py`, `stft-equivalence.json` and its log were retained with the original local experiment. These historical experiment files are not installation prerequisites. The public [executed source capture](../evaluation/v1/source-capture/provenance.json), [comparison](comparisons.md) and [independent installation record](verification.md) identify the runtime and subsequent full-checkpoint checks.
