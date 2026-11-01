# Actual-session demo

Open `index.html` from a local static server or view the MP4 and WAV assets directly.
The video is a captured-session walkthrough with original/edited audio excerpts,
not a live inference recording. Source audio is RealDESED 001738.wav (CC0). The
annotations and metadata credit is Florian Schmid et al., DOI 10.5281/zenodo.20056072,
CC BY 4.0. See `assets/CREDITS.txt` and `assets/demo-provenance.json`.

The task retains the immutable source session and a full aligned DOUBLE WAV
export. To re-render from that export or an independently recovered copy, supply
its folder and the session capture:

```sh
python render_demo.py --export /path/to/aligned-export \
  --screenshot assets/ui-session.jpg --output /path/to/new-render-folder
```

Requirements: Python 3.11/3.12, NumPy, SoundFile and FFmpeg with libx264, AAC,
drawtext and fontconfig. The project's installed dependencies supply NumPy and
SoundFile. The default installed typeface is Arial; use `--font` for another
installed family. Fonts are not bundled. Source paths are arguments; no private
cache or model service is used by the video renderer.

The script refuses an existing output folder, verifies export file hashes,
independently recombines WAVs/envelopes, preserves gain and writes an H.264/PCM
master plus an H.264/AAC MP4. Edit composition text and timing in the retained
script; retain the previous successful master until a replacement passes checks.
The original and edited WAV excerpts provide the comparison without lossy AAC.

The media report separates technical frame/decode/timing checks from unavailable
subjective audiovisual judgment. This natural scene has no clean multitrack
references, so it does not provide quantitative source-separation accuracy.

The prepared local export artifact is `stemscene-demo-export.zip` (16,859,044 bytes),
SHA-256 `dc2d08cca26ea43bf18ff3b671849639086533ee13108f8edca77238cb5d25be`.
It contains only the export manifest and its listed files. Publication remains
subject to owner approval. It is deliberately outside the Git source tree.
With the archive available, run from the repository root:

```sh
python examples/recombine.py /path/to/stemscene-demo-export.zip \
  --report /path/to/new-replay-report.json
```

This independent replay needs NumPy and SoundFile, with no model or engine
imports. For video rendering, extract the ZIP into a fresh directory and pass
that directory as `--export`. The archive includes the byte-preserved original
recording, three DOUBLE WAV estimates, residual, envelopes and edited mix.
Audio licensing is CC0; RealDESED annotations and metadata are CC BY 4.0.

`independent-replay.json` records the direct ZIP replay, and
`media-verification.json` records the final master and MP4 decode/timing checks.
The master remains in task-local evidence and can be regenerated with the script.
Run `python verify_demo.py /path/to/new-render-folder --output /path/to/new-qc`
for the same technical checks. Verification also requires SciPy and ffprobe.
