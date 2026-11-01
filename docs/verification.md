# Installation and export verification

An independent, non-editable `0.1.0.dev0` wheel installation ran actual CPU
inference on a freshly downloaded natural recording, edited one estimated event
and exported an independently replayable bundle. The final `0.1.0b1` beta is
audited separately because its source includes later version, browser concurrency
and optional FUSS startup changes. Neither a successful installation nor exact
reconstruction establishes separation quality. Read the [model card](model-card.md)
and [frozen comparison failures](comparisons.md) alongside these checks.

## Actual independent CPU run

| Check | Observed result |
| --- | --- |
| Environment | Fresh macOS arm64 Python 3.12.13 virtual environment and dependency cache; built wheel, no editable source import |
| Dependencies | All 30 installed distributions matched the locked versions |
| Model setup | Installed public downloader fetched all seven pinned model/tokenizer/license assets into a fresh directory; 40.915 seconds |
| Changed recording | CC0 RealDESED `000809.wav`: 15.701333 seconds, 753664 frames, 48 kHz mono |
| Requests | `keyboard typing`, `footsteps`, `a door opening and closing` |
| CPU process | 328.278 seconds including model load; 3432841216 bytes peak RSS, about 3.20 GiB; four configured threads |
| Selected edit | Door estimate at 5.06–5.87 seconds, gain 0.2, 10 ms fades inside the span |
| Outside-span preservation | Maximum decoded-sample error 0 |
| Saved event operations | Replacement stayed one event; removal restored the decoded original; undo restored the edited mix exactly |
| Persistent estimates | All stem arrays unchanged by editing; typing layer lock retained |
| Independent replay | Unity error 1.3877787807814457e-17; edited mix, envelope recipe and original-decode errors all 0 |
| Model independence | Standalone replay verified that `stemscene`, `torch` and `transformers` were absent from imported modules |

The source bytes, sample rate, frame count and channel count were preserved. The
standalone folder replay, standalone ZIP replay and installed CLI replay each
matched the decoded exported mix exactly. The replay utility came from the built
source archive and uses only NumPy, soundfile and the Python standard library.

The recording and its license/event metadata were obtained anew from the
[official public test archive](https://zenodo.org/api/records/20056072/files/test.zip/content)
using bounded byte ranges. ZIP member length, CRC and recording SHA-256 were
verified; the full archive was not downloaded or checked against its full-file
MD5. The publisher's door interval selected an edit after extraction. No clean
reference, presence label or anchor entered the inference API.

Extraction used explicit newly downloaded checkpoint/tokenizer files and the
installed bundled refiner. The new environment disabled user packages, removed
`PYTHONPATH`, and enabled offline Hugging Face/Transformers controls. No old
model directory, saved estimate or proposal cache supplied the result.

The natural recording has no clean multitrack ground truth. Its classes are
outside the ESC-10 controlled refiner study. This run demonstrates a usable
computation and export path, not a verified clean typing/footstep/door separation.
There was no listening panel or new fourth-query expansion in this particular
run. The separate [browser demo](demo/README.md) and [long-audio test](long-audio.md)
cover additional interactions and retain their own failures and limits.

## Exact version and artifact boundaries

The [portable installation bundle](../evaluation/install-v1/README.md) retains
the executed development report, frozen package versions, public downloader
manifest, core-source hashes and replay checks. Its artifact identities are:

| Artifact | SHA-256 |
| --- | --- |
| Tested development wheel | `0d6df52c016e21e99b6e9992b832fd0db0b6dd16effd7706e0ba566b25e623dc` |
| Audited beta wheel | `281523ec1c6b6da10064c760e8e84aef561711147f77047b5ee5199aba892a1b` |
| AudioSep checkpoint | `f8cda01bfd0ebd141eef45d41db7a3ada23a56568465840d3cff04b8010ce82c` |
| Installed V1 refiner | `ddbc872006da68440ad3f72435c3b89ecacf422f2f7265d68581f1dee879a9a9` |
| Changed recording | `4163c9933885cc72d71695d280ee56cb87b8cbf7965ed2932a32bc362dee8e53` |

The [final beta audit](../evaluation/install-v1/final-beta-audit.json) passed
**59 correctness tests**, a non-editable installed-wheel update, bundled-artifact
and license checks, and installed CLI/API render/export/replay. All 25 installed
package files matched the frozen beta source bytes. All 13 explicitly checked
backend, engine, project/edit, export, refiner, vendored processing and model
artifact files matched the executed development CPU run exactly. The only changed
package files were the version declaration, browser session concurrency handling
and optional FUSS subprocess startup. The current server behavior is covered by
the beta suite; optional real FUSS output parity is recorded in the
[FUSS comparison](fuss-comparison.md).

The beta smoke loaded the real saved development project and preserved its stem
arrays and lock through event replacement, removal, undo and a new export. CLI
render/replay and standalone replay matched that exported mix exactly. No new
heavy inference was performed or claimed for the later wheel. Exact final archive
checksums are recorded outside the source archive to avoid self-reference; the
source archive is repacked after the portable audit record and checked again.

The initial development wheel omitted root third-party notices and Apache/weight
license evidence. Explicit `license-files` metadata corrected that omission;
the rebuilt development wheel and source archive contained all seven checked
artifact/license files byte-for-byte. The beta passed those checks again, plus
private-path scanning, exact inclusion of the three licensed demo WAVs and
exclusion of the private task plan `BUILD_BRIEF.md` despite its presence in the
build input.

## Scope of the checks

Only macOS arm64 Python 3.12.13 CPU was independently installed and inferred in
the run above. Python 3.11 is admitted by package metadata; Linux, Windows and
other hardware do not inherit a verified execution or speed claim. The observed
peak RSS is not a minimum RAM requirement. Cold loading, OS page cache and shared
machine load affect timing. Optional hardware/process probes were denied by the
sandbox and stopped; the exact child's timing and RSS came from `os.wait4`.

This is a built-wheel installation path. The [quickstart](quickstart.md) also
documents locked checkout installation and an official Linux CPU-wheel route;
those command listings are not proof of a newly executed public clean clone.
Remote CI, public clean-clone execution, versioned release downloads and published
links remain unverified at this report snapshot. No public release success is
claimed by these local records.
