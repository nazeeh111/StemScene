# Independent installation evidence

This compact bundle records an actual independent development-wheel CPU run and
a separate final beta package audit. It retains numerical results and hashes
while replacing task-local absolute paths with a portable description. It does
not package model weights, an environment, cached predictions or natural audio.
See [verification](../../docs/verification.md) for the interpretation and limits.

| File | Evidence |
| --- | --- |
| [development-cpu-run.json](development-cpu-run.json) | Executed `0.1.0.dev0` wheel, fresh public input/models, three new CPU queries, process timing/RSS, persistent edit and replay checks |
| [installed-requirements.txt](installed-requirements.txt) | Exact 30-distribution environment used by that historical development run |
| [model-download-manifest.json](model-download-manifest.json) | Byte-exact public downloader manifest: source URLs, sizes, pinned/measured hashes and transfer timings |
| [recombination-report.json](recombination-report.json) | Byte-exact standalone replay result; unity error 1.39e-17 and edited-mix/envelope/source-decode errors 0 |
| [final-beta-audit.json](final-beta-audit.json) | Passed `0.1.0b1` source/package/install audit, exact core/model parity and installed offline edit/export/replay |
| [beta-correctness.log](beta-correctness.log) | Exact executed result: all 59 current correctness tests passed |
| [beta-recombination-report.json](beta-recombination-report.json) | Exact standalone replay result for the beta's new export of the saved real project |
| [provenance.json](provenance.json) | Original task report hash, explicit portability transformation and hashes of exact copied records |

The development wheel SHA is
`0d6df52c016e21e99b6e9992b832fd0db0b6dd16effd7706e0ba566b25e623dc`.
Its original source archive SHA is
`d37df5733e8f189fdb4aa4351afb7cd1129b44a80994653bfb4bc2d1ca0dac95`.
These identify retained historical artifacts, not final beta downloads. The old
source archive was built from a copy that excluded the private task plan, and
still contained earlier task-specific paths in two comparison documents. Those
historical archives are not included in this portable evidence directory.

The separately installed and audited beta wheel SHA is
`281523ec1c6b6da10064c760e8e84aef561711147f77047b5ee5199aba892a1b`.
All 25 installed package files matched the frozen beta source. All 13 declared
core/model hashes matched the actual development CPU run; only the package
version, browser concurrency handling and optional FUSS startup changed. The
beta's offline smoke reused the saved real project, rather than claiming a new
inference run. Final source-archive checksums are retained outside the archive
to avoid self-reference; final contents are checked after this report is packed.

The run used the public CC0 RealDESED `audio/000809.wav` member of the
[official archive](https://zenodo.org/api/records/20056072/files/test.zip/content).
Fresh public metadata identified typing, footsteps and a door. Publisher event
timing selected a gain edit after inference; no presence labels, clean references
or anchors entered the engine. The 15.701-second natural mixture has no clean
multitrack truth and does not establish sound-isolation quality. The 3.20 GiB
process RSS and 328.278-second latency describe this observed CPU process,
including loading, rather than a hardware minimum or a speed guarantee.

## Portable record of the executed workflow

This is a path-normalized description of the executed development-wheel
sequence, not a claim that the command listing itself performed another run.
`WORK` denotes a fresh directory outside the source checkout, `SOURCE` an
immutable source snapshot, and `PYTHON` its new virtual environment interpreter.
The actual source/recording/model/artifact hashes are in the JSON records.

```sh
uv build "$SOURCE" --out-dir "$WORK/dist" --python 3.12
uv venv "$WORK/venv" --python 3.12
PYTHON="$WORK/venv/bin/python"
uv pip install --python "$PYTHON" \
  "$WORK/dist/stemscene-0.1.0.dev0-py3-none-any.whl"
"$WORK/venv/bin/stemscene" --version
"$WORK/venv/bin/stemscene" models download "$WORK/models" --timeout 120
"$WORK/venv/bin/stemscene" init "$WORK/input/000809.wav" "$WORK/project"
"$WORK/venv/bin/stemscene" extract "$WORK/project" \
  --checkpoint "$WORK/models/audiosep_base_4M_steps.ckpt" \
  --tokenizer-dir "$WORK/models/roberta" --device cpu --threads 4 \
  --query "keyboard typing" --query "footsteps" \
  --query "a door opening and closing"
```

The installed Python API then loaded that project, edited the door estimate at
5.06–5.87 seconds with gain 0.2 and 10 ms fades, checked replacement/removal/undo,
and exported DOUBLE WAV with explicit floating-peak preservation. The standalone
`examples/recombine.py` obtained from the built source archive replayed both the
folder and retained ZIP. An actual module check confirmed no STEMSCENE, PyTorch
or Transformers imports in that standalone replay. Installed CLI replay and
both standalone outputs matched the decoded exported mix exactly.

Offline model controls, `PYTHONNOUSERSITE=1` and an absent `PYTHONPATH` isolated
execution from user package/model caches. Fresh dependency/model/input directories
were used; public weights/tokenizer files were downloaded by the installed
explicit downloader, not copied from an earlier trial. No credentials were read
or model API invoked.

Remote CI, a public clean clone and versioned release-download verification
remain pending. A final package audit is not a second heavy inference run and
must not silently replace the development artifact identity above.
