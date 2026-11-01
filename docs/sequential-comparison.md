# Sequential residual baseline

This comparison runs the same local AudioSep checkpoint on each requested layer in two orders. Each later query receives the actual residual from earlier extraction. It measures query-order dependence of sequential extraction; it does not claim that a different order identifies the correct physical sources.

For queries indexed `0, 1, 2`, the forward order is `0, 1, 2` and the reverse order is `2, 1, 0`. Both stored arrays retain the original query-axis order, so query index zero always refers to the same requested layer in scoring.

## Matched audio and computation

The utility reads the original 44.1 kHz mixed WAV and request JSON from each recipe's `inference` folder. It verifies their provenance against the primary independent computation and confirms that resampling the original mixture exactly reproduces the record's stored mixture array. It never reads that recipe's `evaluator` folder. No clean reference or presence label enters the AudioSep API.

For each order:

1. Copy the unchanged original decoded mixture into a residual.
2. Run `AudioSepBackend.separate(residual, original_rate, [query])` once per query.
3. Subtract that returned, aligned proposal from the residual at the original recording rate.
4. Convert each stored proposal to the existing 32 kHz record rate using the same compensated polyphase FIR helper as the independent generator.

Each order uses three separator calls for three requested layers, the same number of model extractions as the independent baseline. The checkpoint hash, chunk/overlap/timeout/thread settings and source bytes must match the independent record. The executed configuration uses MPS for the separator, CPU for the text encoder, four CPU threads and the already tested optional MPS operation fallback. Forward/reverse timing includes real query encoding when not cached, model inference, residual updates and final alignment conversion. Reused embeddings can affect warm timings; timing differences between orders must not be interpreted solely as separation cost.

The primary independent proposals are copied unchanged, not regenerated. FUSS raw outputs and their provenance are also copied unchanged. The new results are a matched comparison on existing controlled mixtures, not a changed-input demonstration or a natural multitrack accuracy claim.

## Integrity and output

The source proposal archive's SHA256, frozen recipe identity and checkpoint must verify before inference. NPZ decoding prohibits pickle and bounds compressed size, expanded size and member count. Output goes to a fresh root outside the source records and original mixture directories. The utility refuses to overwrite an existing result.

The new archive adds `sequential_forward` and `sequential_reverse` as float32 arrays shaped `(queries, samples)`. Its sidecar records a new `record_sha256`, the immediate `parent_record_sha256`, the parent's metadata hash and a `record_ancestry` chain preserving the earlier independent/FUSS parent hashes. Original metadata, clip licensing provenance and FUSS details remain available. The `sequential` block includes individual call timings, source/record rates, settings, device/fallback configuration and cumulative process peak RSS. Peak RSS is not an isolated per-record allocation or total system memory.

Selection is the default split. Confirmation requires a final-model artifact and a decision record matching both model and recipe hashes before confirmation records are listed or opened. Preparation does no model training or model selection. The evaluator computes separation/edit metrics and explicitly measures RMS differences between the two query orders.

## Local development command

From the repository root:

```sh
STEMSCENE_STUDY=/absolute/path/to/study
PYTORCH_ENABLE_MPS_FALLBACK=1 \
.venv/bin/python \
  scripts/prepare_sequential_proposals.py \
  "$STEMSCENE_STUDY/proposals-fuss" \
  "$STEMSCENE_STUDY/mixtures" \
  "$STEMSCENE_STUDY/proposals-comparison" \
  --frozen-recipes data/esc10-split.json \
  --expected-recipes-sha256 bd19a599b3165a0648e1ed9fef17e085e18fd4878f090e25e24945454c3d0728 \
  --checkpoint "$STEMSCENE_STUDY/models/audiosep_base_4M_steps.ckpt" \
  --tokenizer "$STEMSCENE_STUDY/models/roberta" \
  --device mps --threads 4
```

This command computes only the selection split. On the managed Mac it needs the same scoped GPU execution permission as the tested AudioSep MPS path. The emitted `sequential-augmentation.json` is the execution evidence; a listed command alone is not a successful computation.

## Executed selection preparation, 2026-10-29

All **10 frozen selection recipes** completed both orders, making **60 actual separator calls**. Every record retains the independent estimates and raw FUSS outputs and adds two `(3, 160000)` arrays at 32 kHz. This selection preparation opened no confirmation recordings or proposal archives.

An independent read-back check verified full selection coverage, exact dtype/value preservation of all seven parent arrays in all ten records, original 44.1 kHz mixture SHA256s, the complete parent hash chain, new NPZ and sequential-array hashes, finite outputs, and forward/reverse query index sequences. The development readback report, `sequential-selection-verification.json`, and successful preparation log were retained. The public [confirmation summary](../evaluation/v1/summary.md#sequential-execution-verified) records the subsequent full frozen run and its independent readback. The combined root retains its own `sequential-augmentation.json`.

Observed backend load was **21.18 seconds** and cumulative maximum process RSS was **1065222144 bytes**. The expected `aten::col2im` MPS fallback to CPU was observed. First forward orders included uncached query encoding and cannot fairly be compared with warm reverse-order latency without that qualification.

Forward/reverse waveform RMS differences ranged from **0.00840 to 0.06893**, with mean **0.03396**, in the decoded recording's amplitude units. These measurements establish order-dependent estimates on these mixtures; they do not establish which estimate is more accurate or how listeners would judge it. Separation and edit quality must be evaluated against the source references separately.

The initial launch failed before creating output because this isolated environment did not have an editable package installation. The utility now resolves its repository source package explicitly. That failure log was retained beside the successful run evidence; no partial baseline was substituted for inference.
