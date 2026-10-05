# Description diagnostic: implementation and adversarial review

**Result update, 2026-10-05:** all relevant benchmarks are completed and verified. See the [full results and interpretation](final_report.md) and [41-run JSON](recommended_results.json). Historical launch status below is retained as review history.

**Launch update, 2026-10-05:** GPU smoke `27611475` passed (61 tests, five variants, 1m36s). Full cache `27612822` and four dependent predictors are submitted. The review below records the pre-launch design; the [live checklist](experiment_checklist.md) supersedes its historical pending status.

Status on 2026-10-02: implemented and tested on synthetic inputs locally. No description cache, smoke, or benchmark job has run on Snellius. SSH currently refuses connections before authentication.

## Question and controlled comparisons

Do human-provided clip descriptions predict reactions, and do descriptions add information to visual appearance? The official metadata supplies `clip_description` for every clip. This is a text-assisted diagnostic; its results must be reported separately from video-only forecasting.

| Condition | Predictor input | Trainable parameters | Main comparison |
|---|---|---:|---|
| Existing B1 | All visual frames | 169,109 | Video-only reference |
| `description_only` | One frozen description vector | 169,109 | B0 and B1; unchanged B1 predictor architecture |
| `visual_description` | Visual frames plus the clip's description vector | 335,381 | Both unimodal models and both fusion controls |
| `description_visual_control` | Visual features supplied to both fusion branches | 335,381 | Added capacity without descriptions |
| `description_text_control` | Description vector supplied to both fusion branches | 335,381 | Added capacity without visual input |

Fusion applies separate LayerNorm→Linear projections to the two inputs, averages over valid frames, concatenates the two 128-dimensional pools, and applies the common head. The description vector is repeated across a clip's frames, so its mean is the same vector. The three fusion conditions have identical parameter shapes and initialization; only their input modalities change. Duplicate-input controls retain independent trainable branch projections. Description-only conditions pass no visual feature values to the predictor, although the existing visual inventory supplies the original frame-count diagnostics.

All conditions retain B1's official splits, seed 42, optimizer, batch size, training limit, KL objective, and validation-KL checkpoint selection. No label is used for text encoding. Test performance must not select the representation, chunking rule, or architecture.

## Frozen text representation

The experiment reuses the text tower in the already cached `google/siglip2-so400m-patch14-384` checkpoint, revision `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`. The [pinned model configuration](https://huggingface.co/google/siglip2-so400m-patch14-384/blob/e8e487298228002f3d8a82e0cd5c8ea9c567f57f/config.json) gives 1,152 output dimensions. The fixed-resolution checkpoint loads as `SiglipModel`; the code uses `AutoModel` and its public `get_text_features` interface.

The [pinned tokenizer configuration](https://huggingface.co/google/siglip2-so400m-patch14-384/blob/e8e487298228002f3d8a82e0cd5c8ea9c567f57f/tokenizer_config.json) specifies no BOS, an EOS token, right padding, lowercase text, and `input_ids` without a padding attention mask. The implementation explicitly lowercases text and requires the fast tokenizer, avoiding a new SentencePiece dependency. It follows the [versioned Transformers guidance](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/siglip2) to use 64 positions including padding.

Long descriptions are tokenized once, then split into nonoverlapping windows with room for EOS. Every content token is retained. Each window is encoded in frozen evaluation mode with bf16 and converted to float32. The clip vector is the content-token-weighted mean of window outputs, accumulated in float64. No L2 normalization or learned text fine-tuning is applied. There is no prompt template, movie-title field, genre field, or reaction-label text added by this pipeline.

This pooling choice is an experiment design decision. It retains all tokens but discards relationships across windows; it does not make the 64-token encoder a long-context model. Per-clip token/chunk counts and split-level multi-window counts are saved so that this limitation can be quantified after extraction.

## Cache and evaluation provenance

`src/experiments/descriptions.py` stores features on the cluster under `data/features/siglip2-descriptions/<specification hash>/`. Identity includes the pinned encoder settings, official split hashes, implementation checksum, and uv lock checksum. Index records contain sample IDs and exact source-description hashes, without copying the text into outputs. The cache reader verifies IDs, text hashes, split identity, file checksums, dimensions, dtype, and finite values.

Extraction uses an exclusive file lock and durable progress commits. Each commit records the completed row count, a checksum of those rows, and their token counts. Recovery rejects a corrupted committed prefix and recomputes any uncommitted suffix. Complete caches are verified and reused without loading the encoder. Neither text-only nor multimodal training changes the existing visual cache.

Each benchmark uses the existing registry, config/code/environment records, checkpoints, official metrics, per-class diagnostics, and saved predictions. The summary explicitly records its input mode and whether it uses descriptions. Frame-count strata retain the original visual counts even when the model sees a single text vector. Movie-overlap strata remain part of evaluation.

## Adversarial interpretation

- A description-only gain over B0 establishes predictive information in these supplied descriptions under the official split. It does not establish video understanding.
- A fusion gain over B1 alone may come from the larger predictor. Complementarity requires comparison with both equally sized single-modality controls as well as the smaller baselines.
- A strong description result is not proof of label leakage. Scene descriptions can legitimately summarize events absent from sampled frames. They can also identify recurring movies, characters, or situations. The audit found 95.94% of test clips come from movies represented in training; show seen/unseen movie diagnostics and avoid claims of unseen-movie generalization.
- Exact duplicate descriptions were not found across splits in the audit. That does not rule out semantic duplicates, shared plot information, or pretraining overlap.
- The chosen text and visual towers have different pretraining objectives and input information. Even with matched predictor capacity, their comparison does not measure an intrinsic upper bound for either modality.
- This is not a reproduction of the paper's LLaVA text experiment. Do not attribute differences from those published scores solely to the input modality.
- A negative result constrains this frozen text encoder and pooling rule. One seed, one representation, and a small unseen-movie subset limit conclusions. Use paired movie-cluster uncertainty for the predeclared comparisons.

## Verification and launch gate

All **61 local tests pass** in conda `torch`. The 16 new synthetic cases cover long-text token retention and weighting, EOS/padding handling, interrupted-cache recovery, corrupted-prefix and completed-file rejection, changed-description rejection, missing text, input-modality isolation, original frame-count strata, both fusion branches' gradients, padding invariance, capacity/initialization controls, and production fitting/checkpoint reload for all four predictors. The first test run caught a misnamed baseline config reference, which was corrected before the full suite passed.

The real tokenizer and text tower have not been exercised on cluster GPUs by this implementation. The separate `smoke_descriptions_3videos` job uses the same three official training clips as earlier smoke tests, eight frames each, and isolated visual/text caches. It runs B1 and all four new variants through the production fit, reload, prediction, and diagnostic functions. Its fit-on-training scores are not benchmark results.

Once SSH is restored, commit/push/sync the code, check live resources, and ensure the pinned fast tokenizer files are cached on the cluster. This may download tokenizer assets only; the full checkpoint is already cached:

```bash
cd /scratch-shared/gmago/video2reaction/code
source configs/clusters/snellius.env
uv run --frozen --no-sync python -c 'from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained("google/siglip2-so400m-patch14-384", revision="e8e487298228002f3d8a82e0cd5c8ea9c567f57f", use_fast=True); assert t.is_fast; print(type(t).__name__)'
release=$(python3 scripts/freeze_release.py)
bash "$release/scripts/submit_smoke.sh" smoke_descriptions_3videos
# Only after verifying successful smoke completion and its artifacts:
cd "$release"
python3 scripts/submit_descriptions.py
```

The batch has one cache job followed by four predictors with required `afterok` dependencies, bounded to four lanes. The cache has a 90-minute limit; predictors have 30-minute limits. At the previously verified MIG rate, the caps are 224 SBU for the batch and 16 SBU for the smoke. These are limits, not measured runtime estimates. Refresh access, queue, storage, and remaining budget before submission.
