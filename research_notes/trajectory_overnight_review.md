# Overnight expansion: October 6–7, 2026

The user authorized a larger queued batch while the laptop is off. This adds **72 conditions × 3 paired seeds = 216 runs**, with eight concurrent predictors, using the completed DINOv2 cache and the existing visual-only model implementation. The original 48 runs remain unchanged.

## Matrix and hypothesis review

| Additional conditions | Count | Question |
|---|---:|---|
| VAD bandwidth initialization 0.1/0.6/1.2 across seven pooling rules | 21 | Are conclusions sensitive to the optimization starting point? Bandwidth remains learned; this does not test a fixed bandwidth. |
| Relevance temperatures 0.05/0.1/0.5/1.0 under soft, sparse and proximity pooling | 12 | Does selection strength affect accuracy and temporal support? The scorer can partially compensate its scale. |
| Timestamp-free controls | 8 | Does position help under each matched pooling rule? Duration and contextual attention remain present. |
| Hidden width 256 for VAD/free soft/sparse, VAD duration and mean pooling | 6 | Does extra visual capacity help? Mean pooling remains an encoder reference, not a Transformer capacity control. |
| Rarity exponents 0.25/1.0 with/without importance correction under duration/sparse pooling | 8 | Do rare-reaction improvements reflect resampling or a changed objective? |
| Free soft/sparse decoders with rare sampling and importance controls | 4 | Does rarity weighting require a VAD decoder? |
| Free local decoders under duration/power pooling | 2 | Is the fixed VAD bottleneck limiting those mixtures? |
| Two additional semantic permutations under duration/soft/sparse pooling | 6 | Does meaningful geometry outperform several arbitrary label assignments? |
| One-layer context under VAD duration/soft/sparse and free soft/sparse pooling | 5 | Does contextual depth help generalization? |

All conditions preserve official splits, the frozen visual encoder, learning rate 0.0003, 50-epoch maximum, patience 8, validation-KL checkpoint selection, and seeds 42/43/44. Sampling variants alter only training examples and the declared importance correction. No text features, descriptions, comments, or language teachers enter the model.

## Adversarial review

- This is an exploratory expansion, not 216 independent confirmations. Compare matched conditions and aggregate paired seeds; select settings using validation scores. Report the whole matrix, not just the best test score. Earlier primary-run scores have already been observed, so this is not an untouched final test of a newly selected method.
- Temperature changes test optimization sensitivity, not a richer fixed VAD geometry. A failure of all variants would constrain this representation and supervision, not every possible trajectory model.
- Sparse attention can use full-video context. Low attention weight is not proof that a scene was ignored internally. Clip labels do not establish which moments caused real comments.
- Original scene timestamps still give a piecewise-constant estimate. This batch does not add dense observations, independent highlight annotations, fixed-window smoothing, or label priors/residual classifiers.
- Model forward and training code are unchanged. The only new model combinations are already-supported free duration/power decoders. Synthetic checks cover each of the 72 configurations, and all four three-video smoke shards must pass exact-source checks before the launcher submits full runs.
- Each full run has a 20-minute limit, versus observed 1–5-minute primary runs. Timeouts are failures to investigate, not results. Eight concurrent jobs cap filesystem and GPU demand. The requested predictor allocation totals 72 MIG GPU-hours (about 4,608 SBU at the observed billing rate); cluster reservation accounting may be more conservative. The last resource check showed over 81,000 SBU available for submission.

## Autonomous execution

Four GPU smoke shards each cover 18 conditions on the same three videos and 24 keyframes, with 20 fitting updates, reload/resume checks and contribution reconstruction. A staging launcher depends on successful completion of all four and verifies their source identity and combined coverage before submitting 216 predictors. Each submission is recorded atomically and deduplicated. No local process is needed after submission.

A separate staging job is scheduled no earlier than **October 7, 06:00 Amsterdam (CEST)**. It writes scheduler states and saved scores to `results/overnight_status_20261007_0600.json` and `.md` on Snellius, even if a smoke or launcher fails. This is an automated snapshot, not an independent metric recomputation or an automatic chat message. Actual start can be delayed by the scheduler.
