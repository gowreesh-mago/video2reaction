# Tier 0 results

Verified on Snellius at **2026-09-30 06:37:41 UTC**. All five benchmark runs and the corrected feature-cache job completed with exit `0:0`. No jobs remain queued. The three-video smoke tests are separate infrastructure checks.

## Main result

Frozen visual mean pooling improves test KL by **21.0%** over the constant training-label prior. It also has the lowest validation KL, which was the declared checkpoint selection metric. The temporal model and composite loss produce metric tradeoffs, without a clear overall improvement over mean pooling.

| Model | Selected epoch / epochs run | Validation KL ↓ | Test KL ↓ | Cosine ↑ | MRR ↑ | F1@1 ↑ | F1@3 ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| B0: training prior | — | 0.692247 | 0.689281 | 0.751309 | 0.599569 | 0.237624 | 0.558669 |
| B1: visual mean pooling | 9 / 17 | **0.541493** | **0.544414** | 0.825196 | 0.723182 | 0.529069 | 0.619689 |
| B2: temporal transformer | 4 / 12 | 0.545500 | 0.546904 | 0.825163 | **0.729166** | **0.541735** | 0.607179 |
| B2 set control: no positions | 5 / 13 | 0.547861 | 0.546476 | 0.822704 | 0.728609 | 0.535816 | 0.611040 |
| A5: KL + cosine + ranking | 9 / 17 | 0.542901 | 0.544462 | **0.825493** | 0.725885 | 0.535416 | **0.621179** |

These are results on all **2,070 test clips**, after selection on **1,035 validation clips**. Training used seed 42 and the fixed configurations described in the [review](code_and_hypothesis_review.md). The learned runs use source `bbf8c1d`; B0 uses `e91a469`. B1/A5 have 169,109 trainable parameters, versus 565,781 for B2/set.

## What the results say about the hypotheses

- **Visual features beyond the prior: supported for this setup.** B1 reduces KL by 0.144867 and raises MRR by 0.123613 and F1@3 by 0.061020. This establishes predictive value on the official splits. It does not establish generalization to new movies or causal emotion understanding.
- **Temporal positions beat a matched set model: no improvement on the primary test metric.** B2's KL is 0.000428 higher than the matched set control. Its MRR and F1@1 are higher, while F1@3 is lower. This result does not support a general superiority claim for this temporal configuration.
- **The temporal model uses order, but that does not make it superior.** Shuffling its saved test frame order raises KL from 0.546904 to 0.548840; the largest prediction change is 0.128323. B1, A5, and the set control are unchanged within 2.4e-7 probability. Order sensitivity is distinct from beating an order-independent model.
- **The composite loss has small favorable ranking differences.** A5 versus B1 changes MRR by +0.002704, F1@3 by +0.001490, cosine by +0.000297, and KL by +0.000048. These are small point-estimate gains at one coefficient setting and one training seed. They do not establish loss superiority or identify which added term helps.

This batch evaluates the recorded backbone, heads, objectives, coefficients, and training budget. It does **not** prove or disprove the broader temporal, peak, VAD, emotion, or text search space. Those later experiments have not been launched.

## Paired uncertainty checks

We resampled **1,183 test movies** with replacement 10,000 times, keeping all clips from each sampled movie together and pairing the same samples across models. The statistic remains the benchmark's clip-weighted mean KL. Bootstrap RNG seed: 20260930.

| Comparison: first minus second | Test KL difference | 95% percentile interval |
|---|---:|---:|
| B1 minus B0 | -0.144867 | [-0.157545, -0.131936] |
| B2 minus B1 | +0.002490 | [-0.003218, +0.008202] |
| B2 minus set control | +0.000428 | [-0.004238, +0.005015] |
| Set control minus B1 | +0.002062 | [-0.003571, +0.007822] |
| A5 minus B1 | +0.000048 | [-0.001401, +0.001438] |
| B2 shuffled minus ordered | +0.001936 | [+0.000493, +0.003416] |

Negative values favor the first model. The B1-versus-prior improvement is supported by these intervals. All comparisons between separately trained learned models include zero; they do not establish KL superiority or equivalence. The shuffle comparison supports a small order effect within the trained B2 model.

These intervals measure test-sample uncertainty conditional on the trained models. They do not capture training-seed variation, correct for multiple comparisons, or cover MRR/F1/cosine. The shuffle analysis uses one saved permutation per clip. Small ranking improvements remain exploratory.

## Benchmark limits and diagnostics

Only **84 test clips** are from movies absent from training; **1,986** share a movie with training. On the 84 unseen-movie clips, KL is 0.742025 / 0.600715 / 0.594232 / 0.580218 / 0.597690 for B0/B1/B2/set/A5 respectively. This small subset is exploratory, and is insufficient to rank methods reliably for new-movie generalization.

Every run saved per-class precision/recall/F1 and support, plus diagnostics by target entropy, frame count, dominant target probability, and movie overlap. Their JSON files are available under each run's `test/` directory. Aggregate results are in [tier0_results.json](tier0_results.json). A detailed rare-class analysis and the requested 20 qualitative examples remain pending.

## Completion and reproduction

The shared feature cache contains all **455,226 frames**: 317,950 train, 45,964 validation, and 91,312 test. Each split has completion metadata and feature, image-hash, and index checksums. Cache generation took 1h50m35s on one A100; the four small predictors took 2m42s–2m43s each on MIG slices, using cached features.

Job IDs: B0 `27382108`, cache `27382293`, B1 `27382294`, B2 `27382295`, set `27382296`, A5 `27382297`. Earlier failed attempts are preserved in the [checklist](experiment_checklist.md).

Outputs, checkpoints, predictions, configs, provenance, logs, and the registry have been synchronized to the Mac. Frames, raw split files, and feature tensors remain on Snellius.

Verification passed for all five models: validation/test IDs, movie IDs, class order, targets, and frame counts match across runs; all metrics recompute from saved predictions within absolute tolerance 1e-10. The four learned runs have `best.pt` and `last.pt`, exact checkpoint-reload flags, and selected validation KL matching the saved training history. Per-class and stratified diagnostic files are present.

Recompute the aggregate analysis in the cluster's locked uv environment:

```bash
cd /scratch-shared/gmago/video2reaction/code
source configs/clusters/snellius.env
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --frozen --no-sync python scripts/analyze_tier0.py \
  --outputs "$V2R_OUTPUT_DIR/experiments" \
  --output "$V2R_ROOT/results/collected/tier0_results.json"
```

Use NumPy **2.2.6**, as recorded by the experiments. The Mac's conda environment has NumPy 1.26.4, whose default tied-label ordering produces different top-k F1 values. The analysis script rejects that mismatch; the original cluster metrics are retained.

No new experiments were launched during this status check. The next batch remains a user decision; additional configurations selected after these test results should be labeled exploratory.
