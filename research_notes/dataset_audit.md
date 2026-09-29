# Dataset audit

Inspected the actual Snellius keyframes and official split metadata on 2026-09-30. Full numeric evidence is in `dataset_audit.json`. No Video2Reaction samples, frames, or metadata files were copied to the Mac; only aggregate audit output was collected.

## Files and identity

- Original keyframes: `/gpfs/home3/gmago/video2reaction/scratch4/workspace/sidongzhang_umass_edu-v2r/v2r_data/youtube_video/key_frames`.
- Organized access: `/scratch-shared/gmago/video2reaction/data/key_frames` (symlink to the original).
- Official `train.json`, `val.json`, `test.json`: `/scratch-shared/gmago/video2reaction/data/metadata`.
- Dataset repository: `infofusionlab/Video2Reaction`, revision `578d1423f89f1a7b52471b01e78770e08f0a226c`. The old `video2reac` URL redirects here. Split hashes are recorded in the JSON audit.
- 21 classes; exact CAD order: sadness, disgust, grief, fear, disapproval, disappointment, embarrassment, nervousness, annoyance, anger, confusion, realization, caring, curiosity, relief, approval, surprise, excitement, amusement, admiration, joy.

## Completeness and temporal data

All 10,348 official clips have an index and every referenced image: **455,226 JPEGs; zero missing files**. No index has non-increasing start-frame numbers. The parent frame directory also contains 1,390 extra clips not in the official splits; these are excluded.

Overall keyframes: minimum **15**, maximum **176**, mean **43.9917**, median **39**. Percentiles 1/5/25/50/75/95/99: **15 / 15 / 25 / 39 / 58 / 89 / 115**. The observed minimum is 15, whereas the paper reports 16.

Temporal order is retained in `index.csv`: scene_number, start_time, end_time, start_frame, end_frame. One-based `001.jpg` etc. correspond to zero-based scene numbers. JPEG headers of 207 systematically sampled first frames: 197 were 640x360, 9 were 534x360, 1 was 536x360. This is a size sample, not an exhaustive image decode/integrity scan.

## Targets, text and movie metadata

All targets have nonnegative finite values and sum to one within floating-point rounding; no unknown labels. Distributions are sparse maps under `reaction_outcome.reaction_distribution`. All clips have description text, IMDb ID and movie name. Genre is missing for some clips; movie IDs permit leakage diagnostics. The release provides one aggregated target distribution per clip, not the underlying comments or longitudinal distributions in these split files.

| Split | Clips | Mean frames | Median frames | Mean entropy (nats) | Dominant ties | Missing genre |
|---|---:|---:|---:|---:|---:|---:|
| train | 7243 | 43.898 | 39 | 1.4592 | 562 | 1594 |
| val | 1035 | 44.410 | 40 | 1.4528 | 81 | 231 |
| test | 2070 | 44.112 | 40.0 | 1.4595 | 159 | 452 |

Dominant-label ties matter: metrics must retain the reference tie-breaking convention. Metadata dominant labels are always among the tied maxima but may differ from the class-order argmax.

### Label imbalance

| Label | Train mean probability | Train dominant count | Val dominant count | Test dominant count |
|---|---:|---:|---:|---:|
| sadness | 0.023994 | 98 | 14 | 28 |
| disgust | 0.032466 | 74 | 11 | 21 |
| grief | 0.007164 | 19 | 3 | 6 |
| fear | 0.047321 | 344 | 49 | 98 |
| disapproval | 0.213118 | 1927 | 275 | 551 |
| disappointment | 0.041038 | 57 | 8 | 16 |
| embarrassment | 0.001562 | 0 | 0 | 0 |
| nervousness | 0.002776 | 2 | 0 | 0 |
| annoyance | 0.001724 | 0 | 0 | 0 |
| anger | 0.004534 | 3 | 0 | 4 |
| confusion | 0.048446 | 72 | 10 | 21 |
| realization | 0.003190 | 1 | 1 | 0 |
| caring | 0.004213 | 1 | 0 | 0 |
| curiosity | 0.010385 | 6 | 0 | 0 |
| relief | 0.003889 | 1 | 0 | 0 |
| approval | 0.026678 | 17 | 2 | 5 |
| surprise | 0.031769 | 35 | 5 | 10 |
| excitement | 0.015399 | 4 | 2 | 1 |
| amusement | 0.200744 | 1743 | 249 | 498 |
| admiration | 0.272068 | 2825 | 404 | 807 |
| joy | 0.007521 | 14 | 2 | 4 |

Train mean-probability max/min ratio is **174.17**. Admiration, disapproval and amusement carry about 68.59% of train target mass. Dominant-count imbalance is more severe, with zero train-dominant examples for annoyance and embarrassment. These definitions differ from the paper’s reported imbalance factor; do not equate them.

## Split overlap

No duplicate video IDs or exact description strings across splits. Content-level near-duplicate video detection was not performed. Shared movie identities:

| Pair | Shared IMDb IDs | Clips in first split from shared movies | Clips in second split from shared movies |
|---|---:|---:|---:|
| train_val | 722 | 3506 | 989 |
| train_test | 1110 | 5197 | 1986 |
| val_test | 568 | 772 | 1038 |

**1,986/2,070 test clips (95.94%) have a movie in training.** Preserve official splits for benchmark comparability; add same-movie-excluded retrieval and movie-disjoint evaluation as explicitly separate diagnostics later.

## Bundled features and evaluation

The official release offers ViT frame features (K x 768), BERT description vectors (768), CLAP acoustic and HuBERT semantic features (K x 1024), in `.pt` and padded Parquet variants. These were not present in the inspected local dataset tree on Snellius; no multi-gigabyte feature downloads are needed for the smoke check. A cached SigLIP2 SO400M checkpoint is available and will encode only the selected 24 smoke frames.

Exact evaluation source: upstream `src/metrics.py`, audited separately in `metric_audit.md`. Smoke metrics use the same three training clips and are infrastructure checks, never benchmark scores.

Source: [official dataset card](https://huggingface.co/datasets/infofusionlab/Video2Reaction/blob/578d1423f89f1a7b52471b01e78770e08f0a226c/README.md). The card identifies descriptions as Movieclips scene descriptions and documents the supplied feature encoders.
