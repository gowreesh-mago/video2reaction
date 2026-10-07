# Visual trajectory results — October 7, 2026

## Outcome

All **264 benchmark runs** completed successfully by **01:03:13 Amsterdam time**: 16 primary conditions and 72 overnight conditions, each using seeds 42/43/44. All four overnight smoke jobs passed 185 tests and their 18-condition, three-video fitting/export checks. The cluster-side morning report ran at 06:00:03–06:00:32.

The strongest model selected by mean validation KL is **DINOv2 mean pooling with hidden width 256**, at **0.580511 mean test KL**. The proposed fixed-VAD trajectory models did not improve on the same encoder's simple mean-pooling baseline. Sparse relevance and rare sampling did not establish an overall predictive gain in the matched comparisons.

## Main scores

Values are means across three independently trained seeds, not predictions from an ensemble. Lower KL is better. The best VAD row is selected by mean validation KL across the tested VAD conditions.

| Model | Validation KL | Test KL | Test Top-1 F1 |
|---|---:|---:|---:|
| DINOv2 mean pooling, width 256 | 0.585566 | **0.580511** | 0.478200 |
| DINOv2 mean pooling, width 128 | 0.586576 | 0.580898 | 0.483091 |
| Free local classifier + soft relevance | 0.590918 | 0.583717 | 0.486973 |
| Free local classifier + sparse relevance | 0.591040 | 0.583976 | 0.485726 |
| Best VAD: soft relevance, initial temperature 0.1 | 0.844986 | 0.835698 | 0.418709 |
| VAD duration aggregation, initial temperature 0.25 | 0.945592 | 0.936387 | 0.404741 |
| VAD sparse relevance, initial temperature 0.25 | 1.070248 | 1.069716 | 0.429043 |
| VAD single closest moment | 1.825945 | 1.800579 | 0.121246 |

The mean-pooling width change improves test KL by only **0.000386**, with 95% paired movie-bootstrap interval **[−0.002199, +0.001402]**. The interval includes zero, so this does not establish a reliable improvement. The selected wide baseline's three test scores are 0.579657, 0.580185 and 0.581692.

## What the controlled experiments say

1. **The fixed VAD distance decoder appears to be the main limitation in this implementation.** Replacing it with unrestricted visual logits gives much better scores under the same aggregation family. This is an inference from the controls, not proof that all VAD representations fail.
2. **The peak hypothesis did not help here.** Single-peak VAD is particularly poor, and the unrestricted soft/sparse local probability mixtures still do not beat ordinary visual mean pooling on KL. These models use observed scene keyframes and full-video contextualization; they do not identify the actual moments responsible for comments.
3. **Geometry semantics are not supported by the baseline comparison.** The original sparse VAD model scores 1.069716, while its permuted-coordinate control scores 0.927439. Meaningful word-coordinate assignments are therefore not the source of its predictive advantage in this tested setup.
4. **Rare sampling has not produced a clear benefit.** Free sparse prediction worsens from 0.583976 to 0.590981 with rare sampling; its rare-class mean probability MAE also rises from 0.005599 to 0.005619. Some other metrics move differently, so this is not a claim that every rare-class statistic deteriorates. VAD duration's tiny KL change, 0.936387 to 0.935977, is insufficient by itself to establish an improvement.

## Comparison with Video2Reaction

The paper's Table 7 reports **0.588 KL** for SA-BFGS using visual and text inputs. Our strict visual baseline is numerically about **1.27% lower** at 0.580511. However, its **MRR 0.693483 / Top-1 F1 0.478200** trail that paper row's **0.709 / 0.515**. This is a favorable KL comparison, not an overall win across metrics or a matched reproduction of its input/feature pipeline. [Paper Table 7](https://arxiv.org/html/2607.06875v1#S5.T7)

Earlier project runs using SigLIP achieved lower KL (0.540655 for shared attention), but SigLIP uses language-aligned pretraining. The new batch uses purely visual DINOv2 pretraining and no description or text feature inputs. NRC contributes only fixed numeric label coordinates. Earlier text-assisted results are not evidence for the requested pure-vision method.

## Verification and limits

The morning snapshot confirms all 264 runs have successful scheduler exits and saved metrics. Audit job **27711358 completed `0:0` in 3m44s**. All official validation/test metrics for the 264 runs plus five earlier references recompute to absolute tolerance **1e−10** under the locked cluster NumPy version. Sample IDs, movie IDs, targets, frame counts and class order match; validation/test contain 1,035/2,070 distinct clips. Selected checkpoint scores match training histories, exact-reload flags and checkpoint artifacts. Per-class probability MAE also recomputes for all 264 new runs.

Selected paired differences below average each clip's losses across the three trained seeds before resampling movies (10,000 draws). Negative values favor the first model.

| Comparison | Test KL difference | 95% movie-bootstrap interval |
|---|---:|---:|
| Wide mean pooling − standard mean pooling | −0.000386 | [−0.002199, +0.001402] |
| Best VAD − standard mean pooling | +0.254801 | [+0.244745, +0.265172] |
| Free soft − free duration | −0.000151 | [−0.001625, +0.001339] |
| Free sparse − free soft | +0.000259 | [−0.000459, +0.000975] |
| Rare-sampled VAD duration − VAD duration | −0.000410 | [−0.008768, +0.008209] |
| Rare-sampled free sparse − free sparse | +0.007005 | [+0.003106, +0.010854] |
| Permuted sparse VAD − sourced sparse VAD | −0.142277 | [−0.156808, −0.128307] |

This is an exploratory matrix of **88 conditions**, with some earlier test results observed before designing the expansion. Validation ranking is used for model selection. Bootstrap intervals are conditional on these trained models and do not correct for all exploratory comparisons. Full trajectory tensors are not independently recomputed in this metrics audit; training-time exports asserted that contributions reconstruct predictions.

The official split shares movies between training and test, so these results do not establish generalization to unseen movies. The full dataset and feature caches remain on Snellius. [Verified full results JSON](trajectory_results.json) · [06:00 status JSON](trajectory_morning_status.json) · [216 overnight job IDs](trajectory_overnight_jobs.json) · [Experiment checklist](experiment_checklist.md) · [Primary implementation review](trajectory_review.md) · [Overnight review](trajectory_overnight_review.md)
