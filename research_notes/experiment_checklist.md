# Experiment checklist

Last successful cluster check: **2026-10-05 17:54:06 UTC (19:54 Amsterdam)**. **No queued or running jobs.** All 41 benchmark configurations are complete; full artifact verification passed.

## Current status

- **Completed:** audits, locked cluster uv environment, seven three-video GPU smoke runs, three full feature caches, and **41 benchmark configurations** (7 reference models + 34 recommended conditions).
- **Smoke gates:** `27611469` (18 emotion/peak/VAD variants, 2m53s) and `27611475` (five description/visual variants, 1m36s) both passed **61 tests**, exact reloads, and three-video fitting checks on A100 MIG GPUs. They exited `0:0`.
- **Primary batch:** all 19 predictors and two caches completed with exit `0:0`. [Launch receipt](recommended_primary_launch.json)
- **Peak curve:** all 15 remaining K=1/2/8 conditions completed with exit `0:0`. The full K=1/2/4/8 grid includes three emotion scores plus equal-K uniform/random controls. [Curve receipt](emotion_curve_launch.json)
- **Verified results:** all 41 runs' official validation/test scores recompute in NumPy 2.2.6; identities, selected epochs, checkpoints, and VAD assignments agree. Every saved score, selected frame, and pooling weight is verified for all 26 C/D/F models. [Complete JSON](recommended_results.json)
- **Strongest result:** visual+description KL **0.509952**, versus B1 **0.544414** (6.33% lower), with gains over both capacity-matched controls. This is text-assisted prediction. Best visual-only test KL remains shared attention at **0.540655**.
- **Peak/context finding:** every tested subset loses to all-frame B1. Uniform selection has lower KL than emotion ranking at every K. Global+peak loses to its matched global-only control; these proxies do not support peak dominance.
- **VAD finding:** auxiliary regression worsens KL. Geometry regularization beats its permuted control, but its B1 comparison includes zero; the permuted control also has slightly better expected-VAD error. No robust baseline gain is established.
- **Resources:** **87,781:19 SBU** left; today's 40 terminal jobs (39 successful, one cancelled before start) cost **76:41 SBU**. Home has about 119.18 GiB free; scratch usage is 0.1376% of 8 TiB. [Snapshot](cluster_status_20261005.json)
- **Sync:** 1,224 output files (1.86 GB), 109 logs, and the registry/results are local, synced at 17:52:24 UTC. The dataset and shared feature caches remain on Snellius.
- **Source:** all new models used smoke-tested frozen `99b59ad`; launcher-only fix `2f542f1` accepts both SLURM completion separators. Full verifier `cc69100` is pushed. Branch: `research/video2reaction-experiments`.
- **Registry:** one stale `running` entry for completed job `27612816` was reconciled using successful SLURM state and verified artifacts. Audit cross-node registry visibility before the next large concurrent batch. [Record](registry_reconciliation_20261005.json)
- **New highlight work:** three predictors implemented, reviewed, and pushed; all 80 local tests pass. GPU smoke `27623258` passed on H100 (80 tests, 3 videos, four predictors, 1m08s, exit 0:0). Full follow-up benchmarks remain on the experiment list.
- **Report:** [Current report](final_report.md), [LaTeX source](latex/video2reaction_report.tex), and full JSON cover all 41 runs; the 50-page PDF passed visual and LaTeX checks. Optional Tier 2 G/H/I/J remain deferred under the original prioritization.

Checked boxes mean completed and verified. A model passing the three-video smoke test does not complete its full benchmark experiment. Queue information is a timestamped snapshot; refresh it for each status request.

## 1. Audit and setup

- [x] Create and push a separate research branch.
- [x] Inspect repository entrypoints, models, loaders, dependencies, checkpoints, and existing outputs. [Repository audit](repo_audit.md)
- [x] Verify accessible clusters, accounts, partitions, budget, and storage. [Cluster audit](cluster_audit.md)
- [x] Inspect the actual dataset on Snellius: splits, labels, frame paths/order, metadata, imbalance, and leakage. [Dataset audit](dataset_audit.md)
- [x] Audit official metrics and verify Top-k F1 against a hand-worked example and the original implementation. [Metric audit](metric_audit.md)
- [x] Create the locked uv environment on Snellius and verify required imports and CUDA.
- [x] Organize cluster code, data links, feature cache, outputs, logs, and results. [Layout](cluster_layout.md)
- [x] Provision sourced NRC VAD coordinates privately on the cluster, with provenance.
- [x] Implement the locked shared registry and test concurrent writers.
- [x] Keep Video2Reaction frames, split files, and feature tensors on the cluster.
- [x] Install and verify CodeGraph v1.6.1 on Snellius without changing the uv environment. Project indexing remains a user decision.
- [x] Synchronize generated outputs, checkpoints, predictions, logs, and collected results to the Mac; add a repeatable sync command and receipt. [Layout and commands](cluster_layout.md)

## 2. Three-video smoke checkpoint

- [x] Push and synchronize the code before submission.
- [x] Run exactly three training videos, eight chronological frames each, through frozen SigLIP2 on a GPU.
- [x] Exercise mean pooling, temporal transformer, composite loss, and reaction-query attention.
- [x] Verify finite losses, normalized predictions, decreasing training loss, checkpoint reload, and optimizer resume.
- [x] Pass all 14 mock tests locally in conda `torch` and on the cluster in uv.
- [x] Inspect saved artifacts and record scheduler completion. [Smoke report](smoke_report.md) · [Results](smoke_results.json)

| Run | Cluster / partition | Job ID | Status | Exit / elapsed | Scope |
|---|---|---|---|---|---|
| `smoke_3videos_27381153` | Snellius / `gpu_mig` | `27381153` | completed | `0:0` / `00:08:04` | Infrastructure validation on 3 training videos |
| `smoke_3videos_27381915` | Snellius / `gpu_mig` | `27381915` | completed | `0:0` / `00:05:41` | Reviewed code: 21 tests plus the same 3-video GPU integration check |
| `smoke_3videos_27382205` | Snellius / `gpu_mig` | `27382205` | completed | `0:0` / `00:03:31` | Corrected shared image-only loader: 22 tests and 3-video GPU integration check |
| `smoke_3videos_27438252` | Snellius / `gpu_mig` | `27438252` | completed | `0:0` / `00:00:48` | 27 tests and 3-video regression including shared-query control and full attention recorder; source `4fd4068` |
| `smoke_emotion_3videos_27611469` | Snellius / `gpu_mig` | `27611469` | completed | `0:0` / `00:02:53` | 61 tests and 3-video C/D/F/VAD integration; source `99b59ad` |
| `smoke_descriptions_3videos_27611475` | Snellius / `gpu_mig` | `27611475` | completed | `0:0` / `00:01:36` | 61 tests and 3-video description/fusion/control integration; source `99b59ad` |

Tested revisions: original `08b8c30`, expanded regression `e91a469`, corrected shared loader `bbf8c1d`. All five Tier 0 models now have full validation/test results.

## 3. Work required before the first full batch

- [x] Implement the full training/evaluation entrypoint, including validation checkpoint selection, resume, and final test evaluation.
- [x] Finish and validate the reusable frozen feature cache for all 455,226 official split frames on the cluster.
- [x] Add one reproducible config and independent SLURM job per selected experiment, including hypothesis and changed component.
- [x] Add bounded batch submission (`scripts/submit_all.sh`) and result collection (`scripts/collect_results.py`); `scripts/status_all.sh` already exists.
- [x] Save per-run config, code/model/split provenance, checkpoint where applicable, predictions with sample IDs/targets/top-k, metrics, and logs for all Tier 0 runs.
- [x] Verify exact interrupted/resumed training, terminal registry protection, source isolation, cache integrity, and matched controls in synthetic tests: 21 local tests pass.
- [x] Pass all 21 tests and the three-video regression smoke in cluster uv (`27381915`, exit 0:0).
- [x] Fix the production/smoke loader mismatch and pass all 22 tests plus the three-video smoke in cluster uv (`27382205`, exit 0:0).
- [x] Recheck authorized partition/account, queue, budget, and storage. `sbatch --test-only` is also required for every job before submission.

## 4. First proposed batch — Tier 0

### Current jobs

| Experiment | Job ID | Scheduler state | Dependency |
|---|---|---|---|
| All-keyframe feature cache | `27382293` | completed, exit 0:0, 01:50:35 | none |
| B0 — Dataset prior | `27382108` | completed, exit 0:0 | none |
| B1 — Frozen visual mean pooling | `27382294` | completed, exit 0:0, 00:02:42 | successful `27382293` |
| B2 — Temporal transformer | `27382295` | completed, exit 0:0, 00:02:42 | successful `27382293` |
| B2 set control — no positions | `27382296` | completed, exit 0:0, 00:02:43 | successful `27382293` |
| A5 — KL + cosine + ranking | `27382297` | completed, exit 0:0, 00:02:42 | successful `27382293` |

Verified cache `data/features/siglip2-so400m/b69db8858136c6a0`: train **317,950/317,950**, validation **45,964/45,964**, and test **91,312/91,312** frames. Each split has its completion manifest, feature/image/index checksums, and 1,152-dimensional features. All four learned jobs consumed this cache and completed. They trained for 17/12/13/17 epochs (B1/B2/set/A5), selecting epochs 9/4/5/9 by validation KL.

### First attempt and recovery

First submission history follows. The feature job failed because `AutoProcessor(use_fast=False)` also selected a text tokenizer requiring SentencePiece. The corrected path uses `AutoImageProcessor`, shared with the smoke test, and adds a synthetic full cache round-trip test. The dependent jobs were cancelled before training; their IDs remain visible here.

| Experiment | Job ID | Scheduler state | Dependency | Completed |
|---|---|---|---|---|
| All-keyframe feature cache | `27382107` | failed: tokenizer dependency | none | No |
| B0 — Dataset prior | `27382108` | completed, exit 0:0, 00:04:51 | none | Yes |
| B1 — Frozen visual mean pooling | `27382109` | cancelled: dependency failed | `27382107` | No |
| B2 — Temporal transformer | `27382110` | cancelled: dependency failed | `27382107` | No |
| B2 set control — no positions | `27382111` | cancelled: dependency failed | `27382107` | No |
| A5 — KL + cosine + ranking | `27382112` | cancelled: dependency failed | `27382107` | No |

B0 test metrics: KL **0.689281**, cosine **0.751309**, MRR **0.599569**, F1@1 **0.237624**, F1@3 **0.558669**. [Aggregate results](tier0_prior_results.json). No visual input or test-based fitting was used. The successful result was preserved when resubmitting the other jobs with `bash scripts/submit_all.sh --skip-prior`. Dependency cancellations represent infrastructure failure, not evidence against any model hypothesis.

- [x] User requests review, then launch this batch.
- [x] Implement and validate the first-batch code, then commit, push, synchronize, and freeze it before submission.
- [x] Submit B0/B1/B2/set/A5 and record each job ID; learned predictors depend on successful feature-cache completion.
- [x] Check startup, completion/failure, predictions, metrics, and output paths for every Tier 0 run.
- [x] Summarize Tier 0 results and bounded hypothesis conclusions before selecting a later batch. [Report](tier0_report.md)
- [x] Recompute all saved validation/test metrics in the locked cluster uv environment, verify paired sample identities and checkpoint selection, and estimate paired KL intervals with 10,000 movie-cluster bootstrap resamples. B1 improves over B0; KL differences between learned models remain inconclusive.

## 5. Completed recommended batches

All prioritized C/D/E/F, VAD, and description comparisons are complete and verified, including the entire declared peak grid and matched controls.

### Tier 1 and VAD comparisons

- [x] Implement C/D/F, sourced coarse VAD, exact image alignment, interrupted cache recovery, selected-frame diagnostics, and capacity/equal-K controls; pass 37 local mock tests. [Review](tier1_peak_review.md)
- [x] Validate the sourced EmoEditor checkpoint and all 18 C/D/F/VAD variants through GPU smoke `27611469`.
- [x] C — Emotion cache `27612820` and all three evidence variants completed and verified.
- [x] B-VAD1 — Run the implemented expected-VAD auxiliary regression and semantic-permutation control (completed and verified; see job table below).
- [x] B-VAD2 — Run the implemented classifier geometry regularization and semantic-permutation control (completed and verified; see job table below).
- [x] D — Run the implemented K=1/2/4/8 grid for arousal, distance, confidence, uniform, and random selectors; B1 is the shared all-frame reference. Completed and verified after the cache and smoke.
- [x] E — Collect and verify jobs `27438329` and `27438330`, recompute official metrics in cluster uv, and compare with B1 using paired uncertainty. Checksummed attention weights, frame alignment, normalization, entropy, and shared-map equality also pass. Shared attention improves over B1; reaction-specific attention has no established advantage over its shared control. [Results](attention_results.json)
- [x] F — Run the implemented global+peak, global+global, and peak+peak comparisons, all predeclared at arousal K=4. Completed and verified after the cache and smoke.

### Primary batch completed after smoke verification

At most four jobs in this entire table can run concurrently. Cache dependencies use `afterok`; lane dependencies use `afterany`. Every job below completed with exit `0:0`; predictor artifacts passed full verification. The total requested wall-time cap is 832 SBU, not an estimated cost.

| Experiment | Job ID | Required cache |
|---|---|---|
| `b_vad_aux` | `27612811` | none |
| `b_vad_aux_permuted` | `27612813` | none |
| `b_vad_geometry` | `27612816` | none |
| `b_vad_geometry_permuted` | `27612819` | none |
| `emotion_cache` | `27612820` | none |
| `description_cache` | `27612822` | none |
| `c_emotion_logits` | `27612828` | `27612820` |
| `c_emotion_vad` | `27612829` | `27612820` |
| `c_emotion_both` | `27612830` | `27612820` |
| `f_global_peak` | `27612832` | `27612820` |
| `f_global_control` | `27612834` | `27612820` |
| `f_peak_control` | `27612836` | `27612820` |
| `d_arousal_k4` | `27612842` | `27612820` |
| `d_distance_k4` | `27612844` | `27612820` |
| `d_confidence_k4` | `27612847` | `27612820` |
| `d_uniform_k4` | `27612848` | `27612820` |
| `d_random_k4` | `27612849` | `27612820` |
| `description_only` | `27612851` | `27612822` |
| `visual_description` | `27612852` | `27612822` |
| `description_visual_control` | `27612854` | `27612822` |
| `description_text_control` | `27612856` | `27612822` |

### Completed remaining peak grid

| Experiment | Job ID | State |
|---|---|---|
| `d_arousal_k1` | `27614365` | completed, `0:0` |
| `d_distance_k1` | `27614367` | completed, `0:0` |
| `d_confidence_k1` | `27614368` | completed, `0:0` |
| `d_uniform_k1` | `27614371` | completed, `0:0` |
| `d_random_k1` | `27614372` | completed, `0:0` |
| `d_arousal_k2` | `27614373` | completed, `0:0` |
| `d_distance_k2` | `27614374` | completed, `0:0` |
| `d_confidence_k2` | `27614375` | completed, `0:0` |
| `d_uniform_k2` | `27614376` | completed, `0:0` |
| `d_random_k2` | `27614377` | completed, `0:0` |
| `d_arousal_k8` | `27614378` | completed, `0:0` |
| `d_distance_k8` | `27614379` | completed, `0:0` |
| `d_confidence_k8` | `27614380` | completed, `0:0` |
| `d_uniform_k8` | `27614381` | completed, `0:0` |
| `d_random_k8` | `27614382` | completed, `0:0` |

### Required description diagnostics

- [x] Complete description smoke `27611475`, including the real tokenizer, text encoder, and all predictor/control variants.
- [x] Extract the full description cache and evaluate description-only prediction.
- [x] Evaluate visual + description prediction against B1 and both implemented capacity-matched single-modality controls.
- [x] Report text-assisted and video-only results separately, including the observed cross-split movie overlap.

### Additional controlled comparisons and optional Tier 2

- [ ] Decide whether full A1–A4 loss comparisons add useful evidence after A5. CE/KL gradient equivalence is already unit-tested; loss functions are implemented.
- [ ] G — Reaction text prototypes.
- [ ] H — Retrieval from training clips only, with leakage checks.
- [ ] I — Hierarchical reaction prediction.
- [ ] J — Entropy-aware auxiliary prediction.

## 6. Newly requested highlight baselines

Added 2026-10-05; these are separate from the 41 completed configurations. [Methods and adversarial review](highlight_review.md).

- [x] Implement jointly trained sparsemax highlight selection plus reaction prediction (`joint_highlight_sparse`).
- [x] Implement an identical softmax scorer/control (`joint_highlight_soft_control`).
- [x] Pin and download DSNet TVSum split-0 and its GoogLeNet backbone on Snellius; verify source and weight hashes.
- [x] Implement frozen DSNet Top-4 pseudo-label selection and the unchanged B1 predictor (`pretrained_dsnet_k4`), with aligned resumable cache extraction.
- [x] Add configs, independent SLURM jobs, frame diagnostics, and a three-video smoke covering all new predictors; all 80 local mock tests pass.
- [x] Push `71b3b61`, synchronize, verify DSNet assets, and freeze `71b3b610e96d-6dd469fd221b`.
- [x] Pass the new three-video GPU smoke in uv: job `27623258`, `gpu_h100`, 1m08s, exit `0:0`; 80 tests plus four predictor fits/reloads/resumes passed. It replaces cancelled-pending MIG job `27622854`, which did not train. [Smoke evidence](highlight_smoke_results.json) · [Launch receipt](highlight_smoke_launch.json)
- [ ] Full joint sparse/soft benchmark runs.
- [ ] Full DSNet score cache and Top-4 benchmark run.
- [ ] Verify metrics, selectors, support/collapse statistics, matched comparisons, and update the report with new results.

The new full benchmark runs are listed as pending. Exact launch command for this smoke-tested source when launching the follow-up batch:

```bash
cd /scratch-shared/gmago/video2reaction/releases/71b3b610e96d-6dd469fd221b
source configs/clusters/snellius.env
python3 scripts/submit_highlights.py --completed-smoke-job 27623258
```

This submits three predictors and the DSNet cache, with at most two concurrent jobs. The original 41 benchmark results remain unchanged.

## 7. Evaluation and final report

- [x] Collect and recompute official distribution/ranking/Top-k metrics for all 41 completed benchmark runs.
- [x] Collect all runs' per-class precision, recall, F1, target support, and prediction support.
- [x] Save all runs' diagnostics by target entropy quartile, frame count, and dominant target probability, with cut points derived from training.
- [x] For peak/query experiments, save selected indices/timestamps, emotion/VAD scores, and attention weights as applicable.
- [x] Save 20 prediction examples: 5 improvements, 5 failures, 5 high-entropy, and 5 low-entropy clips, with distinct movie IDs and all seven reference-model distributions. The LaTeX appendix shows probabilities and IDs; visual scene explanations remain unverified.
- [x] Update `research_notes/final_report.md` and the self-contained LaTeX report with all 41 runs and controlled peak/VAD/description conclusions. The 50-page PDF passed visual inspection and compiled without warnings.
- [x] Answer whether peaks, global context, or both help, and whether VAD improves prediction, using completed controlled comparisons.

## Status update procedure

1. Refresh Snellius `squeue`, `sacct` for recorded jobs, and the shared JSON registry. Compare scheduler state with per-run artifacts before marking completion.
2. Update this file's timestamp, checkboxes, job table, failures/blockers, and next proposed batch. Keep failed attempts and replacement job IDs visible.
3. Report what is implemented, submitted, running, completed, failed, and still pending. Record unavailable checks explicitly rather than treating stale state as current.
4. Follow the active goal's authorized experiment sequence, with review, smoke, and live resource checks before each new pipeline. Keep one seed (42), fixed official splits, validation-based selection, and matched controls. Use uv on the cluster; local conda `torch` is for mock tests.

Cluster root: `/scratch-shared/gmago/video2reaction`.

Initial release / completed B0: `releases/e91a469f51ba-b53672620486`. Corrected release / completed Tier 0 jobs: `releases/bbf8c1d019a4-9a6a80a3d307`. Source hash suffixes are prefixes; full hashes are saved in each run's code manifest. Frozen releases remain unchanged while the working-copy checklist is updated.

| Record | Cluster path relative to project root |
|---|---|
| Shared registry | `results/experiment_registry.json` |
| Smoke artifacts | `outputs/smoke/smoke_3videos_27381153/` |
| Smoke SLURM log | `logs/slurm/smoke-27381153.out` |
| Code and this checklist | `code/` and `code/research_notes/experiment_checklist.md` |

Read-only status commands on Snellius:

```bash
cd /scratch-shared/gmago/video2reaction/code
bash scripts/status_all.sh
sacct -j 27381153 --format=JobIDRaw,JobName,State,ExitCode,Elapsed -X
sacct -j 27381915,27382107,27382108,27382109,27382110,27382111,27382112 --format=JobIDRaw,JobName,State,ExitCode,Elapsed -X
sacct -j 27382205,27382293,27382294,27382295,27382296,27382297 --format=JobIDRaw,JobName,State,ExitCode,Elapsed -X
source configs/clusters/snellius.env
python3 scripts/collect_results.py --reconcile
```
