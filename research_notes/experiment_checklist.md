# Experiment checklist

Latest verified status, **2026-10-06 at 19:47 UTC (21:47 Amsterdam)**: DINOv2 cache **27682121 completed successfully** at 21:32:13 Amsterdam, exit `0:0`, **1h05m05s**. Its artifacts cover all **455,226 frames** across the official splits. Of 48 predictors, **one is running** (`traj_vad_duration_s42`, **27682123**, started 21:46:18); **one awaits GPU resources** (`traj_vad_peak_s42`, **27682125**); **46 await dependencies**. No full trajectory benchmark has completed or failed at this check. Three-video smoke **27681517 passed all 16 variants and 111 tests**, exit `0:0`, 16m31s. Source `0533611` is pushed and runs from frozen release `05336113586d-bd1c51a39c84`. [Launch receipt](trajectory_full_launch.json) · [Cache results](trajectory_cache_results.json) · [Smoke results](trajectory_smoke_results.json) · [Implementation/adversarial review](trajectory_review.md)

### Deadline estimate: October 7, 22:00 Amsterdam (CEST)

**Updated at 21:47 Amsterdam:** cache completion at **21:32** met the estimate below. The first predictor started after a **14-minute queue wait** and completed dependency warm-up without an error in the inspected log; the second still awaits a GPU. If both lanes become available promptly and each run completes within its 45-minute allocation, the batch can finish around **16:00 tomorrow**, retaining roughly six hours before the deadline. Successful full-data runtimes are still unmeasured, so this remains conditional. Sustained availability of only one lane would invalidate the two-lane estimate.

At the October 6, 20:44 Amsterdam check, extraction had logged **98,304 of 455,226 total frames** (98,304/317,950 training frames; validation/test follow). Recent throughput was approximately **128 frames/second**, leaving about 47 minutes of encoding plus split setup and cache writes. Estimated cache completion: **21:30–22:00 tonight**, provided throughput remains similar.

The 48 predictor allocations, each capped at 45 minutes, form two dependency chains: **24 × 45 minutes = 18 hours** with continuous availability of both slots. A 22:00 cache completion therefore permits dispatch through roughly **16:00 tomorrow**, leaving about **six hours** before the requested deadline for queue gaps, verification or retries. This is an allocation-based planning estimate, not a measured training-completion forecast: no full trajectory predictor has started yet, and exceeding its limit would produce a timeout rather than a completed experiment. At this snapshot, the only pending MIG jobs were these 48 dependency-held runs; Slurm supplied no start estimates while the cache dependency remained unmet. The deadline looks achievable but is not guaranteed. No jobs, concurrency settings or training settings were changed for this estimate.

- [ ] Once the first full predictors finish, replace the allocation-based estimate with observed runtimes and check all jobs for timeouts/failures.

Earlier smoke attempts remain recorded: `27680963` exposed unstable single-peak fitting at learning rate 0.003; the shared rate was corrected to 0.0003. `27681172` passed ten variants before its shortened ten-minute allocation timed out. The unchanged corrected source passed under its original twenty-minute ceiling.

Live Snellius access was verified this turn. The previous DSNet cache `27662939` and predictor `27662940` both completed with exit `0:0` (1h11m41s and 3m54s); their new results have not yet been audited. The earlier 41 benchmarks and two joint runs have verified metrics. [Highlight launch receipt](highlight_full_launch.json) · [Joint results](highlight_joint_results.json)

## Current status

- **Completed:** audits, locked cluster uv environment, eight successful three-video GPU smoke runs and **44 benchmark configurations by scheduler state** (7 reference models + 34 recommended conditions + 2 joint models + DSNet selection). Metrics for 43 are verified; the DSNet result and full selector audit for the two joint models remain pending. The new DINOv2 cache is complete; one of 48 trajectory predictors is running and 47 are pending.
- **Smoke gates:** `27611469` (18 emotion/peak/VAD variants, 2m53s) and `27611475` (five description/visual variants, 1m36s) both passed **61 tests**, exact reloads, and three-video fitting checks on A100 MIG GPUs. They exited `0:0`.
- **Primary batch:** all 19 predictors and two caches completed with exit `0:0`. [Launch receipt](recommended_primary_launch.json)
- **Peak curve:** all 15 remaining K=1/2/8 conditions completed with exit `0:0`. The full K=1/2/4/8 grid includes three emotion scores plus equal-K uniform/random controls. [Curve receipt](emotion_curve_launch.json)
- **Verified results:** all 41 runs' official validation/test scores recompute in NumPy 2.2.6; identities, selected epochs, checkpoints, and VAD assignments agree. Every saved score, selected frame, and pooling weight is verified for all 26 C/D/F models. [Complete JSON](recommended_results.json)
- **Strongest result:** visual+description KL **0.509952**, versus B1 **0.544414** (6.33% lower), with gains over both capacity-matched controls. This is text-assisted prediction. Best visual-only test KL remains shared attention at **0.540655**.
- **Peak/context finding:** every tested subset loses to all-frame B1. Uniform selection has lower KL than emotion ranking at every K. Global+peak loses to its matched global-only control; these proxies do not support peak dominance.
- **VAD finding:** auxiliary regression worsens KL. Geometry regularization beats its permuted control, but its B1 comparison includes zero; the permuted control also has slightly better expected-VAD error. No robust baseline gain is established.
- **Resources (October 5 snapshot):** **87,781:19 SBU** left; that day's 40 terminal jobs (39 successful, one cancelled before start) cost **76:41 SBU**. Home had about 119.18 GiB free; scratch usage was 0.1376% of 8 TiB. [Snapshot](cluster_status_20261005.json)
- **Sync:** 1,655 output files (2.30 GB), 128 logs and 43 registry/result files are local, synced on October 6 at **19:47:19 UTC**. This includes the completed DINO cache metadata, the first predictor startup log, successful trajectory smoke, prior failed attempts, launch receipts and full scene-duration audit. The dataset and shared feature tensors remain on Snellius.
- **Source:** the 34 recommended conditions used smoke-tested frozen `99b59ad`; the highlight follow-up uses frozen `71b3b61`. Launcher-only fix `2f542f1` accepts both SLURM completion separators. Full verifier `cc69100` is pushed. Branch: `research/video2reaction-experiments`.
- **Trajectory source:** `0533611`, frozen release `05336113586d-bd1c51a39c84`, passed 111 cluster tests and all 16 three-video variants. Full cache `27682121` completed **317,950/45,964/91,312 train/validation/test frames**, all 768-dimensional, with feature/index/image hashes recorded. All 48 predictor IDs are listed below.
- **Registry:** one stale `running` entry for completed job `27612816` was reconciled using successful SLURM state and verified artifacts. Audit cross-node registry visibility before the next large concurrent batch. [Record](registry_reconciliation_20261005.json)
- **New highlight work:** sparse/soft predictors `27662937`/`27662938` completed with exit `0:0` in 3m49s/3m50s. Test KL: sparse **0.551409**, soft **0.542966**, B1 **0.544414** (lower is better). Sparse loses to its matched soft control; soft versus B1 is inconclusive under paired movie bootstrap. DSNet cache `27662939` and Top-4 predictor `27662940` have now completed successfully; the latter's metrics are not yet audited.
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
- [x] Recheck budget/quota, pass the same-source smoke gate and all four scheduler preflights, and submit the full batch on October 6. Budget before submission: **87,781:19 SBU**, with dispatch/submission allowed. [Full launch receipt](highlight_full_launch.json)
- [x] Full joint sparse/soft benchmark runs; both completed with exit `0:0`. Validation/test metrics, sample alignment, checkpoint selection, and exact-reload flags verified against saved artifacts in cluster NumPy 2.2.6. [Results](highlight_joint_results.json)
- [ ] Full DSNet score cache and Top-4 benchmark run.
- [ ] Verify metrics, selectors, support/collapse statistics, matched comparisons, and update the report with new results.

### Full highlight batch — submitted October 6

| Experiment | Job ID | State at 14:35:00 UTC | Dependency |
|---|---|---|---|
| Joint sparse highlights | `27662937` | completed, `0:0`, 3m49s | none |
| Matched soft-attention control | `27662938` | completed, `0:0`, 3m50s | none |
| Frozen DSNet score cache | `27662939` | running, 1h03m50s | after `27662937` ends (satisfied) |
| DSNet Top-4 reaction predictor | `27662940` | pending | successful `27662939`, and after `27662938` ends |

All four use `gpu_mig`, account `gusr133332`, and frozen `71b3b610e96d-6dd469fd221b`. Both completed joint predictors saved matching source manifests and used an NVIDIA A100 MIG 3g.20gb GPU. Both trained 13 epochs and selected epoch 5 using validation KL.

The sparse model's test KL exceeds the soft control by **0.008443**, with paired movie-bootstrap 95% CI **[0.005661, 0.011282]**. Against B1, sparse is worse by **0.006995 [0.002009, 0.011919]**; soft is lower by **0.001448**, but its difference CI **[-0.005379, 0.002529]** includes zero. These 10,000-resample intervals describe test-sample uncertainty for seed 42, not training-seed variability. Saved diagnostics report sparse support averaging 20.08 frames and 55.13% of each clip's frames; the full selector artifact audit remains pending. These results do not support a gain from this sparse selector.

Exact launch command used:

```bash
cd /scratch-shared/gmago/video2reaction/releases/71b3b610e96d-6dd469fd221b
source configs/clusters/snellius.env
python3 scripts/submit_highlights.py --completed-smoke-job 27623258
```

This submitted three predictors and the DSNet cache, with at most two concurrent jobs. The launcher records IDs per source hash and reuses them on repeated invocation. The original 41 benchmark results remain unchanged.

## 7. Evaluation and final report

- [x] Collect and recompute official distribution/ranking/Top-k metrics for all 41 completed benchmark runs.
- [x] Collect all runs' per-class precision, recall, F1, target support, and prediction support.
- [x] Save all runs' diagnostics by target entropy quartile, frame count, and dominant target probability, with cut points derived from training.
- [x] For peak/query experiments, save selected indices/timestamps, emotion/VAD scores, and attention weights as applicable.
- [x] Save 20 prediction examples: 5 improvements, 5 failures, 5 high-entropy, and 5 low-entropy clips, with distinct movie IDs and all seven reference-model distributions. The LaTeX appendix shows probabilities and IDs; visual scene explanations remain unverified.
- [x] Update `research_notes/final_report.md` and the self-contained LaTeX report with all 41 runs and controlled peak/VAD/description conclusions. The 50-page PDF passed visual inspection and compiled without warnings.
- [x] Answer whether peaks, global context, or both help, and whether VAD improves prediction, using completed controlled comparisons.

## 8. Paper-inspired follow-up — proposed, not implemented or launched

Reviewed October 6: Agarwal et al., [Why Do Vision Language Models Struggle To Recognize Human Emotions?, v2](https://arxiv.org/html/2604.15280v2), including the methods, limitations and supplementary results. The [project page](https://madhav1ag.github.io/vlm-temporal-emotion-gap/) links a paper and demo but no runnable implementation.

### Evidence and relevance

The paper studies facial-expression classification on balanced MAFW/DFEW subsets. Its MSCE method summarizes four intermediate frames per temporal gap and interleaves these summaries with keyframes. Qwen2.5-VL macro-F1 improves from 0.2449 to 0.2731 on MAFW and 0.4552 to 0.4820 on DFEW (Section 4.2, Table 3). These scores are not Video2Reaction results.

Our most promising transfer is generated scene-change descriptions: visual + supplied description already achieves KL 0.509952, versus B1 0.544414. This motivates an experiment; it does not establish that generated descriptions will help. Our existing peak subsets and joint sparse selector have not improved B1. All-frame mean pooling has no VLM context-window truncation, so its results do not diagnose the paper's proposed attention bottleneck.

### A. Cheap first experiment: rare-reaction training

- [ ] Audit per-class target/prediction mass, macro Top-k F1, supported-class recall and probability errors for B1 and visual + description. Define rare/common groups using training target mass only; retain all official test clips and disclose zero-support classes.
- [ ] Compare equal-step second-stage head training from the same checkpoint using natural sampling, sampling that favors clips containing rare reaction probability mass, and an importance-corrected sampling control. Freeze the learned representation in every arm; keep the original 21-way soft targets and prediction head.
- [ ] Specify sampling strength/caps and seeds using training/validation only. A possible starting weight is the sum of each clip's class probabilities divided by the square root of training class prevalence, with a cap to avoid a few clips dominating. This is our adaptation, not a formula from the paper.
- [ ] Evaluate official KL/MRR/weighted F1 alongside macro/per-class diagnostics. Report improved rare-class recognition with worse KL as a tradeoff, not a distribution-prediction improvement. Use at least three paired training seeds for any follow-up gain claim.

Dominant-label balancing is unsuitable here: annoyance and embarrassment have zero training clips where they are the largest target, despite nonzero target probability mass. Reweighting may distort population calibration. The importance-corrected arm preserves the natural average training objective in expectation and helps separate a changed objective from changed sampling.

### B. Main research experiment: generated temporal context

- [ ] Verify whether raw videos or dense frame sequences exist on Snellius. Only scene keyframes are currently verified. A pilot using those keyframes can test scene-level context; it cannot recover unobserved micro-expressions.
- [ ] On three training videos, generate short descriptions of visible actions, interactions and changes from small ordered frame windows with a frozen, pinned local VLM. Keep comments, target distributions and movie identity out of generation inputs. Audit hallucinations and measure runtime before full caching.
- [ ] Compare the same visual encoder and predictor capacity with: supplied descriptions, generated static descriptions, and generated temporal descriptions. Match generator, visible frames, calls and output token budget between generated-text arms. Include a summary-only diagnostic; retain all-frame visual context in the combined arms.
- [ ] Keep segment text embeddings and timestamps separate for an order-aware predictor. Our current `encode_descriptions` averages token-chunk embeddings into one vector, and the current fusion head mean-pools; simply appending a long timeline would not implement chronological interleaving. Add an identically sized position-free control.
- [ ] Probe shuffled/reversed segment order and shuffled frames before summary generation. Distinguish new visual evidence, extra teacher computation, static semantic information and useful temporal order. An order effect alone does not prove correct temporal reasoning.

Primary hypothesis: generated descriptions of scene changes improve reaction prediction beyond equally budgeted static descriptions. A temporal claim requires an advantage over the position-free control as well as appropriate order sensitivity. A gain over visual-only B1 alone is insufficient, because extra language information could explain it. Stage one must observe every frame it describes; timestamps alone cannot supply unseen motion.

### Interpretation limits and execution order

Character emotion is an input cue for audience reactions, not a substitute target: a frightened character may amuse a viewer. The paper's lexical-frequency analysis is correlational. Its balanced-LoRA versus zero-shot comparison also changes training exposure; our natural-sampling fine-tuning control is therefore necessary. Generation gains also need controls for additional frames and compute.

Run the rare-reaction audit and small generation pilot first, then fix the experiment specification before full training. Existing visual/description caches can support the cheap training comparison. New summary caches, dense videos and model weights stay on Snellius; every new pipeline needs the existing local mocks, pushed source, cluster uv and three-video smoke gate. This section records proposals only; the user has not requested their implementation or launch in this paper-reading turn.

## 9. User proposal: VAD trajectories, salient moments and rare-reaction training

Recorded October 6. The user subsequently authorized implementation and launch. The trajectory batch below implements the refined visual-only mechanism. Earlier experiments did not implement this complete mechanism.

**Input constraint from the user:** this experiment must be purely visual. Moment selection, relevance and VAD prediction must use RGB frames or video windows, with a visual encoder and temporal context. Do not use supplied/generated descriptions, captions, comments, text embeddings, text-to-VAD projection, or language-model outputs as input features or teacher proxies. Use a visual backbone trained without text supervision for the strict primary arm. NRC supplies only the fixed numeric VAD coordinates of the official emotion labels, as requested; existing audience reaction distributions remain the training targets. The text-based proposals in Section 8 are separate and outside this experiment.

### What already exists, and what is missing

| Existing experiment | Implemented mechanism | Difference from this proposal |
|---|---|---|
| `b_vad_aux` | Regress the target distribution's expected VAD as an auxiliary task | Final probabilities still come from an unrestricted 21-class head; no local peak/rest decomposition |
| `b_vad_geometry` | Regularize classifier-weight similarities using fixed VAD distances | VAD distances do not produce the output probabilities |
| C/D/F emotion and peak runs | Estimate frame VAD from eight frozen emotion probabilities; select peaks or fuse global/peak visual features | Output uses the learned classifier; F's global branch includes peak frames rather than representing the remaining video exclusively |
| Joint highlight and DSNet baselines | Learned or pretrained frame selection followed by reaction prediction | No fixed VAD prototype decoder or rare-reaction objective |

All implemented VAD assets use NRC **v1**, with values in [0,1]. The requested [NRC VAD v2 paper](https://arxiv.org/html/2503.23547) specifies [-1,1] scores; the [author's download page](https://saifmohammad.com/WebPages/nrc-vad.html) currently provides **v2.1**. Provision a separate hashed asset, verify all 21 exact labels and the scale, and preserve the benchmark class order. Do not replace old assets or reinterpret prior results.

### Initial peak/rest special case

Interpretation: the 21 emotion coordinates remain fixed, while each video's local peak and remaining context determine its predicted position(s) in that coordinate space.

For fixed label coordinates `e_c`, infer a peak representation `z_peak` and a rest representation `z_rest`, each in the same three-dimensional VAD space. Convert distances into probabilities separately:

```text
p_peak(c) = softmax_c(-||z_peak - e_c||^2 / tau_peak)
p_rest(c) = softmax_c(-||z_rest - e_c||^2 / tau_rest)
p(c) = alpha * p_peak(c) + (1 - alpha) * p_rest(c)
```

Temperatures must be positive; alpha lies in [0,1] and may be predicted from the video. Exclude selected peak moments from the rest branch in the hard-selection version. Mixing the two distributions preserves two distinct affective modes; averaging the two VAD points first can erase them. Start with an isotropic metric and no free class-specific logits or biases, so the primary experiment actually tests distance-based prediction. A later semantic residual must be a separately named model with a distance-only ablation.

### Proposed work and controls

- [ ] Implement the fixed-prototype distance decoder and NRC v2.1 provisioning, with normalization, finite-gradient, scale and class-order checks.
- [ ] Establish how visual moments enter VAD space using only visual features. Train a shared three-dimensional projection from contextualized video-window features, supervised through the final reaction-distribution loss and the fixed prototype decoder. This weak supervision does not uniquely establish the true VAD of each moment. An optional visual-only emotion teacher can provide an auxiliary anchor, but its supervision, class coverage and domain mismatch must be recorded and ablated. Do not use a text-to-VAD projection or language-generated labels as a proxy. Retain the full target distribution as training supervision.
- [ ] Use a predeclared local-window selector and peak/rest budget. Compare learned selection with uniform windows. If adding DSNet or affective teacher supervision, identify it explicitly as a pseudo-label highlight objective and compare against the same selector trained only through reaction KL. High arousal must not be assumed equivalent to audience relevance or rare-emotion evidence.
- [ ] Run the main 2-by-2 comparison: global VAD decoder versus peak/rest VAD decoder, each with ordinary versus rare-reaction training. Use the same source features, decoder family, optimization budget and paired seeds; add a matched global/global branch control for the extra mixture capacity.
- [ ] Add a fixed permutation of the 21 VAD-to-label assignments with identical coordinate geometry, and equal-budget uniform/random selection controls. This separates semantic coordinates from bottleneck regularization and selection from retained frame count. Add peak-only and rest-only ablations to test which branch contributes.
- [ ] Derive rare-reaction weights from training probability mass, with capped weighting or sampling. Do not convert soft targets to dominant labels, and compare against an importance-corrected sampling control. Report official KL/MRR/weighted F1, macro/per-class measures, target versus predicted class mass, and paired movie-bootstrap intervals across multiple training seeds.
- [ ] Record alpha, both branch distributions/VAD points, frame timestamps, selection budget, coordinate provenance, temperature, support statistics and any highlight pseudo-label loss. Run a three-video smoke in cluster uv after local review/mocks and push/sync.

### What this would establish

A gain requires the combined model to beat its VAD-only, peak/rest-only-without-rebalancing, matched global/global and semantic-permutation controls. Better rare-class recall with worse official KL is a tradeoff, not an overall gain. Fixed geometry may share statistical strength across nearby reactions, but it does not inherently boost rare classes; labels with similar VAD can remain difficult to distinguish. Word norms are not video labels, and a character's expressed emotion can differ from the audience's induced reaction. Existing negative peak results and inconclusive VAD gains do not test this specific distance-decoded combination.

### Refined hypothesis: comments concentrate on a few useful moments

The user's clarification is that a video traces a path through VAD space, different emotions can peak at different times, and only some moments attract most comments. Duration and proximity should therefore be weighted by reaction relevance. The two-point peak/rest model above is a restricted comparison; it is not the full trajectory hypothesis. A long quiet passage should not necessarily outweigh a brief salient event.

The proposed data path is `RGB windows -> visual temporal encoder -> {3D VAD point, relevance score} per window -> fixed-coordinate distance decoder -> weighted probability mixture`. Both prediction branches use visual features. Full-video visual context may inform a local window, but no text representation enters the path. Include a matched unrestricted visual reaction head to test the cost or benefit of the three-dimensional bottleneck.

For nonoverlapping intervals `i`, let `dt_i` be actual duration in seconds, `z_i` the estimated VAD position, and `e_c` a fixed emotion coordinate. Define proximity and a momentary categorical distribution:

```text
k_ic = exp(-||z_i - e_c||^2 / (2 * sigma^2))
q_ic = k_ic / sum_c k_ic
```

The proposed main model predicts a nonnegative relevance density `a_i` from video context, allowing zero or negligible values for irrelevant intervals. Normalize duration times relevance into moment weights, then mix the momentary distributions:

```text
w_i = dt_i * a_i / sum_j(dt_j * a_j)
P(c | video) = sum_i w_i * q_ic
```

Interpretation under the model: `w_i` is the share of reaction mass associated with interval `i`, and `q_ic` is the emotion distribution conditional on that interval contributing. Clip-level labels alone cannot verify that these are the actual moments viewers commented on. Relevance is not automatically arousal, teacher confidence, closeness to a prototype, or generic visual highlightness. The gate may use full-video context so that a payoff can be interpreted using its setup. Enforce positive total weight and record any fallback for an all-zero gate.

This weighted mixture is already a probability distribution. `a_i = 1` gives duration-weighted averaging; concentration on one interval gives a single-peak model. Multiple nonzero intervals allow different comments to concern different moments. We should aggregate local probabilities before collapsing VAD coordinates: averaging opposite VAD positions can erase both emotional modes.

An additional proximity-sensitive variant accumulates raw affinities:

```text
E_c = sum_i dt_i * a_i * k_ic
P_proximity(c) = E_c / sum_c E_c
```

This is not equivalent to averaging `q_ic` with the same weights. It also weights each interval by its total affinity to all prototypes, preserving absolute proximity but potentially favoring densely packed prototype regions. Compare these variants explicitly. Normalized distances are model scores; calibration to audience reactions must be evaluated.

For the user's illustrative love/fear example, if the 6-second interval has probabilities `(0.9, 0.1)` and the 3-second interval `(0.4, 0.6)`, equal relevance yields `(0.7333, 0.2667)`. Different relevance can reverse the duration advantage. Love is an illustrative label only; it is not part of the official 21-label taxonomy.

### Aggregation literature and proposed comparisons

- [Attention-based Deep Multiple Instance Learning, Ilse et al.](https://proceedings.mlr.press/v80/ilse18a.html) learns instance contributions using bag-level supervision. Adapting this to a mixture of categorical reaction distributions is our proposal; its original formulation does not establish comment attribution.
- [Sparsemax, Martins and Astudillo](https://proceedings.mlr.press/v48/martins16.html) can assign exactly zero attention weight to some instances. It provides a way to test sparse selection against soft attention, but sparsity alone does not identify useful moments. Specify whether the gate represents interval mass or per-second density before incorporating duration; do not count duration twice.
- [Power pooling, Liu et al.](https://arxiv.org/html/2010.09985) interpolates from mean to max pooling with `s_c = sum_i(q_ic^(r+1)) / sum_i(q_ic^r)`, `r >= 0`. Our comparison would add `dt_i` to both sums and normalize `s_c` across classes. These are adaptations of a sound-event presence model. Strong power pooling can discard relative event duration, so it is a peak-sensitive control rather than a faithful duration accumulator.
- [Auto-pool, McFee et al.](https://arxiv.org/html/1804.10070) learns softmax weights on instance predictions. It offers another mean-to-peak comparison, with the same distinction between event presence and audience reaction shares.

- [ ] Start with the same momentary decoder and features across duration-only, single-peak, per-emotion peak, power-pooling, soft-relevance and sparse-relevance aggregation. For per-emotion closest approach use `max_i k_ic` then normalize across classes; the maximum of `q_ic` need not occur at minimum absolute VAD distance.
- [ ] Compare relevance-weighted local probabilities with relevance-weighted raw proximity. Keep rare-reaction reweighting fixed initially; add it as a separate factorial comparison after isolating aggregation.
- [ ] Compare uniform and learned relevance using the same fixed VAD decoder; compare VAD and unrestricted momentary classifiers under the same relevance mechanism. Existing joint baselines pool visual features before a classifier, so they do not isolate the proposed probability mixture.
- [ ] Compare matched contiguous windows and random windows, using duration budgets in seconds. Use fixed-window averaging or smoothing to test whether peak gains survive removal of isolated noisy frames. Choose all budgets and hyperparameters on training/validation data.
- [ ] Check interval-splitting invariance for a fixed trajectory and relevance density: subdividing a constant interval must not change its contribution. Scene timestamps permit a piecewise-constant approximation only; verify dense observations before claiming measured within-scene dwell times. Handle overlapping windows without counting time repeatedly.
- [ ] Save per-interval VAD, proximity, probabilities, relevance, duration and per-emotion contribution `w_i * q_ic`; report attention support and concentration. Evaluate the official clip metrics and rare-class diagnostics with matched seeds and movie-level intervals.
- [ ] Audit whether selected windows correspond to reaction-relevant content using independent temporal annotations or a separately held-out alignment study if such data become available. Clip-level gains and attention plots alone cannot prove that comments concern the selected moments. Removal tests are supporting diagnostics and can also disrupt narrative context.

The primary test is whether learned relevance improves over duration-only and peak-only aggregation with the same local decoder architecture, and whether sparse relevance improves over an equally trained soft gate. Better clip predictions support this predictive model, but do not by themselves prove the hypothesized comment-generation process. A negative result constrains the tested representation, supervision and pooling choices; it does not eliminate every VAD trajectory model.

### Authorized visual-only batch — implementation and launch checklist

- [x] Recheck Snellius access, account `gusr133332`, existing jobs, keyframe timestamps and available caches before code changes.
- [x] Audit all 455,226 source scene intervals across 7,243/1,035/2,070 train/val/test clips: positive durations, zero gaps, zero overlaps. Scene durations range from 0.04 to 138.08 seconds; estimates remain constant within each observed scene.
- [x] Add a separate pinned DINOv2 cache and provision its visual-only weights on Snellius; no SigLIP/text representation enters this batch.
- [x] Validate exact NRC v2.1 entries for all 21 labels, preserve class order, and transfer its archive to the private cluster cache.
- [x] Implement duration, single-peak, per-emotion peak, soft/sparse relevance, power and raw-proximity aggregation using actual seconds.
- [x] Add unrestricted visual decoder, semantic permutation, timestamp-free and rare-sampling/importance controls. Configure 16 variants × 3 paired seeds with official splits and validation KL selection.
- [x] Export scene identities, local VAD/probabilities, relevance, durations and contributions that reconstruct predictions. Add training-defined rare-class diagnostics.
- [x] Complete code and adversarial hypothesis review, including correction of the duration confound in order perturbation. [Review](trajectory_review.md)
- [x] Pass all 111 local regression/synthetic tests in conda `torch`, including concentrated-target checks added after the first smoke failure; no video dataset was copied to the Mac.
- [x] Push/sync and freeze corrected release `05336113586d-bd1c51a39c84`; provision the separately hashed NRC v2.1 JSON using cluster uv. Retain failed first release `45a69ea34db0-98455bd0db4f` unchanged.
- [x] Pass exact-source three-video/24-keyframe uv smoke **27681517** across all 16 variants, with **111 tests**, verified exports, reloads and optimizer resume; exit `0:0` / `00:16:31`.
- [x] Recheck resources and preflight all requests; launcher **27681911** submitted cache **27682121** and all 48 predictors, with two-job concurrency and successful-cache dependencies.
- [x] Verify cache **27682121** completed `0:0` in 1h05m05s and its recorded split counts cover all 455,226 frames; copy its metadata locally. First predictor **27682123** started; benchmark results remain pending.
- [x] Copy launch receipt and smoke JSON locally; sync outputs/logs at **18:35:42 UTC** (1,653 output files, 2.30 GB; 126 logs). Dataset and shared feature caches stay remote. Full job IDs follow below.
- [ ] After completion, recompute official metrics, audit contributions and compare paired seeds/movie bootstrap intervals. Smoke fitting is not a benchmark gain.

The run matrix, hyperparameters, interpretation limits and resource ceilings are in [trajectory_review.md](trajectory_review.md). `scripts/submit_trajectories.py` deduplicates jobs by source hash. Window-smoothing controls and independent temporal annotation remain future extensions. Trainable local decoders are fitted separately under each pooling rule; the frozen visual encoder, decoder architecture and common initialization are matched.

Resource check before this launch: **87,692:32 SBU** left for dispatch/submission, no active/queued jobs before smoke, home usage 40.5693% of 200 GiB and scratch usage 0.1394% of 8 TiB. `budget-overview`/`myquota` require the cluster login-shell environment. The queued-start estimates favored a five-minute H100 smoke over MIG/A100; job `27680963` actually started at 17:26:06 UTC. The full predictor requests remain on MIG and the full cache on A100.

### Full trajectory job IDs

Latest snapshot above: cache complete; one predictor running, one waiting for resources and 46 pending on throttle dependencies. The [machine-readable launch receipt](trajectory_full_launch.json) preserves the earlier submission snapshot.

| Variant | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|
| `traj_vad_duration` | 27682123 | 27682152 | 27682170 |
| `traj_vad_peak` | 27682125 | 27682153 | 27682171 |
| `traj_vad_class_peak` | 27682126 | 27682154 | 27682172 |
| `traj_vad_soft` | 27682128 | 27682155 | 27682175 |
| `traj_vad_sparse` | 27682131 | 27682156 | 27682176 |
| `traj_vad_power` | 27682132 | 27682157 | 27682177 |
| `traj_vad_sparse_proximity` | 27682134 | 27682158 | 27682179 |
| `traj_vad_sparse_permuted` | 27682137 | 27682159 | 27682180 |
| `traj_vad_duration_rare` | 27682139 | 27682160 | 27682181 |
| `traj_vad_sparse_rare` | 27682141 | 27682161 | 27682182 |
| `traj_vad_duration_importance` | 27682142 | 27682162 | 27682183 |
| `traj_vad_sparse_importance` | 27682145 | 27682163 | 27682184 |
| `traj_free_soft` | 27682146 | 27682164 | 27682185 |
| `traj_free_sparse` | 27682148 | 27682165 | 27682186 |
| `traj_vad_sparse_no_time` | 27682149 | 27682168 | 27682187 |
| `dino_meanpool` | 27682150 | 27682169 | 27682189 |

Cluster outputs: `outputs/experiments/<variant>_s<seed>_<jobid>/`; logs: `logs/experiments/<variant>_s<seed>_<jobid>.log`. Smoke: `outputs/smoke/smoke_trajectory_3videos_27681517/`. Every run stores config, source identity, checkpoints and metrics; trajectory runs additionally save scene contributions.

## Status update procedure

1. Refresh Snellius `squeue`, `sacct` for recorded jobs, and the shared JSON registry. Compare scheduler state with per-run artifacts before marking completion.
2. Update this file's timestamp, checkboxes, job table, failures/blockers, and next proposed batch. Keep failed attempts and replacement job IDs visible.
3. Report what is implemented, submitted, running, completed, failed, and still pending. Record unavailable checks explicitly rather than treating stale state as current.
4. Follow the authorized sequence with review, smoke and live resource checks. Preserve each batch's declared seeds (earlier batches: 42; trajectory batch: 42/43/44), official splits, validation-based selection and matched controls. Use uv on the cluster; local conda `torch` is for mock tests.

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
