# Experiment checklist

Last cluster status check: **2026-09-30 06:37:41 UTC**.

## Current status

- **Completed:** repository/dataset/metric audits, cluster uv setup, and the three-video GPU smoke test.
- **Full benchmark experiments:** B0, B1, B2, B2 set control, and A5 all completed successfully. The all-keyframe feature cache is complete. [Tier 0 results and interpretation](tier0_report.md)
- **Snellius queue:** no jobs remain in the queue. Corrected feature job `27382293` and learned jobs `27382294`–`27382297` all exited `0:0`; no new failures.
- **Results:** B1 has the lowest validation and test KL (test **0.544414**, versus B0 **0.689281**, a **21.0%** reduction). B2 has the highest MRR/F1@1; A5 has the highest cosine/F1@3. Small differences between learned models require cautious interpretation.
- **Tools and artifact sync:** CodeGraph v1.6.1 is installed on Snellius. Completed outputs, logs, and results, including the uncertainty analysis, were pulled to the Mac at **06:47:21 UTC** (205 files, 72.12 MB); raw data and feature tensors remain on the cluster. Repeat with `bash scripts/sync_outputs.sh`.
- **Launch control:** the user requested review, then launch; the first batch is submitted. Later batches remain pending a new request.
- **Branch:** `research/video2reaction-experiments`; current tested/submitted source **`bbf8c1d`**; completed B0 source **`e91a469`**. [Code and hypothesis review](code_and_hypothesis_review.md)

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

## 5. Later batches — select after reviewing Tier 0

Every item below is unlaunched. Grouping is a proposed order, not a submission request.

### Tier 1 and VAD comparisons

- [ ] C — Select a practical pretrained image-emotion model; compare visual, emotion, VAD, and fused evidence. Model/features pending.
- [ ] B-VAD1 — Add expected-VAD auxiliary regression. Coordinates are available; training experiment pending.
- [ ] B-VAD2 — Add geometry regularization. Coordinates are available; training experiment pending.
- [ ] D — Compare peak selection at K=1/2/4/8/all, using arousal, distance from neutral, and confidence-aware intensity. Depends on frame-emotion evidence.
- [ ] E — Run reaction-specific frame attention on the full benchmark. Query model is smoke-tested only.
- [ ] F — Compare global context plus selected peaks with each component alone. Depends on global and peak baselines.

### Required description diagnostics

- [ ] Evaluate description-only prediction.
- [ ] Evaluate visual + description prediction against the visual-only control.
- [ ] Report text-assisted and video-only results separately, including the observed cross-split movie overlap.

### Additional controlled comparisons and optional Tier 2

- [ ] Decide whether full A1–A4 loss comparisons add useful evidence after A5. CE/KL gradient equivalence is already unit-tested; loss functions are implemented.
- [ ] G — Reaction text prototypes.
- [ ] H — Retrieval from training clips only, with leakage checks.
- [ ] I — Hierarchical reaction prediction.
- [ ] J — Entropy-aware auxiliary prediction.

## 6. Evaluation and final report

- [x] Collect official distribution/ranking/Top-k metrics for all five completed Tier 0 runs.
- [x] Collect Tier 0 per-class precision, recall, F1, target support, and prediction support.
- [x] Save Tier 0 diagnostics by target entropy quartile, frame count, and dominant target probability, with cut points derived from training.
- [ ] For peak/query experiments, save selected indices/timestamps, emotion/VAD scores, and attention weights as applicable.
- [ ] Save 20 qualitative examples: 5 improvements, 5 failures, 5 high-entropy, and 5 low-entropy clips.
- [ ] Write `research_notes/final_report.md` with results, supported/rejected/inconclusive hypotheses, failures, benchmark caveats, and reproduction commands.
- [ ] Answer whether peaks, global context, or both help, and whether VAD improves prediction, using completed controlled comparisons.

## Status update procedure

1. Refresh Snellius `squeue`, `sacct` for recorded jobs, and the shared JSON registry. Compare scheduler state with per-run artifacts before marking completion.
2. Update this file's timestamp, checkboxes, job table, failures/blockers, and next proposed batch. Keep failed attempts and replacement job IDs visible.
3. Report what is implemented, submitted, running, completed, failed, and still pending. Record unavailable checks explicitly rather than treating stale state as current.
4. Launch only the batch requested by the user. Keep one seed (42), fixed official splits, validation-based selection, and matched controls. Use uv on the cluster; local conda `torch` is for mock tests.

Cluster root: `/scratch-shared/gmago/video2reaction`.

Initial release / completed B0: `releases/e91a469f51ba-b53672620486`. Corrected release / active jobs: `releases/bbf8c1d019a4-9a6a80a3d307`. Source hash suffixes are prefixes; full hashes are saved in each run's code manifest. Frozen releases remain unchanged while the working-copy checklist is updated.

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
