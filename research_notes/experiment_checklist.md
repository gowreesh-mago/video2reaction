# Experiment checklist

Last cluster status check: **2026-09-30 00:42:39 UTC**. Implementation status updated after the code/hypothesis review; fresh launch status will be recorded below.

## Current status

- **Completed:** repository/dataset/metric audits, cluster uv setup, and the three-video GPU smoke test.
- **Full benchmark experiments:** zero submitted, zero completed. B0/B1/B2/A5 plus the matched B2 set-transformer control are authorized as the first batch after review.
- **Snellius queue:** empty at the check above. The shared registry contains only the completed smoke run.
- **Launch control:** the user requested code/adversarial review, then launch. Complete the cluster regression smoke, then submit the first batch. Later batches remain pending a new request.
- **Branch:** `research/video2reaction-experiments`. [Code and hypothesis review](code_and_hypothesis_review.md)

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
- [x] Keep Video2Reaction data on the cluster; collect only aggregate audit/results files on the Mac.

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

Tested source revision: `08b8c30`. No full validation/test benchmark scores are available yet.

## 3. Work required before the first full batch

- [x] Implement the full training/evaluation entrypoint, including validation checkpoint selection, resume, and final test evaluation.
- [ ] Build and validate the reusable frozen feature cache for the official splits on the cluster; only smoke features exist currently.
- [x] Add one reproducible config and independent SLURM job per selected experiment, including hypothesis and changed component.
- [x] Add bounded batch submission (`scripts/submit_all.sh`) and result collection (`scripts/collect_results.py`); `scripts/status_all.sh` already exists.
- [ ] Save per-run config, code/model/split provenance, checkpoint where applicable, predictions with sample IDs/targets/top-k, metrics, and logs.
- [x] Verify exact interrupted/resumed training, terminal registry protection, source isolation, cache integrity, and matched controls in synthetic tests: 21 local tests pass.
- [ ] Pass the expanded test suite and three-video regression smoke in cluster uv.
- [x] Recheck authorized partition/account, queue, budget, and storage. `sbatch --test-only` is also required for every job before submission.

## 4. First proposed batch — Tier 0

All five full runs are **pending**. No job IDs have been assigned. Feature extraction is a separate prerequisite job.

| Experiment | Question / change | Implementation readiness | Submitted | Completed |
|---|---|---|---|---|
| B0 — Dataset prior | Predict the mean training distribution | Implemented; cluster regression pending | No | No |
| B1 — Frozen visual mean pooling | Establish a learned visual baseline | Implemented; cluster regression/cache pending | No | No |
| B2 — Temporal transformer | Test chronology against matched set transformer | Implemented; cluster regression/cache pending | No | No |
| B2 set control | Same transformer without positional encoding | Implemented; invariance/initialization tested locally | No | No |
| A5 — KL + cosine + ranking | Test distribution shape and reaction ordering | Implemented; cluster regression/cache pending | No | No |

- [x] User requests review, then launch this batch.
- [ ] Finish and validate the prerequisites above, then commit, push, and synchronize the selected code.
- [ ] Submit B0/B1/B2/set/A5 and record each job ID; make learned predictors depend on the validated feature cache.
- [ ] Check startup, completion/failure, predictions, metrics, and output paths for every run.
- [ ] Summarize the results here before selecting a later batch.

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

- [ ] Collect official distribution/ranking/Top-k metrics for each completed full run.
- [ ] Collect per-class precision, recall, F1, target support, and prediction support.
- [ ] Stratify results by target entropy quartile, frame count, and dominant target probability.
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
```
