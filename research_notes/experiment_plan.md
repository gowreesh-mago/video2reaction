# Experiment plan

One seed (42), original splits, no test-based tuning. This is a two-day research run, with infrastructure and viable results first. All changes live on `research/video2reaction-experiments`.

## Execution order

1. Verify accounts/partitions, budget, dataset paths and completeness; audit official metrics.
2. Lock Python 3.11/PyTorch with uv. Run import, GPU, metric, registry concurrency, and tiny training checks on the cluster.
3. Run B0 train prior; prepare all frozen keyframe features once; launch B1 mean pool, B2 temporal transformer, its matched set-transformer control, and A5 KL+cosine+ranking.
4. Prepare a practical pretrained image-emotion model and sourced VAD coordinates; then C emotion/VAD fusion, D peak K=1/2/4/8/all with three intensity definitions, E reaction queries, F global+peak, and B-VAD auxiliary/geometry comparisons.
5. Run description-only and visual+description diagnostics separately from video-only results. Add prototype, retrieval, hierarchy, entropy experiments as resources allow.
6. Collect official metrics, per-class and stratified diagnostics, attention/selected frames, and 20 qualitative examples. Report what completed, failed, or remains blocked; answer peak/context and VAD questions only with controlled evidence.

## Controls

Use a common frozen backbone, all indexed keyframes, identical frame preprocessing/order, AdamW, batch size, epoch limit, validation early stopping and head for aggregation/loss comparisons. Validation KL selects checkpoints; test predictions are produced only after selection. B2 versus its identical transformer without positions tests the effect of frame-position information; B2 versus B1 also changes capacity. CE and forward KL have identical gradients for fixed normalized targets, so they are a numerical sanity comparison rather than distinct scientific hypotheses. Persist config, code revision/hash, split hashes, model revision, checkpoint/resume state, predictions, metrics, and SLURM logs. See the [adversarial review](code_and_hypothesis_review.md) for claim limits and controls required for later batches.

## Cluster policy

Only submit to verified accounts and partitions. Keep one registry on Snellius; avoid unsynchronized registries across clusters. First use a small GPU smoke job, then a bounded batch of independent experiments with dependencies. Check queue and budget before expanding. Never overwrite existing datasets/checkpoints or synchronize with `--delete`.

## Progress and remaining deliverables

Track completed work, job IDs, and pending batches in the [experiment checklist](experiment_checklist.md). The reviewed runner/configs/jobs are implemented. All 22 mock tests pass locally and on the cluster; the three-video shared-loader smoke passes on the cluster. B0 completed. Corrected feature extraction is processing frames, and four learned comparisons are submitted with a cache dependency. Later hypotheses, paired uncertainty analysis, qualitative examples, and the final report remain pending.
