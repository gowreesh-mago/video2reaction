# Joint highlights and pretrained highlight pseudo-labels

Added on 2026-10-05 following the user's request. The 41 completed benchmark results predate these additions. These are new hypotheses, not reinterpretations of D's fixed emotion scores.

## Experiments

| Name | What changes | Main comparison |
|---|---|---|
| `joint_highlight_sparse` | A frame scorer is trained jointly with reaction KL; sparsemax gives exact zero weights to discarded frames. The reaction head receives only the weighted selected-frame representation. | `joint_highlight_soft_control`, then B1 |
| `joint_highlight_soft_control` | Same scorer, head, parameter count, initialization and optimizer; softmax replaces sparsemax. | Isolates the pooling normalization and resulting sparsity |
| `pretrained_dsnet_k4` | Frozen DSNet TVSum split-0 scores select four frames before the unchanged B1 projection/head. | Existing uniform K=4, random K=4, emotion-score K=4, and all-frame B1 |
| `highlight_cache` | Extract aligned DSNet pseudo-labels once on Snellius. | Infrastructure only, no benchmark score |

All use seed 42, the official splits, fixed SigLIP2 visual features, forward KL, and the existing validation-KL checkpoint selection. No reaction descriptions, target labels, or test results are passed to either highlight selector at inference. K=4 and the single external checkpoint are fixed before evaluating these new models.

## Joint model

The B1 projection produces `h_t`. A `128 -> 32 -> 1` scorer with tanh computes `s_t`. Pooling uses `a = sparsemax(s)` over valid frames, then `h = sum_t a_t h_t`; the unchanged head predicts the 21 audience-reaction probabilities. The scorer and predictor are updated together by reaction KL. There is no global bypass and no external emotion-score ranking. The scorer adds 4,160 parameters; both new learned selectors have 173,269 parameters.

Sparsemax is the Euclidean projection onto the probability simplex and supports backpropagation. [Martins and Astudillo, ICML 2016](https://proceedings.mlr.press/v48/martins16.html). Its support size is adaptive, not fixed K. Temperature is fixed at 1. The matched softmax model uses the exact same parameters and initialization. Both inherit B1's shared projection/head initialization.

Existing E already learns reaction-specific soft attention jointly with reaction prediction. Its shared-query control averages 21 query maps. The new model instead has one explicit shared frame scorer and a sparse selected set.

Exports include all frame weights, positive-support positions, original frame identities/timestamps, support counts, effective frame counts, entropy, and the fraction of one-frame/all-frame supports. These weights are exactly those consumed by the reaction predictor.

## Frozen pretrained teacher

[DSNet](https://github.com/li-plus/DSNet) releases a query-free video summarization model and pretrained TVSum/SumMe checkpoints. We pin source `1804176e2e8b57846beb063667448982273fca89`, choose **TVSum split 0** without a checkpoint sweep, and verify the source, checkpoint, and GoogLeNet weights by SHA-256. The assets are recorded in [highlight_model.json](../assets/highlight_model.json). We load only four reviewed inference modules, retain the upstream MIT license, and load tensor checkpoints with `weights_only=True`.

The image path follows the author's `video_helper.py`: RGB, resize 256, center crop 224, ImageNet normalization, sequential GoogLeNet pool5, and L2 normalization. The teacher's per-position score is classification probability times centerness, divided by its clip maximum plus epsilon. Four highest scores select SigLIP2 frames; ties choose the earlier frame, and output order remains chronological. The frozen teacher never receives audience-reaction targets. Cached scores have exact source-frame/image identities and checksums; interrupted extraction resumes from verified completed clips.

**Adaptation:** this applies a video-summary teacher to irregular scene keyframes. It uses DSNet's frame-position confidence, not the original dense-video proposal/NMS/knapsack summary protocol. Predicted proposal boxes are retained as diagnostics in keyframe-position units, not seconds. Original timestamps are carried from the dataset. The training domain and sampling pattern differ from Video2Reaction; the scores are pseudo-labels, not human highlight ground truth. Other candidates such as query-dependent moment retrieval require additional query/temporal inputs; DSNet is a practical first external teacher, not a claim that it is the best highlight detector.

## Adversarial review

- **Does joint training actually reach the selector?** Mock tests verify nonzero, finite reaction-loss gradients in its scorer, projection and classifier. A manual reconstruction checks that logits come only from the selected weighted representation.
- **Can a gain be explained by extra capacity?** Compare sparsemax against the identical softmax scorer, not B1 alone. For frozen DSNet, the reaction model is exactly B1; compare equal K to separate selection quality from frame count.
- **Would a KL gain prove correct highlight localization?** No. There are no human highlight labels here. It would show predictive value for the learned selection under this split. Human localization accuracy and causal faithfulness remain unmeasured.
- **Could sparsemax collapse?** Yes. A one-frame support has zero local score gradient, while diffuse supports may retain every frame. Report support statistics; do not silently change the temperature after seeing test scores.
- **Could pretrained scores be meaningless on scene keyframes?** Yes. Save score ranges/ties and selections, use the equal-K controls, and treat a negative result as evidence about this particular transfer.
- **Could this prove narrative understanding or generalization to unseen movies?** No. The joint scorer pools a set; movie overlap remains. Keep the unseen-movie diagnostic and avoid narrative claims.
- **Can outputs be trusted after the registry discrepancy?** Reconcile every new job with SLURM and per-run artifacts. One older stale registry entry was repaired with evidence; its unresolved cause calls for a cross-node visibility audit before a large concurrent batch. This small batch is capped at two jobs.

## Validation and execution

- **80 local mock tests pass**, including 10 new highlight tests. Local mock checks cover sparsemax's known projection, gradients, shift invariance, masks, padding, permutation invariance, matched controls, fitting/reload, exact saved highlights, stable pretrained selection, interrupted-cache recovery and corruption rejection.
- `smoke_highlights_3videos` uses the same three training videos and eight frames each. It runs B1, both joint selectors, and frozen DSNet K=4; smoke scores are never benchmark results.
- Push/sync/freeze before cluster execution; run the published-model setup on Snellius and the smoke in the locked uv environment.
- Full jobs: `joint_highlight_sparse`, `joint_highlight_soft_control`, and `pretrained_dsnet_k4` (the latter depends on `highlight_cache`). Launch helper: `scripts/submit_highlights.py --completed-smoke-job ID`.

Current execution state is maintained in [the checklist](experiment_checklist.md).
