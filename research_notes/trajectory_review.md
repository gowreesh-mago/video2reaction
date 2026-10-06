# Visual-only VAD trajectory experiments

Implementation and adversarial review, October 6, 2026. Full runs require the exact committed source to pass the three-video cluster smoke. Submission status and IDs are maintained in `experiment_checklist.md`.

## Hypothesis and implemented path

The user proposes that most audience reactions concern a small subset of visual moments. A video's VAD trajectory can support different reactions at different moments; duration, proximity and relevance determine each emotion's final probability.

The implementation is `RGB scene keyframes -> frozen DINOv2 CLS features -> timestamp-aware visual transformer -> per-scene VAD and relevance -> probability aggregation`. The fixed 21 NRC v2.1 coordinates define the VAD decoder. No captions, supplied descriptions, text embeddings, language teachers or comment text enter this path. Existing audience reaction distributions supply training supervision.

DINOv2 base is pinned to `f9e44c814b77203eaa57a6bdbbd535f21ede1415`; its visual self-supervision meets the strict visual-only backbone requirement. Model files remain in the existing cluster Hugging Face cache. The new feature namespace is `data/features/dinov2-base/`; it does not reuse SigLIP features. See the [DINOv2 source](https://github.com/facebookresearch/dinov2) and [pinned model](https://huggingface.co/facebook/dinov2-base/tree/f9e44c814b77203eaa57a6bdbbd535f21ede1415).

NRC v2.1 has exact entries for all 21 official labels. The asset uses [-1,1], keeps the benchmark class order and is distinct from v1. Archive SHA256: `8bcd04831ffda149f683f8d3091c76a3aee138ee3aa1924b67b50b9416d37df0`. Generated private 21-label asset SHA256: `fb2ce00a38d37e6e8b41418a09cb374ad96b51bc9c24c32f0a51bb790a351906`. Provisioning is reproducible with `scripts/setup_vad_v2.py`; coordinates and the archive are excluded from Git.

For scene i, the learned point is `z_i = tanh(visual_head(h_i))`. Fixed label point `e_c` gives `log k_ic = -||z_i-e_c||²/tau`; `q_ic = softmax_c(log k_ic)`. Tau is one learned scalar bounded to (0.05,2), initialized at 0.25. No free class bias or residual classifier bypasses the VAD decoder.

The shared visual relevance scorer is a 128 -> 32 -> 1 network; its scores are divided by a fixed 0.2 in both soft and sparse arms. Let `mu_i = duration_i / total_observed_duration`. Soft relevance uses `w_i = softmax_i(score_i + log(mu_i))`. Sparse relevance finds a threshold t satisfying `sum_i mu_i * max(score_i-t,0) = 1`, and sets `w_i = mu_i * max(score_i-t,0)`. This weighted sparsemax construction treats relevance as a density over seconds. Ordinary sparsemax over a changing number of scene samples would not preserve interval-splitting invariance.

The primary output is `P(c) = sum_i w_i*q_ic`. Computation uses log-sum-exp. All moment contributions are saved and checked against the final distribution. The sparse-proximity alternative integrates `w_i*k_ic` and normalizes across emotions afterwards; it is intentionally a different model.

## Predeclared batch

Every row runs seeds 42, 43 and 44: **16 variants, 48 predictor runs**, plus one full DINOv2 cache and one three-video smoke. Main comparisons share the frozen visual features, projection/temporal architecture, initialization of shared parameters, optimizer, natural official splits, KL objective and validation selection. Trainable momentary decoders are fitted separately in each end-to-end run. These comparisons test the complete learning rule; they are not a post-hoc pooling comparison with identical fitted decoder weights.

| Variant | Change / question |
|---|---|
| `traj_vad_duration` | Sum scene probability times actual duration |
| `traj_vad_peak` | One scene: closest approach to any fixed emotion prototype; earliest tie |
| `traj_vad_class_peak` | Maximum raw proximity for each emotion, then normalize across classes |
| `traj_vad_soft` | Learned soft relevance density times duration |
| `traj_vad_sparse` | Learned sparse relevance density times duration |
| `traj_vad_power` | Duration-aware power pooling, one shared learned exponent in (0,8), initialized at 1 |
| `traj_vad_sparse_proximity` | Integrate raw VAD affinity under sparse relevance |
| `traj_vad_sparse_permuted` | Fixed permutation of coordinate-to-label assignments, seed 271828 |
| `traj_vad_duration_rare` | Duration arm with training soft-mass rarity sampling |
| `traj_vad_sparse_rare` | Sparse arm with the same rarity sampling |
| `traj_vad_duration_importance` | Same sampler, importance-corrected natural KL objective |
| `traj_vad_sparse_importance` | Same corrected-sampling control with sparse relevance |
| `traj_free_soft` | Unrestricted 21-logit momentary decoder, matched soft relevance |
| `traj_free_sparse` | Unrestricted 21-logit momentary decoder, matched sparse relevance |
| `traj_vad_sparse_no_time` | Remove timestamp encoding, retain visual contextualization and scene durations |
| `dino_meanpool` | DINOv2 version of the existing feature-mean MLP baseline |

The unrestricted decoder has 21 outputs instead of 3 and no VAD temperature; the resulting small parameter difference is reported. The ordinary DINO mean-pool baseline lacks the trajectory transformer's additional capacity, so it is a backbone reference rather than the isolation test for pooling. Peak/duration modes do not train the otherwise shared relevance scorer, because it does not enter their prediction. Per-class peak pooling preserves closest approach but deliberately discards relative dwell time. Power pooling can similarly suppress duration effects; it is a comparison, not a duration accumulator.

Rarity is defined exclusively by training probability mass, not dominant labels. Each clip's sampling weight is `sum_c target_ic / sqrt(train_prevalence_c)`, divided by mean weight and capped at 3 before normalization. Draw exactly N training clips with replacement each epoch. The importance arm multiplies per-clip KL by `1/(N * sampling_probability_i)`, preserving the natural objective in expectation. All arms retain the official soft targets. The two rare-sampling arms plus their natural counterparts form the requested duration/sparse by natural/rare comparison.

All predictors use batch 64, AdamW learning rate 0.001, weight decay 0.0001, gradient norm cap 1, at most 50 epochs, and patience 8 on natural validation KL with min delta 0.0001. Test labels never select parameters, temperatures, sampling weights, early stopping or checkpoints. Seeds are paired, not selected by test score.

## Adversarial review: do the changes actually test the hypothesis?

1. **Does duration enter the prediction?** Yes, through scene end minus start. Gaps are not filled by extrapolating a sampled keyframe. Duplicate frames with subdivided durations cannot multiply evidence at the pooling stage. Unit tests cover unequal durations, splitting, masking and sparse gradients.
2. **Can irrelevant moments contribute zero?** Yes, through the sparse density. The learned support can still be broad. Report support fraction and effective contributing moments; do not describe the fitted model as selecting a few moments without those measurements.
3. **Are probabilities decoded from fixed VAD points?** Yes. The primary path has no learned per-class bias, residual logits or text projection. The VAD points remain fixed buffers, and the final distribution is a mixture of decoded local distributions. The free-head controls explicitly remove this constraint.
4. **Could two opposed emotions disappear through averaging?** We aggregate distributions, not their VAD coordinates. Separate moments can support separate emotion modes.
5. **Could the temporal-order test accidentally change duration?** Initially shuffling images into fixed intervals would have changed their duration weights. Review corrected this: move scene features together with their duration, then concatenate the permuted intervals into a new timeline. A test with unequal durations verifies invariance of the timestamp-free model. This diagnostic is a perturbation, not a causal test of narrative comprehension.
6. **Does sparse attention prove actual comment attribution?** No. Supervision is at clip level; many trajectories and relevance maps may explain the same target. Full-video visual context can influence every local estimate. Sparse contribution weights do not mean other scenes were invisible to the encoder. Independent temporal annotations would be needed to substantiate attribution to real viewer comments.
7. **Are measured VAD trajectories available?** No. These are weakly supervised, piecewise-constant estimates from scene keyframes. Existing timestamps do not reveal unobserved within-scene dynamics or exact emotional dwell time. Do not claim continuous video emotion tracking from this dataset representation.
8. **Does rare-class improvement imply better distribution prediction?** No. Reweighting can increase rare-class recall while worsening population calibration and official KL. Save predicted/target class mass, Brier score, per-class probability MAE and macro diagnostics for the six lowest-mass training classes, alongside all official metrics. Zero-support classes are disclosed for recall.
9. **Would a negative result disprove the whole idea?** No. It constrains this visual representation, weak supervision, three-dimensional decoder and pooling family. Conversely, beating mean pooling alone does not establish semantic VAD or sparse selection: compare sparse versus soft, sourced versus permuted coordinates, and VAD versus unrestricted local heads.
10. **Are isolated noisy peaks controlled?** The batch exposes peak-only versus integrated alternatives, but has no dense-frame or fixed-second window smoothing arm. Those remain follow-up controls before interpreting a peak improvement as reliable localization.

## Validation and artifact contract

- Local tests use the user's conda `torch` environment and synthetic data only. The full existing suite and the new trajectory tests pass locally (108 tests before cluster smoke).
- Cluster execution uses the existing locked uv environment with `uv run --frozen --no-sync`; no conda is used there.
- Smoke uses exactly three official training videos, eight chronological keyframes each. It extracts new DINO features and fits every variant, checking finite normalized predictions, lower same-example KL, exact selected-checkpoint reload and a reproducible resumed optimizer update. These are infrastructure checks, not generalization scores.
- Full submission refuses a different source hash, an incomplete smoke, a missing variant, missing contribution exports or a non-successful smoke job. Every SLURM request is preflighted before the first full submission. A locked submission ledger and scheduler comments prevent duplicate jobs on retry.
- The full feature cache runs once on an A100, with an eight-hour ceiling. Predictors use a MIG slice, four CPUs and 24 GiB RAM, with a 45-minute ceiling each. Cache-success dependencies and two submission lanes bound concurrency. The maximum requested allocation is approximately 3,328 SBU excluding smoke; elapsed allocations determine actual usage.
- Code runs from immutable release directories. The dataset and shared visual features remain on Snellius. Logs, configs, source identities, checkpoints, metrics and prediction/trajectory artifacts have separate named output directories and can be synced to the Mac.
- Per-split `trajectory.npz` records interval VAD estimates, probabilities, raw log proximity, relevance scores/densities, base weights, durations and actual per-emotion contributions. `attention_frames.json` links them to exact source keyframes. `trajectory_summary.json` records support, concentration and checksums. For nonlinear pooling, the contribution formula is identified separately from the base temporal weights.

## Primary sources

- [NRC VAD v2](https://arxiv.org/html/2503.23547) and [official resource](https://saifmohammad.com/WebPages/nrc-vad.html): fixed label coordinates only.
- [Attention-based MIL](https://proceedings.mlr.press/v80/ilse18a.html): weakly supervised relevance weighting; our output is a categorical probability mixture.
- [Sparsemax](https://proceedings.mlr.press/v48/martins16.html): sparse attention. Duration-measure weighting is our adaptation.
- [Power pooling](https://arxiv.org/html/2010.09985): mean-to-peak aggregation for sound-event presence. Duration weighting and cross-class normalization are our categorical adaptation.
