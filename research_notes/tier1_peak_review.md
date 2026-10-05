# Emotion evidence, peaks, and context: implementation and adversarial review

**Result update, 2026-10-05:** all relevant benchmarks are completed and verified. See the [full results and interpretation](final_report.md) and [41-run JSON](recommended_results.json). Historical launch status below is retained as review history.

**Launch update, 2026-10-05:** GPU smoke `27611469` passed (61 tests, 18 variants, 2m53s). The primary C/D/F batch is submitted; the K=1/2/8 curve awaits cache completion. The review below records the pre-launch design; the [live checklist](experiment_checklist.md) supersedes its historical pending status.

Status on 2026-10-01: implemented locally; **37 synthetic tests pass**. These comparisons have not been run on the dataset. A three-video cluster smoke must pass before the full cache and benchmark batch.

## Scientific comparisons

All comparisons reuse the completed SigLIP2 cache, seed 42, official splits, optimizer, batch size, KL loss, epoch limit, and validation KL checkpoint selection from B1. The held-out test split must not select K, a score definition, or a fusion variant. The primary F comparison is fixed in advance to **arousal, K=4**.

| Family | Intervention | Required comparison | What a gain would support |
|---|---|---|---|
| C | Add eight coarse-emotion logits, expected VAD, or both through a linear projection | Three variants versus B1 and each other | The usefulness of this frozen model's extra frame evidence |
| D | Keep K=1/2/4/8 frames by arousal, distance, or confidence-weighted distance | Equal-K uniform and random subsets; B1 supplies all-frame pooling | Useful affect-based selection beyond subsampling alone |
| F | Concatenate global and arousal-K4 pools | Equally sized global+global and peak+peak controls | Complementary information from broad clip appearance and selected frames |

The D grid has 12 affect-ranked conditions and eight necessary equal-K controls. There is one shared all-frame B1 reference. We avoid duplicating the same all-frame condition for each score. C has three conditions and F has three. This is a targeted 26-condition design with named controls, using one seed, rather than a hyperparameter search.

### Scores and controls

The frozen predictor gives eight logits. Their untempered softmax is q. The expected VAD point is `q @ E`, where E contains exact NRC v1 entries in the model's class order. Coordinates were extracted from the same checksum-verified archive used for the 21 reaction labels.

The declared neutral reference is `(0.5, 0.5, 0.5)`, the midpoint of the [0,1] coordinate scales. It is a modeling choice, not a measured word coordinate. The three scores are:

1. Expected arousal A.
2. Euclidean distance from the declared reference.
3. Distance multiplied by `1 - H(q)/log(8)`.

Top-K ties favor the earlier original frame. Selected frames are returned in their original chronological order. Uniform selection takes the midpoint of K equal bins; K=1 is the middle frame. Random selection uses a deterministic hash of seed and video ID, sampling without replacement. It does not change across epochs or depend on sample order, target labels, or prediction quality. K at or above a clip's frame count selects every frame.

### Capacity and feature controls

The original visual projection is preserved. C adds a bias-free 11→128 projection; unused input groups are zeros. The visual-only path with zero extra inputs has exactly the original predictions and common-parameter gradients in the synthetic check. All three C models have the same declared parameter shapes, but different numbers of active input columns. The C comparison alone therefore does not isolate semantic geometry from dimensionality or conditioning.

D changes the frame subset before the unchanged B1 predictor. F applies the same frame projection before either pooling operation. Its three variants all concatenate two 128-dimensional pools and use the same wider head: 185,493 parameters versus B1's 169,109. The global+global and peak+peak controls match F's parameter count and initialization exactly. They use duplicated inputs deliberately. Equal parameter count still does not imply identical optimization difficulty.

Global mean pooling represents broad visual appearance. It does not model narrative order. A successful F result cannot by itself establish narrative understanding or identify the events that caused viewers' comments.

## Model provenance and limits

The [EmoEditor release](https://github.com/ZhangLab-DeepNeuroCogLab/EmoEditor) provides an EmoSet-trained ResNet-18 with an eight-class head. Its [inference code](https://github.com/ZhangLab-DeepNeuroCogLab/EmoEditor/blob/9d989ae99e43bfe533f9b7fdccfe49f6c2f23253/test.py) defines RGB images resized to 224×224 and tensors in [0,1], without ImageNet normalization. The adapter follows that convention. The [model manifest](../assets/frame_emotion_model.json) records the source revision, download URL, class order, and checkpoint SHA-256. Only tensor weights are loaded; downloaded Python is not executed.

The alphabetical checkpoint label order differs from the EmoSet dataset repository's order. Mixing those orders would silently corrupt VAD and peak scores; the VAD reader checks the exact order and asset checksum. The released model predicts image-evoked emotion categories, which are proxy evidence for this task. They are not the audience's 21 reaction labels or observed human frame annotations. Its confidence is uncalibrated on Video2Reaction, and it has no neutral class. A confident error can select an irrelevant frame; distance can also favor calm but strongly positive/negative content.

## Engineering review

- The emotion cache verifies every newly read image hash against the completed visual cache and checks the original frame-index hash. It stores eight float32 logits per frame, preserving the same offsets.
- Cache identity includes pinned model/preprocessing settings, the visual-cache specification, extraction implementation checksum, and uv lock checksum. Completed files are checksum-verified before reuse.
- Extraction uses a lock, durable progress records, and a checksum of the committed prefix. Restarting preserves completed rows and recomputes uncommitted rows. Corrupted committed data is rejected.
- Frame-emotion probabilities, VAD, all three scores, global/peak weights, selected positions, and original frame indices/timestamps are exported for validation/test. Selection never receives target labels.
- Evaluation strata continue to use original frame counts. Using K as the frame-count diagnostic would otherwise collapse all subset runs into an artificial count group.
- Frozen releases isolate queued jobs from local work. The existing cluster uv environment is reused without package changes. Model weights, raw frames, full split files, and visual/emotion caches remain on Snellius.

### Tests and remaining gate

All 37 local tests pass in conda `torch` (5.09 seconds). New tests cover interrupted-cache recovery, unchanged completed-cache reuse, image mismatch rejection, exact frame/diagnostic alignment, selection independence from targets, all three score equations, equal-K controls, K beyond clip length, zero-evidence baseline equivalence, and the fusion controls' responses to selected/unselected frames. Config tests verify all 26 conditions retain the baseline data/encoder/training/loss/evaluation settings. Scheduler tests distinguish required-cache `afterok` dependencies from four-lane `afterany` throttling.

The remaining cluster gate is `smoke_emotion_3videos`: three official training clips, eight frames each, both production cache builders, and 14 representative C/D/F variants using the production fitting/checkpoint/evidence-export code. Four additional B-VAD variants now share this smoke (18 variants total; see [VAD review](vad_review.md)). Its metrics are fitting checks on those training clips, never benchmark evidence. The actual pretrained adapter is not yet inference-validated.

## Predeclared interpretation

- A scored subset beating B1 but not equal-K uniform/random controls supports subsampling, not affect-based selection.
- A K=4/K=8 advantage over K=1 and all frames is suggestive only if the same advantage survives equal-K controls. An inverted-U alone is insufficient.
- F beating global+global and peak+peak supports complementary representations under this architecture. Report capacity-matched controls as the primary F comparison.
- Improvements from VAD inputs show predictive usefulness of those inputs. They do not prove that VAD geometry is uniquely responsible; the B-VAD auxiliary/geometry experiments and semantically permuted controls remain separate work.
- Negative results constrain this fixed predictor, feature representation, and scoring family. They do not reject every possible peak or context model.
- Use paired movie-cluster uncertainty for predeclared comparisons. One seed and testing many conditions limit claims. Show all results and validation-selected checkpoints, rather than reporting only the best test score.

## Cluster commands once SSH is available

After commit, push, and `bash scripts/sync_cluster.sh`, run on Snellius:

```bash
cd /scratch-shared/gmago/video2reaction/code
source configs/clusters/snellius.env
python3 scripts/setup_emotion_model.py
release=$(python3 scripts/freeze_release.py)
bash "$release/scripts/submit_smoke.sh" smoke_emotion_3videos
```

After verifying smoke success, scheduler access, queue, and remaining budget:

```bash
cd "$release"
python3 scripts/submit_emotion.py --phase primary
# Once emotion_cache completes successfully:
python3 scripts/submit_emotion.py --phase curve --completed-cache-job JOB_ID
```

The primary phase has one cache job and 11 predictors. Every predictor requires successful cache completion; four dependency lanes bound concurrency. The curve phase has the remaining 15 K conditions, also in four lanes. Each predictor has a 30-minute limit (32 SBU); the cache has a two-hour limit (128 SBU). These are upper bounds, not estimates: primary ≤480 SBU, curve ≤480 SBU, and the 15-minute smoke ≤16 SBU. Check actual resource availability before submission.
