# Emotion timelines: ten best and ten worst VAD cases

[Open the offline explorer](../../outputs/analysis/2026-10-19-best-worst-trajectories/video_trajectories_20.html)
· [Numerical data](../../outputs/analysis/2026-10-19-best-worst-trajectories/trajectory_examples_20.json)

## Selection

These are the **10 lowest-KL and 10 highest-KL videos** across all 2,070 test videos for the existing `on_vad_soft_tau010_s42_27685259` run. Rank 1 is the lowest error. The selector separates the two groups and displays each video's KL. Ties use video ID. Selection was independently checked against the complete prior numerical export.

- Lowest ten KL values: **0.2267–0.2897**.
- Highest ten KL values: **2.2789–3.0830**.
- **20 distinct videos, 619 scenes and 619 source keyframe thumbnails.**
- Ranking uses the VAD model. The DINO baseline remains available for comparison. Selecting extremes does not estimate overall test performance.

The earlier ten-example viewer remains available separately. Its mixed selection included median-error and rare-reaction examples; this new viewer follows the requested best/worst ranking.

## Reading the plots

The first panel shows **all 21 emotion values for every scene** over video time:

- **Stacked scenes:** a column contains the complete scene distribution, totaling 100%. Column width is the recorded scene duration. Hover identifies the emotion and its value; the 21 value boxes below expose even very small probabilities.
- **Emotion × time heatmap:** one row per emotion, with the same scene values and true interval boundaries. Unobserved time gaps remain blank. Colours use an absolute 0–100% probability scale.
- **Contribution to video prediction:** switches the values to saved scene weight multiplied by scene emotion probability. Summing each emotion's contribution across scenes reconstructs its final video probability. The stacked column height is then the scene weight, rather than 100%.
- Clicking a column/cell or value selects that scene and emotion in the frame cards, 3D path and proximity plot.

Directly underneath, grouped bars compare the **whole-video audience labels** and the **final VAD prediction** across the same 21 emotions. The DINO prediction is available through the legend. Audience labels describe the whole video; they do not provide a ground-truth distribution for each scene.

The earlier controls remain: playback, the focused edge from t−1 to t, faded background paths, an emotions-only view, previous/current keyframes, image enlargement, common reaction references and the influence timeline. Keyframe captions use recorded scene intervals because exact capture instants are unavailable.

The timeline visualizes saved model probabilities and temporal contributions. It does not compute pixel saliency or establish which scene caused a viewer's reaction. The Transformer used full-video context; playback is an explanation of its completed prediction.

## Comments and annotations

The repository describes labels derived from YouTube viewer comments and includes comment retrieval and automatic annotation code. The inspected train/validation/test metadata contains clip metadata plus `reaction_outcome.dominant_reaction` and `reaction_outcome.reaction_distribution`. It does not contain original comment text or comment-level emotion annotations.

The project dataset directories and original extracted tree on Snellius contain metadata and keyframes; no raw-comment file was found there. The archive directory contains the keyframe archive. Prompt examples in annotation code are not dataset comments for these videos and are not displayed as such. A path to the original comment-level annotations has been requested. Their absence does not block this viewer.

## Verification

- Seven numerical regression checks pass, including selection over the full test set and preservation of single-scene cases.
- Every exported scene probability matches the saved VAD distance decoder; saved weighted contributions reconstruct the final video prediction.
- Frame export checks video/scene identity and intervals against the original keyframe index, and records original-image and thumbnail hashes.
- Offline browser checks verify every selected video, all 619 images, exact timeline values and scene widths, heatmap boundaries, hover/click synchronization, cumulative mixtures, audience/prediction bars, playback, camera preservation and desktop/mobile layout.

Source: `research/emotion-timelines-best-worst`. No new predictor training or evaluation is required for these views. The model data and source videos remain on Snellius; only the compact selected exports and thumbnails are copied to the Mac.

## Reproduce

On Snellius, using the existing uv environment and this branch's scripts:

```bash
~/.local/bin/uv run --frozen --no-sync python scripts/export_trajectory_examples.py \
  --root /scratch-shared/gmago/video2reaction \
  --analysis /scratch-shared/gmago/video2reaction/reports/2026-10-19-error-analysis \
  --selection kl_extremes \
  --out /scratch-shared/gmago/video2reaction/reports/2026-10-19-best-worst-trajectories/trajectory_examples_20.json
~/.local/bin/uv run --frozen --no-sync python scripts/export_trajectory_frames.py \
  --trajectory /scratch-shared/gmago/video2reaction/reports/2026-10-19-best-worst-trajectories/trajectory_examples_20.json \
  --keyframes /scratch-shared/gmago/video2reaction/data/key_frames \
  --out /scratch-shared/gmago/video2reaction/reports/2026-10-19-best-worst-trajectories/trajectory_frames_20.json
```

After copying those selected exports, build locally:

```bash
python scripts/build_trajectory_explorer.py \
  --input outputs/analysis/2026-10-19-best-worst-trajectories/trajectory_examples_20.json \
  --frames outputs/analysis/2026-10-19-best-worst-trajectories/trajectory_frames_20.json \
  --output outputs/analysis/2026-10-19-best-worst-trajectories/video_trajectories_20.html
python -m pytest tests/test_trajectory_explorer.py -q
```
