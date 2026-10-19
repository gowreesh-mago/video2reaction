# Interactive trajectories for ten videos

[Open the standalone Plotly explorer](../../outputs/analysis/2026-10-19-interactive-trajectories/video_trajectories_10.html)
· [Download its numerical data](../../outputs/analysis/2026-10-19-interactive-trajectories/trajectory_examples_10.json)

The HTML works offline in a desktop browser. It embeds all ten examples and Plotly. No model server, notebook or raw video download is needed.

## What to try

1. Select a video and press **Play**, or drag the scene slider. Rotate and zoom the three-dimensional view; the camera stays in place as scenes change.
2. Turn on **Compare all 10 paths** to see the other trajectories in the same VAD coordinates. Hover for clip identity; click a point to select that video and scene.
3. Choose a reaction, or click a reaction marker or bar. The dashed line and lower plot show its proximity to the current scene. The table shows the six closest reactions.
4. Compare the whole-video audience distribution with the VAD prediction and the current scene. Switch the orange bars to **Mixture through this scene** to see the normalized accumulation of saved contributions. The DINO baseline is available through the legend.
5. Choose the common reference: training mean, mean of these ten examples, or mean across all 2,070 test videos. Cross markers show that reference in the distribution plot. Reaction marker sizes can use either the selected video's audience labels or this reference.
6. Click the timeline bars to investigate influential scenes. Their width is scene duration, height is mixture weight per second, and area is the actual scene weight.

## Data and limits

- Existing run: `on_vad_soft_tau010_s42_27685259`, the best previously selected VAD condition, seed 42. DINO comparison: `dino_meanpool_s42` from the prior audited export.
- Exactly **10 videos and 374 scenes**. Selection uses three lowest VAD errors, two nearest the median, three highest errors, and two remaining videos with the highest rare-reaction target mass. Error is KL divergence, where lower is better. Only clips with at least two scenes qualify; ties use sample ID. These examples are deliberately diagnostic, not a random performance sample.
- VAD means valence (pleasantness), arousal (activation), and dominance (control). All coordinates use the model's actual three axes on a −1 to +1 scale, rather than a PCA projection.
- Scene probabilities use `softmax(−squared_distance / temperature)`, with the saved temperature `0.09071478247642517`. The final distribution is the sum of scene probabilities times the saved scene mixture weights.
- The Transformer uses full-video context. Playback and the running mixture explain a completed prediction; they are not online inference. Connecting lines show scene order, not measured smooth motion. There is no scene-level emotion ground truth.
- Audience targets are the dataset's video-level reaction distributions. Common references are averages of per-video distributions, not pooled comment frequencies.
- The compact numerical export is 554,864 bytes. No raw videos, keyframes or complete feature tensors were copied. The private output contains the 21 NRC VAD v2.1 coordinates used by the run; preserve the source licensing recorded in its JSON. Generated assets stay outside Git.

## Validation and reproduction

The exporter verifies saved source hashes, sample and class alignment, scene chronology, probability normalization, the distance decoder, and reconstruction of each saved video prediction. Six regression tests cover selection, class-order/temperature errors, incorrect mixtures and overlapping scene intervals.

Browser checks exercise all ten videos, final cumulative mixtures, playback, timeline and table clicks, camera preservation, common references, downloads and desktop/mobile layout. The browser runs offline with an isolated profile. Its result and screenshots are in the output directory's `qa/` folder.

Export on Snellius, using the existing uv environment:

```bash
~/.local/bin/uv run --frozen --no-sync python scripts/export_trajectory_examples.py \
  --root /scratch-shared/gmago/video2reaction \
  --analysis /scratch-shared/gmago/video2reaction/reports/2026-10-19-error-analysis \
  --out /scratch-shared/gmago/video2reaction/reports/2026-10-19-interactive-trajectories/trajectory_examples_10.json
```

After copying that compact JSON, build locally with NumPy and Plotly:

```bash
python scripts/build_trajectory_explorer.py \
  --input outputs/analysis/2026-10-19-interactive-trajectories/trajectory_examples_10.json \
  --output outputs/analysis/2026-10-19-interactive-trajectories/video_trajectories_10.html
python -m pytest tests/test_trajectory_explorer.py -q
```

Optional browser verification (requires Playwright and Chrome):

```bash
python scripts/qa_trajectory_explorer.py \
  --html outputs/analysis/2026-10-19-interactive-trajectories/video_trajectories_10.html \
  --browser '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' \
  --out outputs/analysis/2026-10-19-interactive-trajectories/qa
```

Source branch: `research/interactive-trajectories-10videos`. No new model training or predictor jobs are part of this request.
