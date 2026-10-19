# Previous-run error analysis — October 19, 2026

This analysis covers **308 completed prediction runs**, including the 264 DINO/VAD trajectory runs and the earlier baselines. It adds no new predictor training. The advertisement idea is excluded; VAD-to-DINO reconstruction and DINOv3 are not launched.

## Local results

- [PDF report](../../output/pdf/video2reaction_error_analysis.pdf)
- [Interactive video and label embeddings](../../outputs/analysis/2026-10-19-error-analysis/embeddings_3d.html)
- [Interactive saved VAD trajectories](../../outputs/analysis/2026-10-19-error-analysis/vad_trajectories_3d.html)
- [Full numerical analysis](../../outputs/analysis/2026-10-19-error-analysis/analysis.json)
- [Every previous run and all eleven metrics](../../outputs/analysis/2026-10-19-error-analysis/all_previous_runs.csv)
- [DINO failures, sorted by KL](../../outputs/analysis/2026-10-19-error-analysis/dino_meanpool_s42_per_video.csv)
- [DINO class errors and support](../../outputs/analysis/2026-10-19-error-analysis/dino_meanpool_s42_per_class.csv)

The analysis directory also contains matching tables for twelve other representative runs, plot sources, the LaTeX report source, and provenance/checksum files. The PDF bundle includes these results for portable review. These generated files are intentionally excluded from Git.

## Main findings

Detailed diagnostics use seed 42; the report separately gives the earlier three-seed averages.

1. **Dominant predictions cover only five classes for DINO.** Top-1 weighted F1 drops from 0.4846 overall to 0.1939 on the 215 videos whose dominant reaction is outside the three most common training reactions: amusement, disapproval and admiration.
2. **Predictions vary too little between videos.** DINO's median predicted-to-target class standard-deviation ratio is 0.231, and 0.156 among the six rare reactions. Matching average class frequencies does not imply predicting the right distribution for each video.
3. **VAD increases rare probability without locating it correctly.** Rare target mass is 0.01826. DINO predicts 0.01839, with rare-mass correlation 0.173; the best VAD condition predicts 0.08645 with correlation -0.005. Both have zero Top-3 recall on these six rare classes. They have no Top-1 target support, so a zero rare Top-1 score is uninformative.
4. **Saved VAD is not constant.** All three inspected runs have variation on every VAD axis and full-rank covariance. Every saved scene contribution reconstructs the video prediction. This checks the exports, not the correctness of the predicted emotional trajectory.
5. **Three-dimensional plots lose information.** The first three DINO principal components retain 21.7% of training variance; three label-distribution components retain 81.4%. Label-expected VAD is a lexicon centroid, not a measured emotion state.

The best VAD condition's original three-seed test KL is 0.835698 versus DINO's 0.580898. The representative seed-42 VAD-minus-DINO difference is +0.2885, with movie-bootstrap interval [0.2761, 0.3007]. This is a clear degradation for these models, not evidence that every possible VAD method must fail.

## Missing paper predictions

No per-video predictions from the paper's fine-tuned LLaVA/Qwen models were found in the inspected project outputs or original data tree. A location has been requested. Our `visual_description` result is explicitly a separate, text-assisted model; it is not substituted for the paper model. Published aggregate comparisons cannot support a paired error analysis.

## Reproduce

The code is on `research/error-analysis-dino-20261019`.

On Snellius, from a frozen source release with the original uv environment:

```bash
uv run --frozen --no-sync python scripts/analyze_previous_runs.py \
  --root /scratch-shared/gmago/video2reaction \
  --out /scratch-shared/gmago/video2reaction/reports/2026-10-19-error-analysis
```

Locally, after copying the compact analysis exports:

```bash
python scripts/render_previous_run_analysis.py \
  --input outputs/analysis/2026-10-19-error-analysis
cd outputs/analysis/2026-10-19-error-analysis
latexmk -pdf -jobname=video2reaction_error_analysis \
  -outdir=../../../output/pdf error_analysis.tex
```

Raw images, videos and full feature tensors remain on Snellius. Rendering uses the local `torch` conda environment; cluster analysis uses uv.

## Execution record

- `28283046`: numerical recomputation completed; export stopped on one missing metadata string stored as JSON NaN.
- `28283214`: corrected numerical audit completed; trajectory export stopped on the shared environment's missing PyTorch runtime helper.
- `28283298`: completed resumed trajectory/projection export, exit `0:0`, 24 seconds; six regression checks passed.

The export now reads saved arrays and the exact prototype asset verified against run provenance, without importing PyTorch. The shared environment still needs its missing `torch/bin/torch_shm_manager` checked before future model jobs. No environment packages were changed here.
