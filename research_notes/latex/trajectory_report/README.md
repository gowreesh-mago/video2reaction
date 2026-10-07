# Visual emotion trajectory report

This is a self-contained LaTeX report of 264 completed benchmark runs (88 conditions, three seeds), their hypotheses, exact model path, statistical comparisons, and published Video2Reaction scores.

## Compile the delivered source bundle

From the directory containing `main.tex`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=video2reaction_trajectory_report main.tex
```

The source bundle includes all required tables and vector figures. It does not require video data, a GPU, or private NRC assets to compile. `verified_results.json` and `report_manifest.json` provide the evidence and input hashes.

## Rebuild assets in the repository

From the repository root, using Python with NumPy, Matplotlib, and PyYAML:

```bash
python scripts/build_trajectory_report_assets.py
```

This formats previously verified aggregate results. It does not run models or recompute official metrics on the Mac. Official metric recomputation belongs in the locked cluster environment:

```bash
source configs/clusters/snellius.env
uv run --frozen --no-sync python scripts/analyze_trajectories.py --root "$V2R_ROOT"
```

For the repository PDF destination:

```bash
cd research_notes/latex/trajectory_report
latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=video2reaction_trajectory_report -outdir=../../../output/pdf main.tex
```

## Interpretation

Scores average seeds 42, 43 and 44; they are not ensemble scores. Model selection uses mean validation KL. The bootstrap resamples test movies conditional on three fitted models; it does not resample training or correct for exploratory multiple comparisons. All 264 runs completed, and their official metrics were independently checked. Full temporal-attribution claims remain unsupported by independent labels.
