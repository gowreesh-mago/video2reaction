# Video2Reaction report source

This folder is a standalone LaTeX document once its `tables/` and `figures/`
subfolders and `recommended_experiments.tex` are included. It covers all 41 completed benchmark configurations, the dataset
and code audits, hypothesis limits, all official metrics, uncertainty checks,
per-class diagnostics, and 20 selected prediction examples.

## Compile

With a standard TeX Live installation and `latexmk`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error video2reaction_report.tex
```

The result is `video2reaction_report.pdf`. No bibliography service, shell escape,
network access, model weights, or dataset download is needed to compile it.

The delivered ZIP also includes `evidence/` with the aggregate results,
dataset audit, per-class scores, training histories, selected-example JSON, and
the current cluster snapshot. Raw frames, full split metadata, and feature
tensors are excluded.

## Regenerate tables and figures in the repository

From the repository root, using Python with NumPy and Matplotlib:

```bash
python scripts/build_report_assets.py
```

This reads `research_notes/attention_results.json`, `research_notes/recommended_results.json`, the dataset audit, and
`outputs/experiments/*/` artifacts. It does not run training or model inference.
Official scores and top-k memberships are read from cluster outputs rather than
recomputed with a potentially different NumPy tie convention.

Selected movie names were read from only the 20 selected cluster metadata rows.
When regenerating inside the repository, `results/report_examples_metadata.json`
supplies those names. The source archive already contains the generated LaTeX
and needs no regeneration to compile.

## Scope

Benchmark sources: B0 `e91a469`; other Tier 0 `bbf8c1d`; E `a89ab3b`; all 34 new VAD/emotion/peak/description models `99b59ad`.
Full verifier: `cc69100`; launcher-only SLURM parser fix: `2f542f1`.
Report date: 2026-10-05. Budget/storage snapshot: 16:12 UTC.
All prioritized baseline, C/D/E/F, VAD, and description comparisons are complete and verified. The 50-page report includes every run's metrics, selected epoch, job ID, and per-class scores. Optional Tier 2 G/H/I/J and separate A1–A4 benchmark runs remain deferred.

The PDF reports the completed 41-condition benchmark matrix. The source archive's evidence also includes the subsequently requested joint-highlight and pretrained-DSNet implementation review and three-video smoke results; those follow-up full benchmarks are pending.
