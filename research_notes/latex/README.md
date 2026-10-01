# Video2Reaction report source

This folder is a standalone LaTeX document once its `tables/` and `figures/`
subfolders are included. It covers the completed Tier 0 experiments, the dataset
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

This reads the existing `research_notes/tier0_results.json`, dataset audit, and
`outputs/experiments/*/` artifacts. It does not run training or model inference.
Official scores and top-k memberships are read from cluster outputs rather than
recomputed with a potentially different NumPy tie convention.

Selected movie names were read from only the 20 selected cluster metadata rows.
When regenerating inside the repository, `results/report_examples_metadata.json`
supplies those names. The source archive already contains the generated LaTeX
and needs no regeneration to compile.

## Scope

Benchmark source: B0 `e91a469`; learned runs `bbf8c1d`.
Aggregate analysis source: `2a7548e`.
Report date: 2026-10-01. Budget/storage snapshot: 00:38 UTC.
The peak, VAD, and description experiments remain unlaunched. This report does
not imply that the original full research programme is complete.
