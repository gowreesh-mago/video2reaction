# Experiment report: completed Tier 0 and attention, remaining hypotheses

Updated: 2026-10-05. This is the report for completed work; the full research
programme remains incomplete because the peak, VAD, and description experiments
have not yet been conducted.

Launch update, 2026-10-05: C/D/F, VAD, and description comparisons are
pushed and synced; 61 local synthetic tests pass. Three-video cluster smoke jobs
`27611469` (emotion/peak/VAD) and `27611475` (descriptions) are submitted and
pending for GPU resources. See the [live checklist](experiment_checklist.md),
[VAD review](vad_review.md), and [description review](description_review.md).
E jobs `27438329` and `27438330` completed and their results are collected.
SSH access has recovered. The PDF still covers the five Tier 0 experiments;
the table below includes the two new attention results. All seven runs' metrics
recompute in locked cluster uv; E attention artifacts and paired uncertainty
are verified. [E analysis JSON](attention_results.json)

The **self-contained 27-page report** is authored in
[LaTeX](latex/video2reaction_report.tex), with build instructions in
[latex/README.md](latex/README.md). It includes method equations, all official
metrics, hypothesis controls, failure/recovery history, bootstrap intervals,
per-class precision/recall/F1/support for every model, 20 selected prediction
examples, exact run identities, and reproduction commands. Compiled deliverables:
`output/pdf/video2reaction_report.pdf` and
`output/pdf/video2reaction_latex_source.zip`.

## Completed results

All values below are official test scores on 2,070 clips. All runs completed with
exit `0:0`, with validation-only checkpoint selection for learned models.

| Experiment | Hypothesis / change | Status | Cos | Inter | Cheb | KL | MRR | F1@1 | F1@3 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | Constant training-label prior | completed | .751309 | .563053 | .273683 | .689281 | .599569 | .237624 | .558669 |
| B1 | Frozen visual features + mean pooling | completed | .825196 | .630170 | .224348 | .544414 | .723182 | .529069 | .619689 |
| B2 | Add chronological positions in a transformer | completed | .825163 | .636659 | .219920 | .546904 | .729166 | .541735 | .607179 |
| Set | Matched B2 transformer, positions disabled | completed | .822704 | .630530 | .226002 | .546476 | .728609 | .535816 | .611040 |
| A5 | B1 + cosine/ranking loss terms | completed | .825493 | .629496 | .224688 | .544462 | .725885 | .535416 | .621179 |
| E | Reaction-specific attention | completed | .825756 | .633909 | .223387 | .542404 | .731984 | .533487 | .619849 |
| E shared | Capacity-matched shared attention | completed | .827453 | .633245 | .223097 | .540655 | .733887 | .541500 | .619977 |

## 1. Best-performing experiment

E has the lowest validation KL (0.537482); its shared-attention control has the
lowest test KL (0.540655). B1 remains the mean-pooling reference, lowering test
KL by 21.0% versus B0. The shared-attention control scores slightly better than
reaction-specific attention on test KL, so the point estimates do not support
an advantage from separate evidence for each reaction. B2 retains the highest
F1@1 (0.541735); no model wins every metric.

## 2. Supported hypotheses

Frozen visual features add predictive information beyond the global class prior
on the official splits. The paired movie-cluster interval for B1 minus B0 KL is
[-0.157545, -0.131936]. Shuffling worsens B2's KL, showing order sensitivity in
that trained model.

Shared attention improves over B1 by −0.003759 KL, with a 95% paired movie-cluster
bootstrap interval of [−0.006582, −0.000933] over 10,000 resamples. This is a
small positive result for learned shared pooling. It is conditional on these
trained models: one seed, no training-seed uncertainty, and no correction for
multiple comparisons.

## 3. Rejected and inconclusive hypotheses

The strong claim that this B2 positional configuration improves primary test KL
over the matched set model is not supported. The difference is +0.000428 with an
interval spanning zero. All Tier 0 between-learned-model KL intervals include zero;
they establish neither superiority nor equivalence. A5's small ranking/cosine
gains remain exploratory. These results do not reject the wider search space.

Reaction-specific E minus B1 is −0.002010 KL [−0.005060, +0.001068]; E minus its
shared-query control is +0.001749 [−0.000422, +0.003881]. Neither interval
establishes a benefit from reaction-specific queries. Attention is broad:
test normalized entropy averages 0.9773 for reaction-specific maps and 0.9537
for shared maps (1 denotes uniform attention). Distinct class-specific maxima
do not imply sparse attention or causal evidence. These results do not rule out
emotion-ranked peaks; that hypothesis needs the pending D/F comparisons.

## 4. Important failure modes

The first feature job failed through an unnecessary tokenizer dependency;
dependent jobs were cancelled before training. The shared image-only loader
fixed the issue and passed the three-video regression. The failures are retained
in the checklist and do not count as negative model results.

Rare-class performance is weak: B1 has nonzero Top-1 F1 for four classes and no
Top-1 predictions for 14. Twenty selected cases illustrate large improvements,
regressions, and high/low target entropy. They are distribution diagnostics;
visual scene-level causes have not been verified.

## 5. Suspicious benchmark behavior

95.94% of test clips share a movie with training. Top-k target sets can include
zero-mass labels and are sensitive to NumPy tie ordering. All official scores
were verified in the locked NumPy 2.2.6 cluster environment; local NumPy 1.26.4
must not silently replace that ranking convention. Support-weighted F1 can mask
poor coverage of rare reactions. Only one training seed was used.

## 6. Recommended next experiments and unanswered questions

The peak/global-context question remains unanswered: no emotion-ranked Top-K or
global-plus-peak model has been trained. VAD coordinates are sourced, but no VAD
auxiliary or geometry model has been evaluated, so predictive improvement is
unknown. C/D/F with equal-K random/uniform and capacity controls, VAD semantic
controls, and description-only/visual+description diagnostics are implemented.
Their smoke jobs are queued; full benchmark batches await successful GPU smoke
checks. E is complete, and its broad attention distributions do not establish
that sparse peaks dominate reaction forecasting.

## 7. Exact reproduction commands and artifacts

See Section 10 of the LaTeX report for the frozen-release B1 SLURM commands,
environment, cache, paths, and manifest identities. Aggregate metrics and
intervals: [tier0_results.json](tier0_results.json). Selected examples:
[qualitative_examples.json](qualitative_examples.json). Live run history and
pending work: [experiment_checklist.md](experiment_checklist.md).

Attention verification (run from the synced code directory on Snellius):

```bash
source configs/clusters/snellius.env
uv run --frozen --no-sync python scripts/analyze_attention.py \
  --outputs "$V2R_OUTPUT_DIR/experiments" \
  --output "$V2R_ROOT/results/attention_results.json"
```
