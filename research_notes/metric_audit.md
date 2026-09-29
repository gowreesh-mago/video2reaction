# Metric audit

Reference: `src/metrics.py` at upstream `0da6060`, and [paper Table 4](https://arxiv.org/html/2607.06875v1#S4.T4). New implementation: `src/experiments/metrics.py`.

## Preserved definitions

Chebyshev, forward KL(target || prediction), Clark, ordinal CAD, cosine, intersection, MRR, TPE and weighted Top-1/2/3 F1 are implemented. KL preserves the reference `1e-10` stabilization; Clark preserves `1e-8` in its squared denominator. CAD sums absolute CDF differences in the fixed upstream valence/arousal label order after row normalization. Never reorder labels alphabetically.

MRR ranks the target `argmax` in descending prediction order. TPE is absolute probability error **at the target’s dominant class**, corresponding to `mae_top_1_gt`, not the similarly named predicted-top-class diagnostic.

For each k, both target and prediction are converted to multihot arrays using `np.argsort(row)[-k:]`. Per-class F1 is weighted by its frequency in the target Top-k sets. The normalizer is the sum of supports (N*k), as in sklearn multilabel `average='weighted'`; it is not simply N. Even zero-probability classes enter the set if fewer than k labels have positive mass. Preserve this behavior for official scores. Ties use the locked NumPy version’s argsort; MRR uses argmax for its target, which can differ from Top-1 argsort tie breaking. Metadata `dominant_reaction` is not substituted.

Diagnostic per-class precision, recall, F1, target support and prediction support are saved separately. Undefined values use zero, matching sklearn’s default numerical result without its warnings.

## Hand-worked example

Three classes A/B/C and two samples:

| Sample | Target | Prediction | Target Top-2 | Predicted Top-2 |
|---|---|---|---|---|
| 1 | (.6, .3, .1) | (.2, .7, .1) | A,B | A,B |
| 2 | (.1, .7, .2) | (.6, .3, .1) | B,C | A,B |

Top-1 F1 is 0; both target dominant classes have predicted rank 2, so MRR=.5. TPE=(.4+.4)/2=.4. Chebyshev=(.4+.5)/2=.45. CAD=(.4+.6)/2=.5.

At Top-2: A has TP=1, FP=1, FN=0, F1=2/3, support=1. B has F1=1, support=2. C has F1=0, support=1. Weighted F1=(1*(2/3)+2*1+1*0)/4=**2/3**. Top-3 is 1 because all three labels are always selected. High Top-k scores for some classes therefore do not imply uniformly good dominant classification.

## Verification and caveats

Mock tests execute the original metric functions and original Top-k loop verbatim via Python AST, avoiding the legacy loader’s missing NRC file. They compare required outputs on random distributions, sparse targets, and ties, and test the hand-worked example. CAD and all required distribution/ranking metrics agree to numerical precision. TPE is additionally checked against the hand calculation.

The original standalone JS function divides by the number of samples twice. The new JS **training loss** uses the standard formula and has a different interface; official metric definitions remain untouched. CE and forward KL gradients are verified equal for fixed soft targets. No benchmark conclusion is drawn from three-video training-set metrics.
