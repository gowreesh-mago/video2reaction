# Video2Reaction: completed experiments and interpretation

Updated **2026-10-05**. **41 benchmark configurations completed and verified**, plus six GPU smoke runs and three shared feature caches. The queue was empty at 16:28 UTC. Training and official evaluation ran in locked Snellius uv; the dataset remains on the cluster.

- [Self-contained LaTeX source](latex/video2reaction_report.tex); PDF and standalone source ZIP: `output/pdf/video2reaction_report.pdf` and `output/pdf/video2reaction_latex_source.zip`.
- [Complete verified results JSON](recommended_results.json), [live checklist](experiment_checklist.md), [resource snapshot](cluster_status_20261005.json).
- [Primary launch receipt](recommended_primary_launch.json), [remaining peak grid](emotion_curve_launch.json), [20 selected prediction examples](qualitative_examples.json).

## 1. Strongest result

**Visual + description** has the lowest validation KL (**0.510391**) and test KL (**0.509952**). It improves test KL by **6.33%** versus B1 (0.544414), with paired difference **−0.034462**, 95% movie-bootstrap interval **[−0.042130, −0.027175]**. It also improves over equally sized visual+visual and text+text controls. This supports complementary information from descriptions and appearance in this setup.

This is **text-assisted prediction**. The best visual-only test KL is shared attention (**0.540655**). Description-only scores **0.549432**, inconclusive against B1. E has the best validation KL among the seven reference models; shared E's lower test KL does not override validation-based selection.

## 2. What the controlled experiments show

- **Visual features:** B1 improves substantially over the constant prior (0.689281 KL).
- **Learned shared attention:** a small KL improvement over B1, −0.003759 [−0.006582, −0.000933]. Reaction-specific E has no established advantage over its matched shared control.
- **Peak hypothesis:** all 20 K=1/2/4/8 subsets have worse KL than all-frame B1, with their paired intervals above zero. Uniform selection has lower KL than each emotion score at every K. Eleven of 12 emotion-versus-uniform intervals exclude zero in the unfavorable direction; arousal K=8 is inconclusive. Increasing K helps; no inverted-U favoring sparse peaks appears.
- **Global + peak:** KL 0.556444 versus global+global 0.546039 and peak+peak 0.592088. Global context helps a peak-only representation, but adding these peaks hurts the matched global predictor. The pooled representation measures broad appearance, not narrative order.
- **VAD auxiliary:** KL 0.549865, worse than B1 and its permuted-coordinate control (0.547221).
- **VAD geometry:** KL 0.542777 versus permuted control 0.546931; difference −0.004154 [−0.008003, −0.000237]. The B1 comparison is −0.001637 [−0.003355, +0.000050], so a baseline gain is not established. Sourced geometry also has slightly worse expected-VAD error than its control (0.039762 vs 0.039529). We cannot claim more affectively sensible predictions.
- **Extra frame-emotion features:** C logits worsens KL; C VAD and C both have worse point estimates with intervals including zero. These results constrain the chosen frozen affective model and fusion.
- **Temporal/loss hypotheses:** B2 is order-sensitive but does not clearly beat its matched set control; A5's small ranking/cosine gains remain exploratory.

The code implements the stated interventions and matched controls. The results test these configurations, not every possible model in the search space. One seed, one VAD permutation, fixed coefficients, and unadjusted multiple comparisons limit small-gain claims.

## 3. Complete experiment table

All rows below completed with exit `0:0`. Scores use the official 2,070 test clips; checkpoints were selected by validation KL. `description_only`, `visual_description`, and `description_text_control` use descriptions; `description_visual_control` is visual-only. Exact run IDs, epochs, all 11 metrics, uncertainty, strata, and provenance are in the JSON.

| Experiment | Hypothesis | Change | Status | Cos | Inter | Cheb | KL | MRR | F1@1 | F1@3 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `b0_prior` | Class-frequency bias | Training prior | complete | 0.751309 | 0.563053 | 0.273683 | 0.689281 | 0.599569 | 0.237624 | 0.558669 |
| `b1_meanpool` | Visual information | Mean pool | complete | 0.825196 | 0.630170 | 0.224348 | 0.544414 | 0.723182 | 0.529069 | 0.619689 |
| `b2_temporal` | Temporal positions | Transformer + positions | complete | 0.825163 | 0.636659 | 0.219920 | 0.546904 | 0.729166 | 0.541735 | 0.607179 |
| `b2_set_control` | Order control | Transformer, no positions | complete | 0.822704 | 0.630530 | 0.226002 | 0.546476 | 0.728609 | 0.535816 | 0.611040 |
| `a5_distribution` | Distribution/ranking loss | KL + cosine + rank | complete | 0.825493 | 0.629496 | 0.224688 | 0.544462 | 0.725885 | 0.535416 | 0.621179 |
| `e_reaction_query` | Reaction-specific evidence | Separate queries | complete | 0.825756 | 0.633909 | 0.223387 | 0.542404 | 0.731984 | 0.533487 | 0.619849 |
| `e_shared_query_control` | Shared evidence control | Average query maps | complete | 0.827453 | 0.633245 | 0.223097 | 0.540655 | 0.733887 | 0.541500 | 0.619977 |
| `b_vad_aux` | Affective geometry | aux | complete | 0.822601 | 0.636937 | 0.221645 | 0.549865 | 0.727498 | 0.533181 | 0.608039 |
| `b_vad_aux_permuted` | Affective geometry | aux permuted | complete | 0.823819 | 0.637000 | 0.220749 | 0.547221 | 0.726579 | 0.535042 | 0.607344 |
| `b_vad_geometry` | Affective geometry | geometry | complete | 0.825891 | 0.631685 | 0.223796 | 0.542777 | 0.728741 | 0.534576 | 0.619414 |
| `b_vad_geometry_permuted` | Affective geometry | geometry permuted | complete | 0.822858 | 0.631471 | 0.223738 | 0.546931 | 0.725470 | 0.537017 | 0.614993 |
| `c_emotion_logits` | Extra frame emotion evidence | logits | complete | 0.819768 | 0.629506 | 0.225354 | 0.555031 | 0.718845 | 0.526002 | 0.620171 |
| `c_emotion_vad` | Extra frame emotion evidence | vad | complete | 0.823664 | 0.633332 | 0.222518 | 0.546309 | 0.722295 | 0.530046 | 0.615987 |
| `c_emotion_both` | Extra frame emotion evidence | both | complete | 0.823180 | 0.634709 | 0.221415 | 0.548175 | 0.727355 | 0.540566 | 0.611950 |
| `f_global_peak` | Global/peak complementarity | global peak | complete | 0.821135 | 0.631945 | 0.223399 | 0.556444 | 0.724315 | 0.531726 | 0.600588 |
| `f_global_control` | Global/peak complementarity | global control | complete | 0.824158 | 0.630224 | 0.224977 | 0.546039 | 0.729315 | 0.542446 | 0.614177 |
| `f_peak_control` | Global/peak complementarity | peak control | complete | 0.803569 | 0.615551 | 0.236073 | 0.592088 | 0.701311 | 0.496467 | 0.589874 |
| `d_arousal_k4` | Sparse emotional peaks | arousal k4 | complete | 0.803579 | 0.615184 | 0.235206 | 0.582722 | 0.696136 | 0.482662 | 0.593611 |
| `d_distance_k4` | Sparse emotional peaks | distance k4 | complete | 0.804944 | 0.616154 | 0.234782 | 0.582355 | 0.704684 | 0.490379 | 0.592189 |
| `d_confidence_k4` | Sparse emotional peaks | confidence k4 | complete | 0.806370 | 0.614688 | 0.235310 | 0.581480 | 0.703940 | 0.492537 | 0.596288 |
| `d_uniform_k4` | Sparse emotional peaks | uniform k4 | complete | 0.808205 | 0.618083 | 0.234314 | 0.571789 | 0.704261 | 0.498136 | 0.597324 |
| `d_random_k4` | Sparse emotional peaks | random k4 | complete | 0.806701 | 0.617433 | 0.233552 | 0.578721 | 0.701115 | 0.493597 | 0.591611 |
| `description_only` | Description complementarity | Text only | complete | 0.812237 | 0.625449 | 0.232537 | 0.549432 | 0.707047 | 0.503028 | 0.609576 |
| `visual_description` | Description complementarity | Visual + text | complete | 0.838024 | 0.646469 | 0.213240 | 0.509952 | 0.747897 | 0.563316 | 0.625897 |
| `description_visual_control` | Description complementarity | Visual + visual control | complete | 0.825674 | 0.638793 | 0.219294 | 0.547418 | 0.730116 | 0.542795 | 0.613900 |
| `description_text_control` | Description complementarity | Text + text control | complete | 0.811787 | 0.622111 | 0.234198 | 0.551493 | 0.704908 | 0.504131 | 0.609321 |
| `d_arousal_k1` | Sparse emotional peaks | arousal k1 | complete | 0.786265 | 0.593537 | 0.250454 | 0.621340 | 0.672368 | 0.446899 | 0.582725 |
| `d_distance_k1` | Sparse emotional peaks | distance k1 | complete | 0.786023 | 0.599984 | 0.246860 | 0.622143 | 0.670257 | 0.445769 | 0.582505 |
| `d_confidence_k1` | Sparse emotional peaks | confidence k1 | complete | 0.784619 | 0.598132 | 0.247868 | 0.622401 | 0.676473 | 0.447044 | 0.579615 |
| `d_uniform_k1` | Sparse emotional peaks | uniform k1 | complete | 0.796656 | 0.603943 | 0.242768 | 0.606587 | 0.690428 | 0.481068 | 0.586533 |
| `d_random_k1` | Sparse emotional peaks | random k1 | complete | 0.792675 | 0.605288 | 0.244160 | 0.610679 | 0.668951 | 0.447572 | 0.581584 |
| `d_arousal_k2` | Sparse emotional peaks | arousal k2 | complete | 0.797505 | 0.604829 | 0.242366 | 0.599454 | 0.690178 | 0.471438 | 0.589194 |
| `d_distance_k2` | Sparse emotional peaks | distance k2 | complete | 0.791783 | 0.606680 | 0.242237 | 0.605554 | 0.680779 | 0.451205 | 0.582860 |
| `d_confidence_k2` | Sparse emotional peaks | confidence k2 | complete | 0.798008 | 0.605696 | 0.240900 | 0.598986 | 0.685914 | 0.461666 | 0.587394 |
| `d_uniform_k2` | Sparse emotional peaks | uniform k2 | complete | 0.803085 | 0.611418 | 0.237495 | 0.590774 | 0.701294 | 0.497615 | 0.592070 |
| `d_random_k2` | Sparse emotional peaks | random k2 | complete | 0.799548 | 0.611202 | 0.239431 | 0.594896 | 0.695252 | 0.484312 | 0.597496 |
| `d_arousal_k8` | Sparse emotional peaks | arousal k8 | complete | 0.815214 | 0.626658 | 0.229398 | 0.563740 | 0.717950 | 0.520358 | 0.614063 |
| `d_distance_k8` | Sparse emotional peaks | distance k8 | complete | 0.814755 | 0.630320 | 0.226910 | 0.566619 | 0.714668 | 0.517064 | 0.602767 |
| `d_confidence_k8` | Sparse emotional peaks | confidence k8 | complete | 0.814179 | 0.623501 | 0.229561 | 0.566023 | 0.711281 | 0.509469 | 0.603148 |
| `d_uniform_k8` | Sparse emotional peaks | uniform k8 | complete | 0.819422 | 0.625039 | 0.228571 | 0.557994 | 0.721079 | 0.527125 | 0.615327 |
| `d_random_k8` | Sparse emotional peaks | random k8 | complete | 0.814633 | 0.623584 | 0.230654 | 0.563697 | 0.710777 | 0.501896 | 0.603482 |

## 4. Failures and verification

The first visual-cache attempt failed because an unused tokenizer dependency entered the production path. A shared image-only loader fixed it; dependent cancellations are retained as infrastructure failures. The final curve launcher initially rejected valid SLURM output without a trailing separator. Fix `2f542f1` accepts both forms; 12 scheduler tests pass. It stopped before submitting any jobs and did not change model code.

Both new three-video smoke jobs passed 61 cluster tests, real pretrained inference, normalized predictions, decreased fitting loss, and exact checkpoint reload. All 34 new predictors use frozen source `99b59ad`. The full verifier `cc69100` recomputed every validation/test metric in NumPy 2.2.6, checked sample identities and checkpoint selection, verified VAD assignments, and recomputed all saved frame scores/selections/pooling weights for 26 C/D/F models. No benchmark model ran on the Mac.

## 5. Limits and comparison with the paper

The final registry audit found one stale `running` entry for completed VAD geometry job `27612816`. Its SLURM exit `0:0`, source identity, metrics, and prediction checksum were verified before restoring `completed` through the locked API. The cause remains unresolved; audit cross-node registry visibility before another large concurrent batch. [Reconciliation record](registry_reconciliation_20261005.json). All 41 model artifacts remain valid.

95.94% of test clips share a movie with training. Visual+description has KL 0.581016 on the 84 unseen-movie clips versus B1 0.600715, but this small subset has no separate uncertainty analysis. No claim of movie-disjoint generalization follows. Support-weighted F1 can conceal rare-class failure; per-class scores accompany every run. Official top-k ties depend on NumPy, so the locked cluster convention was preserved.

We do **not** beat the paper overall. Our visual+description KL 0.509952 is numerically lower than the paper's SA-BFGS 0.5976, but our MRR 0.747897 and F1@1 0.563316 remain below LLaVA's 0.7833/0.6521 and Qwen's F1@1 0.6577. Inputs, backbones, and training regimes differ. [Paper Tables 5–6](https://arxiv.org/html/2607.06875v1)

## 6. What remains and next research decisions

The prioritized baseline, C/D/E/F, VAD, and description comparisons are complete. Optional G/H/I/J and separate A1–A4 full runs remain deferred under the original prioritization. The user subsequently requested joint highlight/reaction learning and a pretrained highlight teacher. These new baselines are implemented separately from the completed matrix; their execution status is in the checklist and their [review](highlight_review.md). A next batch should target a distinct uncertainty—such as movie familiarity or a stronger salience proxy—and be declared before looking at its test results.

**Answer to the central questions:** this setup favors broad visual aggregation over the tested sparse emotional peaks. Adding supplied descriptions helps beyond added capacity. It does not establish narrative understanding. VAD geometry has a control-relative KL benefit but no established improvement over B1, while auxiliary regression harms KL.

## 7. Reproduction and artifacts

All new predictors use `/scratch-shared/gmago/video2reaction/releases/99b59ad10207-2f524d3cd694`. An intentional repeat of the best text-assisted model:

```bash
cd /scratch-shared/gmago/video2reaction/releases/99b59ad10207-2f524d3cd694
source configs/clusters/snellius.env
sbatch --test-only slurm/visual_description.sbatch
sbatch --output="$V2R_ROOT/logs/slurm/visual_description-%j.out" slurm/visual_description.sbatch
```

This creates a new allocation and output directory. For read-only verification, run `scripts/analyze_recommended.py` from the current cluster code, passing the two exact submission JSON files named in `recommended_results.json` (exclude `.intent.json`). It reads saved outputs and private VAD assets; it does not fit models.

Each run has config, source/environment provenance, selected/last checkpoint, validation/test predictions, all metrics, per-class scores, strata, and logs. Peak/query runs additionally save frame evidence or attention. Registry: `results/experiment_registry.json`. Local synchronization: `bash scripts/sync_outputs.sh`; code synchronization: `bash scripts/sync_cluster.sh`. Neither copies the full dataset to the Mac.
