# VAD to DINO: completed results

All three seeds completed on Snellius H100 before 00:18 Amsterdam on 22 October, ahead of the requested 06:00 deadline. The source was frozen commit `81b34098f278`. Smoke job `28448335` and seed jobs `28448337_0`, `28448337_1`, `28448337_2` all exited `0:0`.

Each seed fits linear, quadratic and radial-basis regressions plus a shuffled-pair control using training scenes. Hyperparameters are chosen on validation reconstruction error. All 2,070 test clips and 91,312 test scenes are evaluated with the original frozen DINO reaction head. The additional original-feature, training-mean, three-component and 32-component controls distinguish reconstruction effects from a general visual bottleneck.

| Method | Mean KL (lower better) | Mean Top-1 F1 (higher better) |
|---|---:|---:|
| Original DINO | 0.5809 | 0.4831 |
| VAD to DINO, validation-selected radial basis | 1.1072 | 0.2272 |
| Original VAD predictor | 0.8357 | 0.4187 |

The new method does not improve reaction prediction. Its reconstruction error reduction relative to the training-mean feature is only 1.47%, compared with 10.76% for the three-component DINO control. Some matched visual signal is retained, but the inverse mapping cannot recover most of the original feature information. The result applies to these mappings and the frozen head; it does not rule out every VAD approach or a retrained classifier.

The six rare classes have observed probability mass 1.8256%. Original DINO predicts 1.8211%; the reconstructed-feature predictor predicts only 0.0157%. Every clip receives the same top-three set: disapproval, amusement and admiration. Clark distance improves even while this happens, demonstrating why that metric alone cannot establish better rare-reaction prediction.

[Local output index](../../outputs/analysis/2026-10-21-completion/index.html) · [JSON summary](../../outputs/analysis/2026-10-21-completion/03_vad_to_dino/summary/summary.json) · [New PDF](../../output/pdf/video2reaction_vad_dino_supplement.pdf) · [All metrics](../../outputs/analysis/2026-10-21-completion/03_vad_to_dino/summary/all_seed_metrics.csv).

The exact paper-model paired comparison remains blocked by missing fine-tuned predictions/checkpoints. The optional DINO emotion-anchor extension was implemented but not launched: automatic approval review rejected its submission as outside the clarified scope, and a separate approval question is pending.
