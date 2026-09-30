# Code and adversarial hypothesis review

Reviewed 2026-09-30. Starting revision: `107f593`. Scope: the changes since upstream `0da6060`, the original experiment request, and the implementation added in response to this review.

## Verdict

The starting code correctly supported a three-video infrastructure smoke test. It did **not** implement full benchmark experiments, so its results could neither support nor reject the research hypotheses. The overfitting losses must not be compared as benchmark scores.

The first full batch now tests five specific configurations: B0, B1, B2, a matched B2 set-transformer control, and A5. These can support or weaken narrow hypotheses under the pinned backbone, data, loss coefficients, and training budget. **They cannot prove or disprove the entire model search space.** One seed gives no estimate of variation across training seeds. An inconclusive result is not a rejection of all temporal, emotion, VAD, or peak approaches.

## Findings, ordered by consequence

### P1 — No full experiment runner at the reviewed starting revision

Evidence: `scripts/smoke_test.py` selected 2–3 explicit training IDs, optimized on those clips, and evaluated those same targets. `configs/experiments/smoke_3videos.yaml` contained only the infrastructure hypothesis. There were no full-run configs, validation checkpoint selection, or baseline jobs.

Consequence: successful smoke execution established environment/model plumbing only. It did not reproduce B0/B1/B2/A5 or measure generalization.

Resolution: added `train.py`, shared training/cache/diagnostic modules, resolved configs, independent jobs, and bounded submission. Test predictions are produced only after validation KL has selected the checkpoint. Full benchmark results remain pending until those jobs complete.

### P1 — Mean pool versus transformer confounds chronology with capacity

Evidence: at the smoke dimensions (1152 input, 128 hidden), mean pooling has **169,109** trainable parameters and the temporal model has **565,781**. `ReactionPredictor` adds two attention/MLP layers as well as positional encoding. A synthetic reversal probe changes temporal logits but leaves mean/query predictions unchanged to numerical tolerance; sensitivity alone is not evidence that the information helps.

Consequence: B2 > B1 would show that the whole aggregation change helps. It would not identify temporal order as the cause.

Resolution: added `b2_set_control`, using identical weights at initialization, layer count, optimizer, head, parameter count, and data, with positional encoding disabled. An adversarial test verifies its permutation invariance. Saved shuffled-order predictions probe reliance on chronology separately. Even B2 > set control supports use of frame-position information, not proof of narrative understanding; rank positions also encode beginning/end cues, and scene intervals are irregular.

### P1 — Queued jobs could execute newer code than the submitted revision

Evidence: the previous SLURM job entered the mutable synchronized working directory. Recording a manifest does not prevent a later rsync from changing Python/config files before a queued job starts.

Consequence: results might be attributed to the wrong source/configuration.

Resolution: batch submission creates a separate release from verified tracked files. Jobs execute there, verify its hashes, and use the existing uv environment with `--no-sync`. A regression test changes the working copy after freezing and checks isolation and tamper detection. Do not mutate the shared uv environment while jobs are active; `uv.lock` and package versions are recorded.

### P1 — Duplicate invocation could corrupt completed registry state

Evidence: `update_registry` guarded only a late `submitted` update. A repeated smoke run refusing to overwrite output could then mark the existing completed run `failed`. This was reproduced with synthetic registry updates.

Resolution: completed runs reject status regression; a run ID cannot be assigned another job ID; missing job IDs do not erase an existing ID. New attempts use new IDs. Submission locking, recorded IDs, scheduler comments, and persisted submission intent prevent blind resubmission after an interrupted submitter. Terminal scheduler failures can be reconciled by the result collector, including dependency cancellations whose Python runner never starts.

### P2 — The previous resume check did not establish faithful resumption

Evidence: the smoke check verified exact checkpoint logits and one finite optimizer step. It did not compare an interrupted trajectory with uninterrupted training or save RNG state for a full data loop.

Resolution: full checkpoints save model, optimizer, RNG states, selected weights, epoch/history, and patience. An interruption test confirms exactly matching resumed CPU training history and final weights. Resume refuses a changed config/source identity. CUDA execution still needs the cluster check; deterministic flags do not establish cross-hardware identity.

### P2 — Sparse smoke sampling cannot test peaks or all-frame context

Evidence: the smoke config caps every clip at eight uniformly spaced frames. A peak absent from those eight cannot be recovered by a later selector.

Resolution for this batch: encode **all** indexed keyframes once, preserving IDs, frame indices/timestamps, image-byte hashes, split hashes, and pinned encoder/processor identity. Cache readers reject incomplete data, ID mismatches, and checksum changes. Later peak methods must select from this common candidate set. Keyframes still omit within-scene motion, audio, and some events; they do not represent the complete video.

### P2 — Loss coefficients and research comparisons need explicit scope

Evidence: cosine/ranking weights, ranking margin, and target gap previously came only from function defaults. CE and forward KL have equal gradients for normalized fixed targets. A5 changes two terms simultaneously.

Resolution: resolved configs save all coefficients; A5 changes only the objective relative to B1, using matched initialization and training settings. The ranking term is tested against a hand-derived value/gradient and a no-pair case. A5 tests the **combination at fixed coefficients**; it cannot identify which term helped. Add KL+cosine and KL+ranking controls in a later batch if the combination merits explanation. Avoid treating CE versus KL as independent learning hypotheses.

### P2 — Official split overlap and label imbalance limit interpretation

Evidence: 1,986 of 2,070 test clips share a movie with training; only **84** test clips come from unseen movies. Three classes carry about 68.59% of mean training target mass. Top-k ground-truth sets may include zero-probability labels and use reference tie conventions. CAD follows a fixed one-dimensional class order, not full three-dimensional VAD distance.

Resolution: retain official metrics/splits, add seen/unseen-movie, entropy, frame-count, and dominant-probability diagnostics plus per-class scores. Derive stratification cut points from training only. Do not call unseen-subset results a newly trained movie-disjoint benchmark, or treat the small subset as decisive. Weighted F1 can conceal rare-class failures. A CAD gain alone would not validate VAD geometry.

## Does each experiment implement its hypothesis?

### Additional integration finding from the first launch

Feature job `27382107` exposed a gap between the smoke and cache loaders: `AutoProcessor(use_fast=False)` propagated the flag to the unused Gemma text tokenizer and required SentencePiece. The prior smoke had left that flag unset. B0 completed, but the four dependent learned runs were cancelled before training. The fix uses `AutoImageProcessor` and a shared frozen-encoder loader in both smoke and full extraction, avoiding an unnecessary tokenizer dependency. A new synthetic end-to-end cache round-trip test covers extraction, image hashes, ordering, serialization, loading, and revisiting a completed cache. All **22 local tests pass** after this fix; the real three-video shared-loader regression must pass before replacements are submitted. This demonstrates why a smoke test must exercise the same loader as production.

The first B0 result is a valid class-prior floor (test KL .689281, cosine .751309, MRR .599569, F1@3 .558669). It provides no evidence about visual, temporal, peak, or VAD models.

| Experiment | Actual intervention / needed comparison | What a favorable result supports | What it cannot establish |
|---|---|---|---|
| B0 | Mean of training reaction distributions, held constant on val/test | Performance explained by class-frequency bias | Movie familiarity, temporal or visual evidence |
| B1 | Frozen SigLIP2 frame features, per-frame LayerNorm/projection, mean, common MLP; compare B0 | Visual features help beyond the constant prior on these splits | Causal scene semantics, generalization to new movies, or exact reproduction of the paper's different encoder |
| B2 | Chronological positions in a two-layer transformer; compare matched set transformer, with B1 secondary | Frame-position information helps at the fixed representation/training budget | General temporal reasoning, motion understanding, or superiority of all temporal models |
| B2 set control | Same transformer with positions disabled | Attention/capacity benefit without chronology | That chronology is useless if B2 ties it |
| A5 | B1 architecture with KL + 0.2 cosine + 0.1 ranking; gap .03, margin .1 | This composite improves the recorded metric tradeoff | Which term caused it, optimal coefficients, or global loss superiority |
| B-VAD1/2 | **Pending:** auxiliary target expectation / classifier geometry | Needs reaction-only and matched random/permuted-geometry controls | Sourced coordinates alone do not implement VAD learning; CAD alone is insufficient |
| C | **Pending:** visual vs visual+emotion vs visual+VAD vs all | Incremental predictive value of perceived-emotion evidence | Perceived emotion equals induced audience response; external pretraining is unconfounded |
| D | **Pending:** emotion-ranked K versus uniform/random K with the same candidate frames, backbone, and head | Selection adds value beyond frame-count reduction | Small K beating all frames alone could be regularization, not emotional salience |
| E | Query attention is smoke-tested; full run **pending** | Learned reaction-specific pooling helps versus matched mean/shared attention | Attention weights identify causal evidence; query pooling has no chronology |
| F | **Pending:** global, peak, and fused predictors plus a matched extra-capacity control | Complementary value of both sources | Fusion wins solely because it uses both narrative and emotion, without ruling out added capacity |
| Description diagnostics | **Pending:** description-only, visual-only, joint | Dependence on semantic text/metadata | Text-assisted performance is video-only forecasting |
| G/H/I/J | **Pending:** prototypes/retrieval/hierarchy/entropy | Specific controlled additions, once implemented | These branches have been searched or rejected already |

## Adversarial questions before interpreting results

1. **Could a simpler explanation give the same outcome?** Capacity, regularization, movie familiarity, class priors, and pretraining are alternatives to the headline hypotheses. Use the controls above.
2. **Did the model have access to the alleged signal?** Static keyframes contain ordering but limited motion. Frozen visual features may discard emotion cues. A negative result rules out neither signals absent from the input nor untested encoders.
3. **Was only one component changed?** B2/set and A5/B1 now satisfy this at the config level. B2/B1 is an architectural comparison, not a clean order ablation. B1's per-frame normalization is part of the documented baseline.
4. **Was selection decided before test scores?** All trained first-batch models use the same maximum 50 epochs, AdamW settings, constant learning rate, patience 8, minimum KL improvement .0001, and validation KL checkpoint rule. Ranking gains that sacrifice KL may not be selected; interpret that as a property of the predeclared rule, not a complete rejection of ranking objectives.
5. **What would count as counterevidence?** Report paired metric differences, failures to improve, and conflicts between KL/ranking/rare classes. B2 failing against the matched control weakens this particular chronology implementation; A5 failing at one coefficient choice weakens this recipe. Neither rejects the broad search space.
6. **Could random variation explain the margin?** One seed is the user's constraint. Use movie-clustered paired bootstrap intervals when collecting full results; these estimate test-sample uncertainty, not training-seed variance. Small differences remain provisional. Do not select additional configs using these test outcomes and then present them as an untouched confirmation set.
7. **Do exploratory branches have fair negative controls?** Peaks need equal-K random/uniform selection; VAD needs permuted geometry and random auxiliary labels/embeddings with matched scale (permute only the auxiliary assignment, preserving the published class/CAD order); retrieval must exclude the test clip and separately exclude same-movie neighbors; entropy/hierarchy need capacity-matched auxiliaries.

## Validation and launch decision

- Original 14 mock tests passed before changes.
- After fixes/additions, **21 synthetic tests pass locally** in the allowed conda `torch` environment. No benchmark data was brought to the Mac.
- Tests now cover the order control, ranking math, registry terminal state, immutable releases, interrupted/resumed training, cache corruption/identity, and configuration equality, in addition to the earlier metric/input/padding tests.
- **Cluster regression passed:** job `27381915`, Snellius uv, all 21 tests plus the three-video GPU integration check, `COMPLETED`, exit `0:0`, elapsed `00:05:41`. Tested revision `e91a469`.
- **First batch submitted:** feature cache `27382107`, B0 `27382108`, B1 `27382109`, B2 `27382110`, set control `27382111`, A5 `27382112`, from frozen release `e91a469f51ba-b53672620486`. The four learned predictors depend on cache success. Live status is maintained in the [checklist](experiment_checklist.md). No Tier 1/2 results are implied by this first batch.

Remaining limitations: one seed, fixed coefficients/backbone, official movie overlap, static scene keyframes, unfinished higher-tier hypotheses, and post-run confidence/qualitative analysis. These limit scientific claims; they do not invalidate the bounded first-batch comparisons.
