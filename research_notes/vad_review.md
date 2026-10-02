# VAD objectives: code and hypothesis review

Status: implemented locally on 2026-10-02; dataset experiments and the cluster smoke are pending SSH access. No VAD result is claimed.

## What changes

Both objectives start from B1 and retain its official splits, visual cache, seed 42, optimizer, batch size, epoch limit, and validation-KL checkpoint selection.

- **B-VAD1:** a linear 128→3 head predicts the target distribution's expected VAD. The added loss is the mean over clips of the sum of squared errors across the three axes, with weight 1.0. It updates the auxiliary head and shared visual projection. The reaction head's initial parameters and predictions remain exactly equal to B1. The extra head adds 387 parameters.
- **B-VAD2:** the existing final classifier's 21 row vectors are regularized toward `exp(-2 * squared_VAD_distance)` in cosine similarity. The loss averages the 210 distinct unordered pairs, excludes diagonal entries, and has weight 0.1. It adds no parameters.
- Each objective has a **permuted-coordinate control** with identical architecture, initialization, coefficient, and training settings. Seed 271828 fixes one permutation of the label-to-coordinate assignments. The coordinate cloud and all pairwise distances as an unordered set are preserved.

The coefficients and permutation were fixed before seeing any result. Validation and test metrics use the reaction distribution only. Auxiliary predictions do not replace the 21-label output.

## Provenance and engineering checks

`src/experiments/vad.py` loads the private, checksum-verified NRC v1 asset from `V2R_REACTION_VAD_FILE`. It requires exact entries for all 21 reaction words, their original class order, finite coordinates in [0,1], and no undocumented approximation. Each run records the source, asset checksum, mapping, and assigned source word for every class in `vad_provenance.json`.

History records the reaction loss, unweighted VAD loss, and combined objective separately. Checkpoints include the auxiliary head when present; the production fit routine verifies exact reaction predictions after reloading the selected checkpoint. Resume retains the existing config/source identity checks and optimizer/RNG state.

Local synthetic tests check hand-computed losses, pair normalization, gradient direction, the auxiliary gradient path, exact common initialization, coordinate-permutation provenance, corrupted-asset rejection, control-config equality, and production training/checkpoint reload for both objectives. The local suite has 45 passing tests. These tests establish implementation behavior, not benchmark performance.

## Adversarial scientific review

**Does the code implement the proposed hypotheses?** Yes: one intervention supervises expected affect; the other shapes classifier directions using sourced affective distances. Neither maps the 21 targets onto the coarse eight-class frame taxonomy.

**Would better performance establish that meaningful VAD geometry helps?** A gain over B1 alone is insufficient. It could come from generic regularization or the additional auxiliary task. The primary semantic comparisons are each sourced objective versus its permuted counterpart, alongside B1. Use paired movie-cluster uncertainty and report all four runs.

The auxiliary permutation does not preserve class-frequency-weighted target means, variances, or regression difficulty. The geometry control preserves the unweighted distance set, but changes its relation to class frequency and the supervised reaction loss. One permutation, one training seed, and one coefficient setting limit conclusions. A negative result constrains these implementations, not all affective geometries.

NRC word ratings describe words; they are not ratings of these videos or audience comments. The Gaussian similarity target is always positive and can encourage shared classifier directions. Its normalization and coefficient must be reported. Official CAD uses cumulative probabilities in the benchmark's class order; it is not a VAD distance. A CAD improvement alone cannot demonstrate fewer affectively distant errors. Any such claim needs an additional explicitly defined diagnostic using the sourced coordinates for every model, including the controls.

## Launch gate and commands

`smoke_emotion_3videos` now includes these four variants in addition to the 14 C/D/F variants. It fits only three official training clips and checks finite, normalized predictions, decreased KL, frame-order invariance, and selected-checkpoint reload. This smoke has not run on Snellius yet.

After committing, pushing, synchronizing, checking the live account/partition/budget, freezing a release, and verifying that smoke:

```bash
cd /scratch-shared/gmago/video2reaction/code
source configs/clusters/snellius.env
release=$(python3 scripts/freeze_release.py)
cd "$release"
python3 scripts/submit_vad.py
```

The launcher submits four individually runnable jobs, at most four concurrently. Each uses one MIG GPU with a 30-minute limit; the batch cap is 128 SBU at the previously verified rate. It reuses the completed visual cache and does not need the full emotion cache. Refresh resource availability before submission.
