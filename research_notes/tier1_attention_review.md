# Tier 1 preparation: reaction attention and frame emotion evidence

Prepared on 2026-10-01. This is a code and hypothesis review; it contains no new benchmark metric values. The completed Tier 0 LaTeX report remains a snapshot of the five evaluated baselines. E jobs `27438329` and `27438330` completed successfully at the scheduler level; their artifacts still need collection.

## E: what changes

`e_reaction_query` uses the existing frozen SigLIP2 cache, a shared frame projection, 21 learned queries, a softmax over valid frames for each query, and the existing head. Class c uses the c-th head output evaluated on its own pooled representation. It retains Tier 0's split hashes, seed 42, hidden width 128, optimizer, loss, batch size, and validation KL checkpoint selection.

`e_shared_query_control` computes the same 21 attention distributions, averages them, and supplies this one shared distribution to every reaction. All queries remain active in the computation. Both variants have exactly the same state-dictionary shapes, parameter count, initialization, and training settings. Only the use of separate versus shared attention changes. Averaging query vectors directly would collapse the control to a single effective query; averaging their softmax distributions avoids that linear collapse.

Both variants have 171,797 trainable parameters, compared with B1's 169,109. This is why E versus B1 alone cannot isolate reaction-specific attention. E versus the shared control is the primary comparison.

## Does the implementation test the hypothesis?

Yes, within a narrow scope: it tests whether retaining class-specific attention distributions improves prediction relative to learned salience shared across classes. It does not require the external emotion classifier and can run from the completed visual feature cache.

Predeclared interpretation:

- Lower E test KL than the shared control, supported by paired movie-cluster uncertainty, supports this implementation of class-specific pooling.
- Similar E/shared performance with both better than B1 supports learned salience but gives no evidence that distinct reaction maps help.
- Neither improving on B1 argues against these query models under the fixed features and training recipe. It does not reject sparse evidence in general.
- Distinct-looking attention maps alone do not establish predictive benefit or identify the frames that caused human reactions.

The shared model can learn different internal query maps before averaging. Its final maps are identical across classes by construction; this is a deliberate control, not a bug. Equal parameter count does not guarantee equal optimization difficulty. Report validation curves and selected epochs alongside test comparisons.

Both models are permutation invariant. They cannot prove that temporal order or narrative understanding helps. The unchanged B2/order-control experiments and future global/peak comparisons address other parts of the search space.

## Review of diagnostic artifacts

Full validation/test predictions can now save `attention.npz` and `attention_frames.json`. Weights have shape (total valid frames, 21), with sample IDs, offsets, class order, attention entropy, highest-weight frame positions, and the generalized Jensen–Shannon divergence among class maps. The JSON links each row to the original keyframe index and available timestamp fields. It includes the NPZ checksum. Raw images are not copied.

The recorder rejects missing/duplicate samples, mismatched frame inventories, padded attention mass, nonfinite or unnormalized weights, and shuffled-frame exports. Synthetic round-trip tests compare saved attention against direct model calls, including variable-length videos spanning multiple batches. The three-video GPU smoke exercises the same recorder used by full evaluation.

## Practical model for C/D/F

The [official EmotionCLIP release](https://github.com/Xeaver/EmotionCLIP) provides pretrained representations and a linear-probe evaluation pipeline, rather than an immediately specified eight-class image predictor. The [EmoSet repository](https://github.com/JingyuanYY/EmoSet) documents the dataset but does not provide a classifier in its main file listing.

The [EmoEditor authors](https://github.com/ZhangLab-DeepNeuroCogLab/EmoEditor) provide a small ResNet-18 emotion checkpoint. [Their paper, section 3.1](https://arxiv.org/html/2403.08255v3#S3.SS1), states that this predictor was trained on EmoSet. Its eight-way head, input transformation, and alphabetical label order are explicit in the released code. The checkpoint was downloaded directly to Snellius and hashed; the pinned source revision and checksum are in [the model manifest](../assets/frame_emotion_model.json).

Loading and real-image inference remain unverified. Do not treat the download as a completed C experiment. Before peak experiments: validate the adapter on the same three training clips, source exact NRC entries for all eight labels, declare the neutral reference, and align the emotion cache to existing visual-cache image hashes. Keep temperature and scoring definitions fixed without consulting test results.

## Status

- Implemented E and its control, configs, individual SLURM scripts, and full attention export.
- All 27 local synthetic tests passed in conda `torch` (7.55 seconds). Warnings are confined to undefined-label metrics in the original reference implementation exercised by the equivalence test.
- Source `4fd4068` is pushed and synchronized. Three-video cluster regression smoke `27438252` completed with exit `0:0` in 48 seconds, from frozen release `4fd4068f2733-b298ef71977d`. All 27 tests and five variants passed, including the shared-query control and attention recorder. Verified through `sacct` and the saved smoke log at 2026-10-01 01:20:54 UTC.
- The bounded E batch contains exactly two independent MIG jobs, each capped at 30 minutes (combined maximum 64 SBU). `scripts/submit_attention.py` reuses the locked/idempotent submission logic and runs both scheduler preflights before submitting either job. Two new local scheduler tests pass. No predictor or evaluation changes were made after the successful cluster smoke.
- Submitted E source is `a89ab3b`. `e_reaction_query_27438329` and `e_shared_query_control_27438330` were both confirmed `COMPLETED`, exit `0:0`, elapsed `00:01:10` by `sacct` at 2026-10-01 01:36:25 UTC. Subsequent SSH connections were refused, so metrics and attention artifacts remain uncollected. Do not infer performance from scheduler success.
- C/D/F remain unlaunched. `scripts/setup_vad.py` now extracts exact NRC entries for the eight coarse labels; the original archive checksum matches the one used for the 21 reaction labels.

Submission from a synchronized code directory on Snellius:

```bash
source configs/clusters/snellius.env
release=$(python3 scripts/freeze_release.py)
cd "$release"
python3 scripts/submit_attention.py
```
