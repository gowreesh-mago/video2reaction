# Three-video smoke report

Status: **passed**. This is the requested infrastructure checkpoint; full benchmark experiments remain pending.

## Execution

- SLURM job: **27381153**, `COMPLETED`, exit **0:0**, elapsed **00:08:04**.
- Cluster: Snellius; account `gusr133332`; partition `gpu_mig`; node `gcn2`.
- GPU: NVIDIA A100-SXM4-40GB MIG 3g.20gb, 19.75 GiB visible, bf16 supported.
- Environment: **uv**, Python **3.11.16**, PyTorch **2.8.0+cu128**, Transformers **4.57.1**. No conda on the cluster.
- Tested Git revision: `08b8c30` on `research/video2reaction-experiments` (pushed before submission). Source snapshot SHA-256: `727a588d3f56952f4a4247bb056af55a46f6146f8c65512d989126ce70cf3f77`.
- Inputs: exactly three official training videos: `4nSkJZ3i2-g`, `eiqBbLVXbQg`, `iKp5ARBBpyc`; eight chronological frames per video (**24 total**). The train split hash is pinned in the config.
- Frozen encoder: cached `google/siglip2-so400m-patch14-384` at revision `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`; output dimensions 8 x 1152 per clip.
- Frame encoding time after model loading: **46.89 seconds**.
- SLURM allocation: one MIG device, nine billed CPU cores, 24 GiB host memory. Approximate charge from allocation and elapsed time: **8.60 SBU**; central accounting may update later.

## Checks

- Required dependency imports pass in the uv environment. Legacy `src.dataset` and `src.metrics` imports pass after fixing the class assignment and provisioning sourced NRC coordinates.
- **14 mock tests pass locally** in the user-approved conda `torch` environment and **14 pass on the cluster**. Tests cover reference metric agreement, hand-calculated metrics, finite loss gradients, CE/KL gradient equivalence, padding masks, invalid inputs, and 16 concurrent registry writers.
- Each model ran 25 optimizer steps on the same three examples. Loss decreased, predictions were finite and normalized, checkpoint reloaded predictions matched exactly, and a resumed optimizer step remained finite.
- Per-variant checkpoints, predictions, metrics, loss histories, per-class diagnostics and query attention were saved. Artifact inspection confirmed all prediction arrays are 3 x 21 and row sums are one within tolerance.
- Shared registry status is `completed`; all run-specific artifacts and stdout are present.

| Variant | Objective | Initial training loss | Final training loss | Reload/resume |
|---|---|---:|---:|---|
| meanpool | KL | 1.643282 | 0.028483 | passed |
| temporal_transformer | KL | 1.745008 | 0.012147 | passed |
| distribution_loss | KL + cosine + ranking | 1.782490 | 0.024368 | passed |
| reaction_queries | KL | 1.643298 | 0.033656 | passed |

**These are training-set smoke results, not benchmark scores or evidence that one architecture is better.** No full training jobs, hyperparameter search, or test-set evaluation were launched.

## Findings and fixes

1. Original loader used an unassigned class list and a missing working-directory-relative NRC file. The fixed loader uses the explicit upstream class order and a configurable asset path. All 21 exact NRC v1 entries are available; their VA sorting matches the original order. No VAD values were invented.
2. The author site returned HTTP 406 to Python download requests on Snellius. A downloaded original reference archive was copied to the private cluster cache; the generated coordinates are excluded from Git. This lexical resource is separate from Video2Reaction.
3. Shared-filesystem cold reads dominated startup. The job completed after dependency files were read in parallel on the allocated node (three diagnostic srun steps, all successful). `warm_imports.sh` now automates the same dependency-prefetch approach for subsequent runs; its shell syntax was checked. The model/training code is unchanged from the successful run.
4. A slower login-node import check completed successfully after the GPU job was submitted. The new submission lock correctly detected the existing job and prevented a duplicate.
5. sklearn warnings in the reference-metric tests concern undefined scores for absent classes in synthetic data. Numerical behavior agrees with the original implementation.

## Storage and access

Project root: `/scratch-shared/gmago/video2reaction`.

- `code/`: synchronized branch and `.venv`.
- `data/metadata/`: pinned official splits.
- `data/key_frames`: link to existing frames under `/gpfs/home3/gmago/video2reaction/scratch4/workspace/sidongzhang_umass_edu-v2r/v2r_data/youtube_video/key_frames`.
- `outputs/smoke/smoke_3videos_27381153/`: run artifacts.
- `logs/slurm/smoke-27381153.out` and `logs/smoke/smoke_3videos_27381153/stdout.log`: logs.
- `results/experiment_registry.json`: locked shared registry.

Only aggregate audit and smoke-result JSON were collected on the Mac. No Video2Reaction frames, split metadata, feature tensors, or checkpoints were copied to the Mac.

Verified access: Snellius GPU A100/H100/MIG and staging/build products; UvA partitions all/hava/cees/all6000 through matching accounts. Only Snellius received a smoke job. Initial remaining budget was 88,148:49 SBU; home had about 120.05 GiB free under a 200 GiB quota, and scratch was almost empty under an 8 TiB quota before setup.

## Dataset caveats

The audit checked all 10,348 split clips and all 455,226 referenced image paths: none missing. Observed frame counts are 15–176 (median 39). There are no duplicate video IDs across splits, but **95.94% of test clips share a movie with training**. Keep official splits, and report movie-disjoint diagnostics separately in later experiments. Full details: `dataset_audit.md` and `metric_audit.md`.

## Reproduce

On the Mac, run `bash scripts/sync_cluster.sh`. Then on Snellius:

```bash
cd /scratch-shared/gmago/video2reaction/code
source configs/clusters/snellius.env
export PATH="$HOME/.local/bin:$PATH"
bash scripts/setup_env.sh
uv run --frozen python scripts/setup_vad.py --output "$V2R_ROOT/data/lexicons" --archive "$V2R_ROOT/cache/nrc-vad.zip"
bash scripts/submit_smoke.sh
bash scripts/status_all.sh
```

The launcher prevents another submission of an already tested source snapshot. Inspect its recorded job rather than accidentally duplicating it. Full experiments, the final research comparison and the peak/context/VAD hypotheses remain for the next phase.
