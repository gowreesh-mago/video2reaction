# Snellius project layout

Project root: `/scratch-shared/gmago/video2reaction`

```text
video2reaction/
  code/                         rsync copy of the research branch
    .venv/                      uv Python 3.11 environment (Linux/CUDA)
    code_version.json           Git revision and source checksums
  data/
    metadata/                   pinned official train/val/test JSON
    key_frames -> /gpfs/home3/gmago/video2reaction/.../key_frames
    archives/key_frames.zip -> /gpfs/home3/gmago/video2reaction/key_frames.zip
    features/                   future shared frozen features
  cache/uv/                     uv download cache
  releases/<commit>-<hash>/    frozen tracked source; .venv links to existing uv environment
  outputs/experiments/<name_jobid>/  resolved config, provenance, resume/selected checkpoints, val/test results
  logs/experiments/<name_jobid>.log  full-run stdout/stderr
  outputs/smoke/<run_name>/     config, provenance, predictions, checkpoints, metrics
  logs/setup/environment.log   uv installation and import checks
  logs/slurm/smoke-<jobid>.out  scheduler stdout/stderr
  logs/smoke/<run_name>/stdout.log
  results/
    dataset_audit.json
    experiment_registry.json   shared, file-locked run status
```

Original dataset files are preserved; links avoid duplicating 44 GiB. The temporary audit download directory `/scratch-shared/gmago/video2reaction-data` contains only metadata and audit JSON and is no longer the active data path. Existing home model cache is reused through `HF_HOME`; no model or dataset is transferred to the Mac.

## Commands

From the Mac, synchronize code without deleting remote outputs:

```bash
bash scripts/sync_cluster.sh
```

On Snellius:

```bash
cd /scratch-shared/gmago/video2reaction/code
source configs/clusters/snellius.env
export PATH="$HOME/.local/bin:$PATH"
bash scripts/setup_env.sh
bash scripts/submit_smoke.sh
bash scripts/status_all.sh
```

The smoke job uses exactly three explicit official **training** videos, at most eight chronological frames per video, one A100 MIG slice (20 GiB), and a ten-minute limit. The authorized MIG partition had the earliest estimated start when compared with A100 and H100. It checks frozen SigLIP2 encoding, four small predictors, 25 optimizer steps each, finite normalized predictions, lower training losses, exact checkpoint reload and a resumed optimizer update. Metrics on these training examples do not measure generalization.

The first run needed dependency prefetching because cold imports on shared storage took several minutes. `scripts/warm_imports.sh` now automates those verified read commands at job startup; set `V2R_WARM_IMPORTS=0` to skip on a warm node. This reads Python dependency files into the node's filesystem cache, not dataset files. The submission script locks and records one job per synchronized source snapshot to prevent duplicate launches.

For legacy imports, provision the private NRC VAD asset once:

```bash
uv run --frozen python scripts/setup_vad.py --output "$V2R_ROOT/data/lexicons" --archive "$V2R_ROOT/cache/nrc-vad.zip"
```

`V2R_VAD_FILE` selects that file. All 21 exact NRC v1 entries exist and reproduce the original class order. Generated coordinates and their source README remain on the cluster; they are excluded from Git. No VAD training experiment is part of this smoke check.

The author website returned HTTP 406 from Snellius; the original archive was downloaded on the Mac and copied to the private cluster cache. Only this lexical resource was transferred from the Mac. Omitting `--archive` downloads directly where the website permits it.

Do not synchronize over code while a job is using it. Subsequent full experiments should use a frozen release directory or the recorded revision. Collect summaries/logs back to the Mac; keep frames, split data, feature tensors and checkpoints on the cluster. Scratch is not a durable archive.

## Reviewed first batch

`bash scripts/submit_all.sh` freezes the committed source, validates all six resource requests, and submits the feature cache, B0, B1, B2, B2 set control, and A5 once per release. The four learned predictors depend on successful feature extraction. The cache encodes all 455,226 indexed frames and retains image-byte hashes and frame provenance. Training compares fixed seed 42, full keyframes, batch size 64, AdamW at .001, up to 50 epochs, and validation-KL patience 8.

Feature extraction requests one A100 for at most 24 hours. B0 requests a MIG slice for 30 minutes; each learned predictor requests a MIG slice for two hours. The allocation ceiling is approximately 3,616 SBU (3,072 + 32 + 512), excluding regression smoke; actual billing follows elapsed allocations. No general CPU partition is assumed accessible for B0.

Jobs use `uv run --frozen --no-sync` against the already provisioned environment. Do not change that environment while runs are active. A failed learned run can resume into a new attempt directory by exporting `V2R_RESUME_FROM` to its `last.pt` when submitting the same experiment from the same frozen release. The checkpoint restores epoch, selected weights, optimizer, RNG, and patience; changed config/source identities are rejected. Cache extraction resumes from its last flushed frame cursor.

Collect or reconcile results on the cluster:

```bash
source configs/clusters/snellius.env
python3 scripts/collect_results.py --reconcile
```
