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

For legacy imports, provision the private NRC VAD asset once:

```bash
uv run --frozen python scripts/setup_vad.py --output "$V2R_ROOT/data/lexicons" --archive "$V2R_ROOT/cache/nrc-vad.zip"
```

`V2R_VAD_FILE` selects that file. All 21 exact NRC v1 entries exist and reproduce the original class order. Generated coordinates and their source README remain on the cluster; they are excluded from Git. No VAD training experiment is part of this smoke check.

The author website returned HTTP 406 from Snellius; the original archive was downloaded on the Mac and copied to the private cluster cache. Only this lexical resource was transferred from the Mac. Omitting `--archive` downloads directly where the website permits it.

Do not synchronize over code while a job is using it. Subsequent full experiments should use a frozen release directory or the recorded revision. Collect summaries/logs back to the Mac; keep frames, split data, feature tensors and checkpoints on the cluster. Scratch is not a durable archive.
