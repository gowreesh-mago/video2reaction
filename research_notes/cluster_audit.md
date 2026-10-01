# Cluster access and budget

## Latest resource check

**2026-10-01 00:38 UTC:** `budget-overview` reports **87,861:20 SBU** left for dispatch and submission on `gusr133332`, with no active or queued jobs. The completed Tier 0/cache job IDs still show `COMPLETED`, exit `0:0`. Home uses 40.4098% of its 200 GiB quota (about 119.18 GiB available); scratch uses 0.1124% of its 8 TiB quota. Home inode usage is 699,411/1,000,000; scratch inode usage is 1.2616% of 3,000,000. Features occupy about 2.1 GiB and outputs about 70 MiB. [Structured snapshot](cluster_status_20261001.json)

The original access/storage audit below is preserved with its earlier measurements.

## Initial audit

Observed 2026-09-30. SSH aliases from the user's config were tested; account associations and partition allow-lists were inspected. No jobs have been submitted at this audit checkpoint.

| Cluster | Verified account(s) | Verified candidate partitions |
|---|---|---|
| Snellius | `gusr133332` | `gpu_a100`, `gpu_h100`, `gpu_mig`, `gpu_vis`, `staging`, `cbuild` |
| UvA FNWI, `ivi-h1.science.uva.nl` | `linuxusers`, `havausers`, `ceesusers`, `all6000users` | `all`, `hava`, `cees`, `all6000` |

Snellius active budget `2307089_26/L2/436`: initial 100,000 SBU; used 11,851:10; remaining **88,148:49 SBU**; expires 2026-12-31; no excess usage. The listed budget covers GPU products and staging/build, not general CPU partitions. Visible partitions alone are not proof of budget entitlement.

Snellius home quota: 200 GiB, 39.975% used (~79.95 GiB), ~120.05 GiB remaining; 693,833 of 1,000,000 inodes used. Scratch quota: 8 TiB (10 TiB hard limit), usage rounded to 0%, ~732 of 3,000,000 inodes. Store new environments/features/results on scratch to avoid home inode pressure. Scratch is temporary storage; collect durable results.

No user jobs on Snellius at audit time. UvA has an existing interactive job `243781` on `hava` / `ivi-cn008`; leave it untouched. UvA monetary/SBU budget was not reported by the inspected SLURM association data.

Snellius A100 costs 128 SBU per GPU-hour, H100 192, MIG 64 according to live TRES billing weights. One A100 defaults to 18 CPUs/120 GiB; request small memory/CPU amounts but expect the GPU billing floor. Partition max wall time is five days except visualization (one day). Inspected normal/GPU QOS records reported no per-user job/submission/TRES limits; this does not imply unlimited practical capacity. Start bounded and inspect the queue.

Data found: 22,089,639,938-byte `~/video2reaction/key_frames.zip`; extracted frames under `~/video2reaction/scratch4/workspace/sidongzhang_umass_edu-v2r/v2r_data/youtube_video/key_frames` (11,738 clip directories). Dataset tree occupies ~44 GiB. No JSON split files or bundled feature arrays found there in the initial inventory. Fetch original metadata from the official dataset at pinned revision `578d1423f89f1a7b52471b01e78770e08f0a226c`; never create replacement splits.

Existing SigLIP2 SO400M/384 checkpoint revision `e8e487298228002f3d8a82e0cd5c8ea9c567f57f` is cached on Snellius. Use it for frozen semantic features if the GPU smoke check passes.

Sources: live commands above; [SURF budget accounting](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/300974118/Getting%2Baccount%2Band%2Bbudget%2Binformation) notes central budget updates can lag by about a day. Jobs must pass `sbatch --test-only` on a verified account/partition before submission.
