# Repository audit

Audited upstream commit `0da6060` on 2026-09-30, before implementation changes.
Work branch: `research/video2reaction-experiments`.

## Existing interfaces

- `src/dataset.py`: `Video2Reaction`, multimodal `collate_fn`, taxonomy and NRC VA ordering. Metadata is a JSON object keyed by video ID. Targets are under `reaction_outcome.reaction_distribution`; descriptions, movie name, and genre are separate inputs.
- `src/model.py`: temporal convolution plus context attention in `MultimodalReactionPredictor`. Forward accepts acoustic audio, semantic audio, visual sequence, description embedding, and genre.
- `src/baselines/cubemlp.py`: CubeMLP; `src/tcn.py`: temporal convolution and attention modules.
- `scripts/train_cubemlp.py`: argparse plus external JSON model config, train/validation/test and checkpoint evaluation. Uses cached feature tensors or per-video NumPy files.
- `scripts/run_ldl_baselines.py`: classical LDL algorithms through python-ldl/TensorFlow; concatenates configurable modalities.
- `scripts/vlm_classification.py`: pretrained VLM evaluation. `scripts/finetune_vlm_classification.py`: LoRA training with Hugging Face Trainer, next-token letter logits over 21 classes; text/visual ablation switches.
- `src/metrics.py`: reference distribution metrics, CAD, MRR, ground-truth top-1 probability error, weighted multilabel Top-k F1, extra threshold metrics.
- `data_preprocessing/`: collection, scene extraction, visual/audio extraction. Frame `index.csv` stores scene number, timecodes, and frame numbers; images use one-based zero-padded scene numbers.
- `src/automatic_annotation/`: comment annotation pipeline; not part of the new experiments.

## Paths, environment, outputs

Existing feature convention: `<root>/visual/<encoder>/<video_id>/full.npy` (sequence) or `pooler_mean.npy`; analogous audio directories. Description embeddings are computed with BERT, even when visual-only features might suffice. ViViT full tokens are spatially averaged into 16 temporal chunks.

Dependencies are unpinned in `reaction-video-venv-requirements.txt`; some scripts additionally need TensorFlow, IPython, and python-ldl. No pyproject, lockfile, checked-in model config, experiment registry, or SLURM scripts existed. The local `data/nrc-python/` directory was empty.

CubeMLP saves model/optimizer/scheduler dictionaries and validation metrics in `.pth`, plus plots/results. VLM saves adapters and processor under `final_model` and reports to mandatory W&B. No common per-sample prediction schema or run manifest existed.

## Confirmed issues

1. The fixed 21-label list is a bare expression; `REACTION_CLASSES` is used without assignment. Import fails.
2. Dataset import opens `./data/nrc-python/nrc_vad.json`, which is absent and depends on the working directory.
3. CubeMLP applies `log_softmax` to already softmaxed predictions for KL; cross-entropy also receives probabilities. New experiments must use logits correctly; legacy behavior is documented rather than silently changing old result definitions.
4. CubeMLP's in-memory `best_model = model.state_dict()` can track later parameter updates. New runs reload an on-disk best checkpoint.
5. Reference JS divides by sample count twice. Preserve official required metrics; use a separately named mathematically correct JS training loss.
6. `src/utils.py` already exists. A new `src/utils/` package could shadow it, so put the locked registry in `src/experiments/registry.py`.
7. Legacy loaders can discard missing-frame samples. New runs must fail on missing required samples and preserve split membership.

## Extension points

Add an isolated `src/experiments/` package and root `train.py --config ...`. Reuse the metadata schema and exact taxonomy; keep old model and script interfaces. Cache frozen frame features once, then compare pooling, temporal, loss, VAD, peak, and query modules with a common optimizer/head/evaluation path. Use SLURM job dependencies for cache preparation and a locked shared registry plus per-run artifacts.

## Initial fixes

The legacy loader now imports the fixed class list and resolves NRC coordinates through `V2R_VAD_FILE` or a repository-relative path. `scripts/setup_vad.py` obtains the author's NRC v1 data, verifies exact entries for all 21 labels and checks that their VA order matches upstream. No coordinates are approximated. Generated lexicon data stays private on the cluster because NRC's distribution terms exclude redistribution; the fetch/build script and provenance format are versioned. The earlier loss and checkpoint issues are recorded for later legacy training repair; the new smoke predictors use logits correctly and verify on-disk reload/resume.
