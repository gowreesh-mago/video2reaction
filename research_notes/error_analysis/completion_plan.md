# Complete the requested analysis — 21 October 2026

Deadline requested: **22 October 2026, 06:00 Europe/Amsterdam (04:00 UTC)**.
Branch: `research/vad-dino-completion-oct21`.
Local delivery folder: `outputs/analysis/2026-10-21-completion/`.

## Checklist

- [x] Audit previous results: 308 prediction runs, all eleven benchmark metrics.
- [x] Locate video DINO principal-component plots, label-distribution plots, and actual saved three-coordinate VAD trajectories.
- [x] Locate DINO and VAD distribution differences, per-video errors, and training-defined rare-reaction analysis.
- [x] Check Snellius access and account `gusr133332`; no jobs running at the initial check.
- [x] Confirm missing experiment: actual VAD-to-DINO reconstruction.
- [x] Implement projection experiment and controls with validation-only parameter selection.
- [x] Pass 12 local numerical tests.
- [x] Pass the three-video cluster smoke test (28448335): all mappings, original-prediction reproduction, metrics and exports completed.
- [x] Queue three paired seeds (42, 43, 44): 28448337_0, 28448337_1, 28448337_2, dependent on smoke success.
- [x] All jobs completed, results reviewed, plots rendered and compact outputs synchronized locally.
- [x] Prior plots/reports and new results consolidated in the local delivery folder.
- [ ] Exact paired comparison with the paper's fine-tuned model: **blocked by missing predictions/checkpoint**.

## Question and experiment

Can a video's learned scene-level valence, arousal and dominance (VAD) coordinates reconstruct useful DINO visual features? Does using those reconstructed features with the original DINO reaction predictor preserve its reaction-distribution accuracy?

The inputs are three coordinates predicted by our existing pure-vision trajectory checkpoint. No descriptions, comments, text embeddings, or held-out reaction labels enter the reconstruction. Fixed NRC coordinates were already part of the original trajectory model. Reconstructing DINO cannot create information absent from VAD; this experiment tests how much useful information the three coordinates retain.

Each seed uses the corresponding existing trajectory checkpoint and DINO mean-pooling checkpoint. The selected trajectory condition is `on_vad_soft_tau010`, from the previous validation-selected study. Original checkpoints are frozen.

### New mappings

1. Linear ridge regression: three VAD inputs to 768 DINO dimensions.
2. Quadratic ridge regression: linear terms plus all squared and pairwise-product terms.
3. Radial-basis ridge regression: 128 centers taken from training coordinates, plus linear terms. A radial basis activates near its center in VAD space.
4. Shuffled-pair radial-basis control: training VAD trajectories are shuffled between clips, preserving each trajectory internally. This checks whether matched VAD/video information matters.

For each seed, fit on at most eight evenly spaced scenes per training clip, with equal total weight per clip. Standardization and centers use training data only. Ridge strengths are 0.0001, 0.01 and 1.0; radial-basis bandwidths are 0.5, 1 and 2 after standardization. Select each mapping on equal-clip validation reconstruction mean squared error using the same scene-sampling rule. Freeze all choices before evaluating every test scene.

This is 24 candidate fits per seed, 72 in total, producing 12 selected decoders across three seeds. Candidate fits are parameter selection, not 72 independent scientific results.

### Controls and evaluation

- Original DINO features and original saved predictions: verify runtime and exact feature/head wiring.
- Constant training-mean feature reconstruction.
- DINO reconstruction through three principal components: a general three-dimensional visual bottleneck without VAD restrictions.
- DINO reconstruction through 32 principal components: a higher-capacity information-retention reference.
- Original VAD reaction predictions for context.

Report reconstruction mean squared error, explained error reduction relative to the training-mean predictor, cosine similarity both before and after subtracting the training mean, and within-test clip retrieval. Raw cosine alone can be misleading if features share a large common mean.

Pass reconstructed scenes through the **unchanged original DINO head**, including its per-scene layer normalization and projection before pooling. Report all eleven benchmark metrics, each emotion's error/support, rare-reaction mass and recall, per-video errors, and movie-cluster bootstrap intervals for paired KL differences. No new reaction head is fitted, so failure can reflect both lost information and incompatibility with the frozen head. Success would support information retention, not a causal interpretation of scene emotion or comment attribution.

## Optional projected-emotion decoder — implemented, not launched

Automatic approval review rejected launching this extension because it judged DINO-space emotion prototypes outside the user's clarified VAD-to-DINO reconstruction request. No smoke or full jobs were submitted for this extension. A separate optional approval question is pending; the authorized reconstruction analysis proceeds independently.

The proposed extension would place the 21 fixed NRC emotion coordinates into DINO space and test that use directly: compare each original scene DINO feature with these projected emotion anchors, convert distances to probabilities, and average those probabilities across scenes. This uses no trained reaction classifier.

Evaluate the four fitted maps (including the shuffled control), two distances (mean squared distance and cosine distance after subtracting the training DINO mean), and uniform or actual-duration pooling. Select temperature from 0.003, 0.01, 0.03, 0.1, 0.3, 1 and 3 using validation KL. Freeze temperature and the preferred non-shuffled model before test scoring. If approved, this would produce 16 variants per seed, 48 across three seeds; temperature candidates are calibration choices, not independent results. Record which fixed emotion points fall outside the learned training VAD bounding box, since applying the reverse map there requires extrapolation. All eleven metrics and rare-class diagnostics are retained.

## Paper comparison dependency

The paper is [Video2Reaction, arXiv:2607.06875v1](https://arxiv.org/html/2607.06875v1). Its fine-tuned LLaVA/Qwen outputs are not our DINO or visual-description runs. Published aggregate scores do not provide the paired predictions needed for distribution/error plots.

On 21 October, the [lab's Hugging Face account](https://huggingface.co/infofusionlab) lists no public model checkpoints. The [dataset release](https://huggingface.co/datasets/infofusionlab/Video2Reaction/tree/main) lists metadata and pretrained feature files, not the fine-tuned predictor. Searches of the Snellius project, original extracted data tree, model directory, and Hugging Face cache found no fine-tuned adapter. A cached base Qwen model is not the paper's fine-tuned checkpoint. The user has been asked for the missing path. An independent retraining would be a new reproduction, not the paper's saved predictions.

## Runtime and storage

The shared uv PyTorch 2.8.0 environment fails because `torch/bin/torch_shm_manager` is absent. A separate uv CPU environment was created for checking the code. The first CPU submission was rejected because the account has no CPU-product budget. The first three-video smoke test was accepted on the funded `gpu_mig` partition as job `28448175`, then cancelled while pending. The CUDA smoke is `28448335`; the dependent three-seed array is `28448337`, on `gpu_h100`. H100 had the earliest tested start estimate. Shared jobs cannot combine the MIG, A100 and H100 partitions, so the submitted command explicitly selects H100. A separate CUDA environment, `envs/projection-gpu-20261021`, accelerates the full runs on H100. Preserve the existing environment and checkpoints. Full jobs have a one-hour wall limit, and smoke tests have a 20-minute override. Jobs run independently of the Mac after submission. Scheduler delays remain outside our control.

All full feature arrays and train/validation coordinate exports stay on Snellius. Only plots, reports, selected model parameters, compact prediction arrays, logs and manifests are synchronized to the Mac.
