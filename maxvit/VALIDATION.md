# Integration validation

Checks performed on 2026-11-03; these are integration checks, not a new CV score
or a claim of bitwise-identical retraining on every supported machine.

## Results

- **20 tests passed** using `python -m unittest discover -s tests -v`: 11 new
  MaxViT tests plus all nine pre-existing teammate tests.
- Shared L85 targets: maximum absolute difference **0** against the original
  supplied `labels_1_85_conf1.csv`; fold assignment disagreements **0**.
  Confidence fields are ignored. Official gold overrides match.
- Shared geometry: spacing and direction match; center coordinates differ by
  at most **5.7e-14 pixels**, from CSV floating-point serialization only.
- Original versus portable datasets: identical ordered series membership and
  targets for train/validation across **all five folds**. Sampled tensors in
  axial, coronal and sagittal planes match exactly for both modes with matched
  augmentation RNG (30 sample comparisons).
- Actual fold-0 preflight: **3,525 training studies / 11,195 FS series** and
  **882 validation studies / 2,813 FS series**. Actual sampler gives 220
  optimizer updates/epoch, 4,400 total, 440 warmup. Across folds there are
  14,008 eligible FS series. Incomplete series coverage is rejected.
- Fresh DICOM conversion versus original PNGs: exact filenames and pixels for
  one axial (18 slices), coronal (16) and sagittal (22) series.
- All **five final EMA checkpoints** load strictly: 635 state tensors each,
  including BN buffers; no missing/unexpected keys.
- Exported final fold-0 tensors match the original deployment file exactly
  (635/635 tensors). The pinned ImageNet downloader and offline initialization
  path were also exercised successfully.
- Full-depth, full-resolution forward passes on an RTX A6000: finite twelve-label
  logits and losses for all three planes under BF16. Checkpointed backward also
  produced finite gradients.
- Actual trainer dry-run: two batch-16 optimizer updates, encoder chunk 256,
  BF16, gradient checkpointing, ordinary EMA and final checkpoint saving passed.
  This diagnostic used random initialization, one GPU with the `auto` strategy,
  disabled validation and no SyncBN; it was not a full reproduction training run.
  Its shortened LR schedule is not used by the published recipe. MLflow 3.13
  required `MLFLOW_ALLOW_FILE_STORE=true` for the temporary dry-run file store;
  normal documented training uses the required HTTP tracking server instead.
- Repository timm **1.0.25** versus original **1.0.27**: identical BF16 logits
  (max absolute difference 0) for the tested sagittal and column-routing inputs
  using the same final fold-0 checkpoint.
- New integration Python passes Ruff; `git diff --check` passes. Core numerical
  source remains unmodified; only converter default paths were made portable.
- Combined root and optional MaxViT requirements resolve successfully for
  Python 3.12. The optional requirements use `mlflow-skinny`, not full MLflow,
  so they do not downgrade/conflict with the root's pandas 3.0.6 pin. A separate
  server environment is documented.

## Environment and limits

Tests used Python 3.11.5, torch 2.6.0+cu124, torchvision 0.21.0+cu124,
Lightning 2.6.5, and an RTX A6000. The entire NVIDIA 25.09 container was not
recreated locally. timm 1.0.25 compatibility was tested as described above,
not inferred solely from checkpoint key names. Training on the container's
different CUDA/PyTorch build can change numerical results.

No full CV training, inference submission, or leaderboard evaluation was run
for this integration. No private raw images, reports, secrets, or checkpoints
are included in the PR. Generated preparation files/caches and downloaded
ImageNet weights remain ignored under `assets/`; original upstream model code
and requirements are untouched.
