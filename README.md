# RSNA Knee Abnormality Detection — 9th place

Training and label code for the 9th place knee MRI solution.  

Training source for teammate components `v307`, `v310`, `v311`, and `v313` is included below. The `l185conf1` dual-view MaxViT FS router is documented in [maxvit/README.md](maxvit/README.md), including preprocessing, the exact training recipe, final EMA exports and OOF pooling. Their inference is already embedded in the inference notebook.

## Environment

Use the NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:25.09-py3` and install this repository's requirements into that container. Keep the image's PyTorch and torchvision builds. Python 3.12 is required by the retained prompt sources.

```bash
docker run --gpus all --ipc=host --shm-size=64g -it \
  -v "$PWD:/workspace/kaggle-knee-9th-place" \
  -w /workspace/kaggle-knee-9th-place \
  nvcr.io/nvidia/pytorch:25.09-py3 bash
python -m pip install -r requirements.txt
python -m pip check
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=8
```

The teammate `v307`/`v310`/`v311`/`v313` pipeline uses this same environment; it does
not need a second requirements file. Keep the container's CUDA-enabled `torch` and
`torchvision` instead of reinstalling them from pip. The existing `requirements.txt`
supplies timm, NumPy/pandas, OpenCV, pydicom and its JPEG/JPEG-2000 plugins,
safetensors, scikit-learn, and Hugging Face Hub support.

Verify the GPU runtime before starting a multi-hour run:

```bash
python - <<'PY'
import torch, timm, cv2, pydicom
assert torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__, "timm:", timm.__version__)
print("opencv:", cv2.__version__, "pydicom:", pydicom.__version__)
PY
```

All commands below run from the repository root. Training, validation, and transfer inference use `lab/main.py`. Inputs and regenerated caches live under `assets/`; checkpoints, OOFs, and logs live under `outputs/`. 

The isolated `l185conf1` MaxViT component instead uses `python -m maxvit.cli`
and additional dependencies from `maxvit/requirements.txt`. It reuses the same
bundled L85 labels/folds and frozen geometry, but has its own PNG preprocessing
and single-GPU training path; do not substitute the other models' image caches.

Slow-R50 recipes use four GPUs and the retained `8_*` ConvNeXt recipes use eight, with batch size two per GPU for DDP training. Each `v307`/`v310` fold and each `v311`/`v313` fullfit member uses one GPU with study batch two and two-step accumulation.

## Data, downloads, and fixed folds

The ensembled study-label table is `assets/data/labels_1_85.csv`; the process for populating the labels is described below. The label table contains all 4,407 study identifiers, twelve raw pseudo-label probabilities, `fold`, `fold_group`, `is_gold`. Fold assignments include the original gold-study overrides. `fold_group` identifies studies sharing a report; each group stays in one fold. 

Raw probabilities on the 58 gold rows are retained for provenance. Preparation joins official `train.csv` and replaces **effective** `soft_*` targets with exact gold values. All derived label/training tables inherit the bundled splits. Confidence weighting is disabled in the retained competition recipes.

Download competition data after accepting its rules:

```bash
mkdir -p assets/data/rsna-knee-abnormality-detection
kaggle competitions download -c rsna-knee-abnormality-detection -p assets/data
unzip assets/data/rsna-knee-abnormality-detection.zip \
  -d assets/data/rsna-knee-abnormality-detection
```

Expected layout is `train.csv`, `train_series.csv`, `sample_submission.csv`, and `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm`, with corresponding test files. A scoring notebook receives the hidden test studies in this layout.

The teammate pipeline converts the Kaggle training DICOMs into a processed
study cache named `cached_shards`. The converter preserves its 12-slot selection,
right-knee canonicalization, 110 mm crop, p0.5/p99.5 normalization, 320 px resolution,
32 deterministic centers, and approximately 3 mm neighboring-slice triplets:

```bash
# Smoke test one study first.
python scripts/dicom_series_to_processed_npz.py --study-limit 1

# Full resumable CPU conversion.
python scripts/dicom_series_to_processed_npz.py --workers 16
```

By default it reads `assets/data/rsna-knee-abnormality-detection` and writes one
uncompressed `assets/data/cached_shards/<StudyInstanceUID>.npz` per study. Each archive
contains only filled slots: `imgs`, `smask`, `slot_names`, `slot_ids`, `series_uid`, and
`n_errors`; `manifest.json` records coverage and live-slot counts. Use `--input-root` or
`--output-dir` to override either location. Existing study files are validated and
skipped, so interrupted conversions can be resumed.

The `v307`/`v310`/`v311`/`v313` configs read this `cached_shards` format directly.
Christof's other configs continue to use their separate native all-slice caches and
their existing `prepare_knee`/`prepare_knee_cache` workflow.

Download pretrained initialization:

```bash
mkdir -p assets/weights/pretrained/pytorchvideo
curl -L https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOW_8x8_R50.pyth \
  -o assets/weights/pretrained/pytorchvideo/SLOW_8x8_R50.pyth
python - <<'PY'
from huggingface_hub import hf_hub_download
for name, revision in [
    ('convnext_tiny.fb_in22k_ft_in1k', 'bc48a87f119bae1b06418f5340bfa97ef75d4609'),
    ('convnext_small.fb_in22k_ft_in1k', 'fe6b358aa409414cdb522ae3d43a2c9922f7cc1b'),
    ('tf_efficientnetv2_s.in21k_ft_in1k', 'ea9abc143ea2b9d8e1ec1de277bce02149b9cf0e'),
]:
    hf_hub_download(repo_id='timm/' + name, filename='model.safetensors',
                    revision=revision, local_dir='assets/weights/pretrained/timm/' + name)
PY
```




## Labeling with the retained prompts

Some LLVM based prompts need image preparation. Run commands below to prepare image-labeling inputs. `labeling/prompts/` contains prompt text, schemas, formatting, and response parsers. There are no labeling services, provider clients, inference endpoints, or orchestration dependencies. `labeling.prompts.export` writes request JSONL for your own model runner and parses completed responses into probability CSVs. The shared labeling resources are bundled:

- [Host rubric](labeling/host_guide.md), copied from the [host discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343).
- [Teaching plate 1](labeling/host_reference_1.jpg) and [teaching plate 2](labeling/host_reference_2.jpg), attached by prompts 023/041.
- [Report vocabulary hints](labeling/parag_v18_lexicon.json), loaded by prompts 016/023.

The exporter resolves these files automatically. Run prompts 012, 016, 023, 036, 041, and 048 to produce the per-study probability CSVs used below; no frozen prompt-output downloads are provided.

| Input | Model/settings | Prompt and role |
|---|---|---|
| Full report | Claude Opus 5.5, medium | `prompt_012`: report-only probabilities |
| Full report plus vocabulary hints | Claude Opus 5.5, medium | `prompt_016`: report-only structured probabilities |
| Report, committed report assessment, MRI navigation | GPT-6.1 Sol, medium | `prompt_023`: report-first review; overview plus up to three navigation images |
| Full report | Gemini 4 Argon, low, temperature zero | `prompt_036`: graded report probabilities |
| Report, committed report assessment, MRI navigation | GPT-6.1 Sol, medium | `prompt_041`: prompt036 report-first review; overview plus up to six navigation images |
| Individual MRI slices; no report | Gemini 4 Argon, medium, temperature zero, maximum 8192 output tokens | `prompt_048`: image-only probabilities |

For example:

```bash
python -m labeling.prompts.export --prompt 012 \
  --input-csv assets/data/rsna-knee-abnormality-detection/train.csv \
  --output outputs/labeling/prompt012.requests.jsonl
```

Run each request once using the indicated model/settings and retain the unrounded returned values. 


## Composing L85

`Lxx` identifies a label recipe, not a requirement to train model `xx`.

```text
L26 = 0.5*Opus(prompt012) + 0.5*Opus(prompt016)
L51 = 0.5*L26 + 0.5*histogram_match(Sol(prompt023) -> L26)
L63 = 0.6*L51 + 0.2*Gemini_low(prompt036) + 0.2*Sol(prompt041)
L77 = L63, with Effusion/Synovitis/Baker's replaced by
      0.5*L63 + 0.5*Gemini_images(prompt048)
L85 = L77, with both menisci/all three OA compartments/Effusion replaced by
      0.5*L77 + 0.25*Andrija_v700 + 0.25*OA_map
```

Use the bundled `assets/data/labels_1_85.csv` to train the competition models; the formula above records its provenance. To regenerate the report/image-label branch through L77 for OA transfer mapping:

```bash
python -m lab.utils.prepare_knee_labels compose \
  --opus012 outputs/labeling/prompt012.csv --opus016 outputs/labeling/prompt016.csv \
  --sol-review outputs/labeling/prompt023.csv \
  --gemini-report outputs/labeling/prompt036.csv \
  --sol-review-six outputs/labeling/prompt041.csv \
  --gemini-images outputs/labeling/prompt048.csv
```

Outputs go to `outputs/labeling/composed/`; the committed L85 file is not overwritten.

## From L77 to L85 with two OAI vision models

Download knee MRI images and MOAKS annotations from the [OAI full-download portal](https://nda.nih.gov/oai/full_downloads). Train two separate vision models on these OAI images, differing in how the OAI annotations become competition-class probabilities:

1. **OAI vision agent mapped:** convert the OAI annotations to competition-class targets using the fixed rules in [OAI_LABEL_MAPPING.md](labeling/OAI_LABEL_MAPPING.md), then train a vision model directly on those mapped targets. Apply it to the competition MRIs to obtain the `Andrija_v700` predictions. The training config for this branch is not yet included in this repository.
2. **OAI vision learned mapping:** train a vision model on the native MOAKS categories using [cfg_18_7](lab/configs/cfg_18_7.py). Apply it to the competition MRIs with [cfg_18_transfer](lab/configs/cfg_18_transfer.py), then use [map_oai.py](lab/utils/map_oai.py) to learn a mapping from its MOAKS predictions to the twelve competition classes by optimizing against the L77 soft labels. Fit and apply the mapping out of fold to produce `OA_map`.

Use `cfg_18_7` for both OAI data preparation and training; [prepare_oai.py](lab/utils/prepare_oai.py) provides the baseline-label, baseline-cache, and longitudinal preparation stages. Train the learned-mapping vision model with:

```bash
torchrun --nproc_per_node=4 lab/main.py -C cfg_18_7 --fold 0
```

Combine the two models' competition-image predictions with L77:

```text
L85 = 0.5*L77 + 0.25*Andrija_v700 + 0.25*OA_map
```

Apply this blend to both menisci, all three OA compartments, and Effusion; **keep the remaining L77 targets unchanged**. The resulting L85 labels supervise the competition models. The bundled `labels_1_85.csv` already contains this blend.

## Image preparation and stepwise competition training

Using the official competition data, decode DICOMs to native uint16 NPZs, construct effective L85 targets, and build the eight-slot cache:

```bash
python -m lab.utils.prepare_knee --workers 32
python -m lab.utils.prepare_knee_labels base
python -m lab.utils.prepare_knee_cache --config cfg_1_85 --workers 16
```

Canonical slots are `AXI_FS,AXI_NFS,COR_FS,COR_NFS_FL,COR_NFS_ST,SAG_FS,SAG_NFS_FL,SAG_NFS_ST`. Select the longest series per slot; add up to four remaining series ordered by descending slice count and series UID. Use the official `Fluid_Sensitive` flag for the non-FS coronal/sagittal slot split, apply no laterality flips, normalize with p0.5/p99.5 over real selected slices, and retain slice/slot masks.

**1. Train `1_85` CV and collect final ordinary OOFs.**

```bash
for fold in 0 1 2 3 4; do torchrun --nproc_per_node=4 lab/main.py -C cfg_1_85 --fold "$fold"; done
python -m lab.utils.prepare_knee_labels oof --config cfg_1_85
python -m lab.utils.prepare_knee_labels fuse --source cfg_1_85
```

This produces `outputs/cfg_1_85/final_oof.csv`, `assets/data/loss_weights_1_85.csv`, and the `1_91` label/training tables. OOF collection checks all five held-out partitions, final history epochs, and checkpoint epochs. It expects one completed seed per fold; use `--run-root` for an isolated run directory.

```text
Y85 = effective L85 targets, including exact gold
P85 = final epoch-25 ordinary held-out probabilities
W85 = 1 - abs(Y85 - P85)
Y1_91 = 0.5*Y85 + 0.5*P85, restoring exact gold
```

W85 multiplies main and auxiliary BCE elementwise; retain the ordinary mean over all elements. Gold rows use this same weighting formula.

**2. Train five `1_91` fullfits.** Preserve twelve slots, 125 mm/288 px/48 slices, BatchNorm-preserving stage checkpointing, 25 epochs, and unweighted BCE without EMA. Validation uses the original L85 targets on fold zero.

```bash
python -m lab.utils.prepare_knee_cache --config cfg_1_91 --workers 16
for repeat in 1 2 3 4 5; do
  torchrun --nproc_per_node=4 lab/main.py -C cfg_1_91 --fold -1 --output_dir "$PWD/outputs/cfg_1_91/repeat$repeat"
done
```

**3. Train `8_29` CV using L85 and W85.** Use native twelve-slot 110 mm/320 px inputs, 32 centers and physical adjacent triplets approximately 3 mm apart, 35 epochs, coarse dropout, and EMA decay `0.999`.

```bash
python -m lab.utils.prepare_knee_cache --config cfg_8_29 --workers 16
for fold in 0 1 2 3 4; do torchrun --nproc_per_node=8 lab/main.py -C cfg_8_29 --fold "$fold"; done
python -m lab.utils.prepare_knee_labels oof --config cfg_8_29
python -m lab.utils.prepare_knee_labels fuse --source cfg_8_29
```

Use OOFs from this revised training run. The historical `8_29` checkpoint set was trained with different loss weights.

**4. Train five `8_31` fullfits.** Targets are `0.5*Y85 + 0.5*OOF_8_29_EMA`, with exact gold restored. Use unweighted BCE and final EMA at epoch 35. Inputs share the `1_91` 125 mm/288 px/depth-48 cache: choose 32 fixed centers `round(linspace(0,47,32))` and cached offsets `-1,0,+1`.

```bash
for repeat in 1 2 3 4 5; do
  torchrun --nproc_per_node=8 lab/main.py -C cfg_8_31 --fold -1 --output_dir "$PWD/outputs/cfg_8_31/repeat$repeat"
done
```

**5. Train `8_32` CV.** Use L85 and W85 directly, ConvNeXt-Small, the same native physical-triplet cache as `8_29`, 25 epochs, final EMA, and no coarse dropout.

```bash
for fold in 0 1 2 3 4; do torchrun --nproc_per_node=8 lab/main.py -C cfg_8_32 --fold "$fold"; done
```

Stages 2, 3, and 5 are independent after step 1. Step 4 follows step 3. This packaging task does not launch these full training runs.

## Teammate models v307, v310, v311, and v313

This repository contains the complete config-driven training path for both branches:

```text
v310 final epoch-12 EMA OOF -> v311 labels
v311 non-gold = 0.5*L85 + 0.5*v310_final_EMA_OOF

v307 final epoch-14 EMA OOF -> v313 labels
v313 non-gold = 0.5*L85 + 0.5*v307_final_EMA_OOF

gold rows remain unchanged in both raw label tables
```

The existing bundled `labels_1_85.csv` supplies v307/v310 supervision and canonical
folds. The v307/v310 OOFs, v311/v313 fused labels, and all effective training tables are
generated locally under ignored paths. Preparation replaces the effective targets of
all 58 gold rows with official truth. No new label CSV is committed.

The reproducible versions use the existing L85 labels, pinned standard timm ImageNet
initialization for the backbone, normal initialization for the depth mixer and
task-specific heads, and all available selected series. The v307/v310 OOFs and
v311/v313 fusion labels are regenerated locally.

A fresh clone needs only:

1. the files committed in this repository;
2. the official Kaggle competition download in the documented directory;
3. the two pinned standard timm weight files downloaded by the command above; and
4. generated caches/OOFs produced by the commands below.

No precomputed v307/v310 OOF is required.

### Single-GPU training profile

Run one fold or one fullfit member per GPU. The default study batch-two profile runs
on 24 GB cards such as an RTX 3090; it also runs unchanged on RTX 6000 Pro GPUs. Do not
combine GPUs into one DDP run for these four experiments; use additional GPUs to run
independent folds or members in parallel. Full-resolution forward/backward optimizer
steps for both backbones were smoke-tested on an RTX 3090.

| Setting | Value |
|---|---|
| Train/eval batch per GPU | 2 / 2 studies |
| Gradient accumulation | 2; effective train batch 4 |
| Precision | fp16 autocast with gradient scaling |
| Input | 12 slots, 32 centers/slot, 3-channel approximately 3 mm triplets |
| Geometry | 110 mm crop, 320 × 320 pixels |
| Tensor memory format | contiguous; no model-wide channels-last conversion |
| Optimizer | AdamW |
| Backbone/head LR | `1.5e-4` / `1.0e-3` |
| Weight decay / gradient clip | `1e-4` / `1.0` |
| Schedule | 15% warmup, cosine decay, 2% LR floor, 25-epoch horizon |
| EMA | `0.999`; final EMA is used for OOFs and exported fullfits |
| Model head | 512-d embedding, 384-channel depth mixer, kernel 3, dropout 0.1 |
| Loss | unweighted BCE + 0.2 auxiliary BCE; no confidence weighting |
| Augmentation | ±10° rotation, ±10% scale, ±5% shift, ±20% gamma, 15% slot dropout |
| Data workers | 12 train, 3 validation |
| Balanced sampler grouping | two consecutive microbatches preserve four-study size strata |
| Checkpointing | one atomic resumable last checkpoint; no epoch archives |

| Experiment | Role | Backbone | Epochs executed | Run seeds |
|---|---|---|---:|---|
| v307 | five-fold OOF | ConvNeXt-Tiny | 14 | `2026 + fold` |
| v310 | five-fold OOF | EfficientNetV2-S | 12 | `2026 + fold` |
| v311 | five full-data members | EfficientNetV2-S | 19 | `3026..3030` |
| v313 | five full-data members | ConvNeXt-Tiny | 21 | `4026..4030` |

All four stop at the listed epoch while retaining the 25-epoch LR horizon. v307/v310
validate only after their final epoch. OOF pooling rejects incomplete histories,
non-final checkpoints, missing EMA state, unexpected seeds, or incomplete fold coverage.

### End-to-end reproduction

First complete the environment, competition download, and pinned timm weight download
described above. The two teammate backbones require:

```text
assets/weights/pretrained/timm/convnext_tiny.fb_in22k_ft_in1k/model.safetensors
assets/weights/pretrained/timm/tf_efficientnetv2_s.in21k_ft_in1k/model.safetensors
```

Their SHA-256 values are pinned in `cfg_v31x.py` and checked when a model is built.

**1. Build effective CV tables and the teammate `cached_shards` image cache.**

```bash
# These commands also build train_1_85.csv when it is absent.
python -m lab.utils.prepare_v31x prepare --experiment v307
python -m lab.utils.prepare_v31x prepare --experiment v310

# All four experiments read the same converter-generated study NPZs.
python -m lab.utils.prepare_v31x cache \
  --experiment v307 --workers 16 --laterality-workers 8
```

This creates ignored effective `train_v307.csv` and `train_v310.csv` tables plus
`assets/data/cached_shards/<StudyInstanceUID>.npz` and its `manifest.json`. Cache
construction is CPU/disk work and is resumable. For a cheap initial check, append
`--study-limit 1`; rerun the command without the limit before training. The loader
refuses to start if any requested study is absent. Hidden-test inference calls the same
shared DICOM preprocessing functions as the converter, so cached training tensors and
on-demand inference tensors follow one pixel path. Run the focused preflight tests
before renting GPUs:

```bash
python -m unittest discover -s tests -v
```

**2. Train v307 and v310 CV.**

The commands below run sequentially on GPU 0. Re-running an identical command resumes
from its atomic `checkpoint_last_seed*.pth`.

```bash
export CUDA_VISIBLE_DEVICES=0
for fold in 0 1 2 3 4; do
  python -m lab.utils.prepare_v31x train \
    --experiment v307 --fold "$fold" --master-port 29607
done
for fold in 0 1 2 3 4; do
  python -m lab.utils.prepare_v31x train \
    --experiment v310 --fold "$fold" --master-port 29610
done
```

On a host with multiple GPUs, launch folds in separate terminals
with one `CUDA_VISIBLE_DEVICES` value per terminal. If runs share a host, give each concurrent
run a unique `--master-port`. Keep the default `--nproc 1`; accumulation preserves the
effective batch-four optimizer and scheduler profile.

Outputs are written to `outputs/cfg_v307/fold0..fold4` and
`outputs/cfg_v310/fold0..fold4`.

**3. Pool final-epoch EMA OOFs and generate fullfit labels.**

Only run pooling after all five folds of that experiment are complete:

```bash
python -m lab.utils.prepare_v31x oof --experiment v307
python -m lab.utils.prepare_v31x oof --experiment v310

python -m lab.utils.prepare_v31x prepare --experiment v311
python -m lab.utils.prepare_v31x prepare --experiment v313
```

This writes:

```text
outputs/cfg_v307/final_oof.csv
outputs/cfg_v307/final_oof.provenance.json
outputs/cfg_v310/final_oof.csv
outputs/cfg_v310/final_oof.provenance.json
assets/data/labels_v311.csv
assets/data/train_v311.csv
assets/data/labels_v313.csv
assets/data/train_v313.csv
```

The v311 preparation requires the generated v310 OOF; v313 requires the generated v307
OOF. The two branches are otherwise independent. Label preparation refuses to overwrite
an existing fusion by default; pass `--force` only when rebuilding it from
newly regenerated OOFs.

**4. Train five v311 and five v313 full-data members.**

```bash
export CUDA_VISIBLE_DEVICES=0
for repeat in 1 2 3 4 5; do
  python -m lab.utils.prepare_v31x train \
    --experiment v311 --repeat "$repeat" --master-port 29711
done
for repeat in 1 2 3 4 5; do
  python -m lab.utils.prepare_v31x train \
    --experiment v313 --repeat "$repeat" --master-port 29713
done
```

Independent members can also be distributed across GPUs or machines. Preserve the
repeat number because it determines the seed and output folder. Outputs are
`outputs/cfg_v311/repeat1..repeat5/fold-1` and
`outputs/cfg_v313/repeat1..repeat5/fold-1`.

**5. Export the five final EMA members for inference.**

```bash
python -m lab.utils.export_knee --config cfg_v311
python -m lab.utils.export_knee --config cfg_v313
```

The exporter verifies every final checkpoint reached the configured final epoch and
writes inference bundles under `outputs/export/cfg_v311/` and
`outputs/export/cfg_v313/`.
