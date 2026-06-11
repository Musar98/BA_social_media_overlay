# Python Training and Evaluation Code

This directory contains the Python code used to train, validate, evaluate, and
export the image transformation models for the social media overlay project.

The Android app and TypeScript overlay engine run inference in the WebView. The
Python code here is the offline research/training side: it trains PyTorch models,
evaluates image-editing behavior, and supports exporting the learned
parametric transformation model that is later used by the app through ONNX.

## Important Path Convention

Run commands from this directory:

```sh
cd python
```

Most training scripts use relative paths such as `../data/coco` and
`../data/models/...`. That means the expected data folder is at the repository
root:

```text
BA_social_media_overlay/
+-- data/
`-- python/
```

The `data/` directory is intentionally ignored by Git because it contains large
datasets, checkpoints, model weights, and experiment outputs.

## Directory Layout

```text
python/
+-- pyproject.toml
+-- uv.lock
+-- readme.md
+-- apptainer/
|   `-- Legacy container definitions for DGX/HPC usage
+-- notebooks/
|   +-- Training, evaluation, plotting, inference, and export notebooks
|   `-- tests/
|       `-- Older exploratory/test notebooks
`-- src/
    +-- social_media_overlay/
    |   +-- ddp_train_parametric_main.py
    |   +-- ddp_train_parametric_multivariate_main.py
    |   +-- ddp_train_latent_main.py
    |   +-- ddp_train_latent_dense_main.py
    |   +-- ddp_eval_parametric.py
    |   +-- ddp.py
    |   +-- models/
    |   |   `-- Parametric, latent-style, regressor, and transformation modules
    |   `-- ext/
    |       `-- External/reference code kept separate from project code
    `-- tests/
        `-- Unit and integration tests
```

The Python package is still named `social_media_overlay`, even though the thesis
and app naming evolved over time.

## Setup

This project uses `uv` for dependency management and Python `>=3.12`.

```sh
# From python/
uv sync
```

Register a Jupyter kernel for notebooks:

```sh
uv run python -m ipykernel install --user --name=social-media-overlay --display-name "Python social-media-overlay"
```

Then open a notebook and select `Python social-media-overlay`.

### CLIP Dependency

`ddp_train_parametric_main.py` imports `clip` and loads OpenAI CLIP when
`CLIP_WEIGHT > 0`. The default script currently has `CLIP_WEIGHT = 0.05`.

Use one of these options before running the default parametric training script:

```sh
# Install OpenAI CLIP into the uv environment.
uv pip install git+https://github.com/openai/CLIP.git
```

or set this constant in `src/social_media_overlay/ddp_train_parametric_main.py`:

```python
CLIP_WEIGHT = 0.0
```

The optional real-CLIP integration test also requires this package and may
download model weights on first use.

## Data and Model Requirements

Training and evaluation expect COCO data and pretrained Gebhardt model assets
under the repository-root `data/` directory.

Expected COCO layout:

```text
data/
`-- coco/
    +-- train2017/
    +-- val2017/
    +-- test2017/
    `-- annotations/
        +-- captions_train2017.json
        `-- captions_val2017.json
```

Expected pretrained Gebhardt assets:

```text
data/
`-- gebhardt/
    `-- models/
        +-- va_pred_all
        `-- imaginaire_munit_200000_s5.pt
```

Main paths used by the scripts:

```text
../data/coco
../data/gebhardt/models/va_pred_all
../data/gebhardt/models/imaginaire_munit_200000_s5.pt
../data/models/...
```

`../data/models/...` is used for experiment checkpoints and final model files.
Use a separate checkpoint directory/name for each experiment to avoid mixing
outputs from unrelated runs.

## Training Entrypoints

The training scripts are configured by constants near the top of each file. Before launching a run, open the corresponding
`*_main.py` file and review the data paths, checkpoint settings, batch size,
number of epochs, and model-specific toggles.

### Parametric Model

Primary script:

```text
src/social_media_overlay/ddp_train_parametric_main.py
```

This is the main training path for the learned parametric image editor. The
model predicts interpretable transformation parameters in this order:

```text
exposure, saturation, tone, color, contrast, sharp, blur
```

The full flattened parameter vector has 37 values:

```text
exposure:   1
saturation: 1
tone:       8
color:      24
contrast:   1
sharp:      1
blur:       1
```

Important constants to review:

```text
COCO_DATA_PATH
TRAIN_SPLIT
VAL_SPLIT
BATCH_SIZE_PER_GPU
NUM_EPOCHS
CHECKPOINT_DIR
CHECKPOINT_NAME
RESUME_CHECKPOINT_PATH
SAVE_CHECKPOINTS
LOAD_OPTIMIZER_FROM_CHECKPOINT
REGRESSOR_MODEL_PATH
REGRESSOR_USE_AROUSAL
GENERATOR_FREEZE_BACKBONE
GENERATOR_UNFREEZE_BACKBONE_BLOCKS
CLIP_WEIGHT
ONLY_RUN_VALIDATION
```

Run with `torchrun`:

```sh
uv run torchrun --standalone --nproc_per_node=<GPU_COUNT> src/social_media_overlay/ddp_train_parametric_main.py
```

Example for 8 GPUs:

```sh
uv run torchrun --standalone --nproc_per_node=8 src/social_media_overlay/ddp_train_parametric_main.py
```

Validation-only mode is controlled by:

```python
ONLY_RUN_VALIDATION = True
```

Resume from a checkpoint by setting:

```python
RESUME_CHECKPOINT_PATH = "../data/models/<run>/<checkpoint>.pt"
```

If you are intentionally changing trainable parameters for finetuning, review:

```python
LOAD_OPTIMIZER_FROM_CHECKPOINT = False
GENERATOR_FREEZE_BACKBONE = False
GENERATOR_UNFREEZE_BACKBONE_BLOCKS = 1
```

When continuing an already-started finetuning run, set
`LOAD_OPTIMIZER_FROM_CHECKPOINT` back to `True` if the checkpoint optimizer state
matches the current model setup.

### Multivariate Parametric Model

Script:

```text
src/social_media_overlay/ddp_train_parametric_multivariate_main.py
```

This is an experimental multi-condition variant of the parametric model. It uses
`MultiConditionalParametricResidualMobileNetGenerator` and conditions on
multiple target values instead of a single alpha.

Important notes:

- It follows the same DDP/checkpoint pattern as the primary parametric script.
- The validation split behavior is not fully implemented.
- Treat this path as future-work reference for a multivariate setup.

### Latent Style Model

Script:

```text
src/social_media_overlay/ddp_train_latent_main.py
```

This approach uses a pretrained MUNIT generator and trains a small
`LatentStyleModel` to move through the generator style latent space. The
generator and emotion regressor are normally frozen and used as fixed pretrained
components.

Important constants to review:

```text
COCO_DATA_PATH
COCO_SPLIT
BATCH_SIZE_PER_GPU
NUM_EPOCHS
CHECKPOINT_DIR
CHECKPOINT_NAME
RESUME_CHECKPOINT_PATH
IMAGINAIRE_CONFIG_PATH
GENERATOR_CHECKPOINT_PATH
REGRESSOR_MODEL_PATH
ATTRIBUTE_MIN
ATTRIBUTE_MAX
```

Run:

```sh
uv run torchrun --standalone --nproc_per_node=<GPU_COUNT> src/social_media_overlay/ddp_train_latent_main.py
```

### Dense Latent Style Model

Script:

```text
src/social_media_overlay/ddp_train_latent_dense_main.py
```

This is a denser variant of the latent-style approach. It trains
`LatentStyleDenseModel`, which uses a deeper direction network to map an
attribute value to a style-space direction.

Run:

```sh
uv run torchrun --standalone --nproc_per_node=<GPU_COUNT> src/social_media_overlay/ddp_train_latent_dense_main.py
```

## Evaluation and Metrics

The main evaluation helper module is:

```text
src/social_media_overlay/ddp_eval_parametric.py
```

It is used from validation code and notebooks to compare original images with
transformed images. It reports:

- Low-level image properties such as brightness, saturation, contrast,
  colorfulness, blur, and lighting diversity.
- Valence/arousal prediction changes from the pretrained emotion regressor.
- Pixel distance between original and transformed images.
- Distribution metrics using FID and KID.
- Runtime, throughput, and VRAM metrics.

Typical usage is:

1. Evaluate original validation images as a baseline.
2. Evaluate transformed images for a negative alpha, often `-0.1`.
3. Evaluate transformed images for a positive alpha, often `+0.1`.
4. Print a summary with `print_validation_summary(...)`.

The notebook `notebooks/parametric_approach_eval.ipynb` contains exploratory
evaluation workflows around this helper module.

## Notebooks and Export

The `notebooks/` directory contains experiment notebooks for training,
evaluation, plotting, inference, and model export:

```text
notebooks/DDP_train.ipynb
notebooks/parametric_approach_eval.ipynb
notebooks/parametric_approach_inference.ipynb
notebooks/latent_approach_inference.ipynb
notebooks/transformation_model_export.ipynb
```

Use notebooks for exploration and report figures, but keep repeatable training
runs anchored in the `src/social_media_overlay/*_main.py` scripts.

## Checkpoints and Outputs

Checkpoints are saved by rank 0 only. The shared helper
`src/social_media_overlay/ddp.py` stores checkpoint dictionaries with:

```text
epoch
batch
model_state_dict
optimizer_state_dict
loss_history
val_loss_history
```

Typical output paths are below `../data/models/...`, for example:

```text
data/models/test/
data/models/checkpoints-test/
data/models/test-dense/
```

The exact path and filename prefix are controlled by:

```python
CHECKPOINT_DIR
CHECKPOINT_NAME
```

Periodic checkpoints use suffixes created by the training loop. Final
checkpoints usually use `-final.pt`.

## Tests

Run the non-integration test suite:

```sh
uv run pytest -m "not integration"
```

Run all tests:

```sh
uv run pytest
```

Run a specific file:

```sh
uv run pytest src/tests/test_parametric.py
```

The tests cover alpha sampling, CLIP-loss helpers, COCO dataset loading,
external image transformation patches, low-level image metrics, parametric
models/utilities, shader parity checks, and shared utilities.

Integration tests may require external model downloads, a working GPU setup, or
OpenAI CLIP. CUDA-specific tests are skipped automatically when CUDA is not
available.

## Common Troubleshooting

### Dataset or checkpoint path not found

Make sure you launched from `python/`, not from the repository root. The scripts
expect paths like `../data/coco` relative to `python/`.

### DDP environment errors

Launch training scripts with `torchrun`, not plain `python`, because
`setup_ddp()` reads `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` from the environment:

```sh
uv run torchrun --standalone --nproc_per_node=<GPU_COUNT> src/social_media_overlay/ddp_train_parametric_main.py
```

### CUDA/NCCL errors

The DDP helper initializes the `nccl` backend and binds each process to
`LOCAL_RANK`. This requires CUDA-capable GPUs and a PyTorch build with CUDA/NCCL
support. For local CPU-only checks, run unit tests rather than DDP training.

### Out-of-memory during training

Reduce `BATCH_SIZE_PER_GPU`, reduce image crop size where appropriate, disable
CLIP loss with `CLIP_WEIGHT = 0.0`, or use fewer/more suitable model settings.
The latent MUNIT paths are especially memory-heavy because they use large image
sizes and pretrained generator/regressor components.

## Legacy DGX / Apptainer Setup

This section is kept for older DGX/HPC workflows. The current local Python setup
uses `uv`; only use the Apptainer flow if you are reproducing the old cluster
environment.

### Set Cache and Temp Directory

```sh
mkdir -p ~/data/.apptainer/tmp
mkdir -p ~/data/.apptainer/cache

export APPTAINER_TMPDIR=~/data/.apptainer/tmp
export APPTAINER_CACHEDIR=~/data/.apptainer/cache

echo $APPTAINER_TMPDIR
echo $APPTAINER_CACHEDIR
```

### Build the Container

```sh
cd python/apptainer
apptainer build ms_pytorch_jupyter.sif ms_pytorch_jupyter.def
```

### Start the Apptainer Instance

```sh
APPTAINERENV_CUDA_VISIBLE_DEVICES=<GPU numbers> \
apptainer instance start --nv \
  --bind ~/data/BA_social_media_overlay/python:/workspace \
  ~/data/BA_social_media_overlay/python/apptainer/ms_pytorch_jupyter.sif \
  mas_pytorch_jupyter
```

### Start Jupyter

```sh
apptainer run instance://mas_pytorch_jupyter
```

### Forward the Jupyter Port

Open a new shell on your local machine:

```sh
ssh -L 8889:localhost:8889 marcel.schubert@dgx-1
```

Then open:

```text
http://localhost:8889/
```

Keep the SSH shell open while using the forwarded notebook server.

### Stop the Instance

```sh
apptainer instance stop mas_pytorch_jupyter
```
