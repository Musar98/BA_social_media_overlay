# Training and Evaluation Code

This directory contains all Python code used for training and evaluating the different approaches.

## Directory Structure

```text
.
├── apptainer/
│   └── Old Apptainer images
│
├── notebooks/
│   └── Jupyter notebooks used for training, evaluation and plotting
│
├── src/
│   ├── social_media_overlay/
│   │   ├── Main Python package
│   │   └── ext/
│   │       └── External code kept separate from the main package code
│   │
│   └── tests/
│       └── Unit tests
```

The package is still called `social_media_overlay`, even though this is the old project name.

## Data Requirements

The project expects all data to be located inside the `data/` directory.

The COCO datasets must be stored in `data/coco/`.

Expected structure:

```text
data/
└── coco/
    ├── train2017/
    ├── val2017/
    ├── test2017/
    └── annotations/
        ├── captions_train2017.json
        └── captions_val2017.json
```

The Gebhardt components are expected to be stored in `data/gebhardt/`.

Expected files and directories:

```text
data/
└── gebhardt/
    ├── clf_best_cont_midu_va_1024_2024_07_22_16_01_14
    ├── clf_new_params_midu_va_512_2024_07_11_09_10_03
    ├── imaginaire_munit_200000_s5.pt
    └── va_pred_all
```

## Installation

This project uses `uv` for dependency management. The installation and dependency synchronization steps are shown below.

For more information about the project dependencies, see the `pyproject.toml` file.

```sh
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Synchronize dependencies
uv sync

# Register a Jupyter notebook kernel
uv run python -m ipykernel install --user --name=social-media-overlay --display-name "Python social-media-overlay"
```

After registering the kernel, open a notebook and select `Python social-media-overlay`.

## Run Tests

Run all tests:

```sh
uv run pytest
```

Run a specific test file:

```sh
uv run pytest src/tests/filename.py
```

## Useful Commands for DGX

Check current GPU usage:

```sh
nvidia-smi
```


# Older Setup with Apptainer

## Set Cache and Temp Dir to Data
```
mkdir -p ~/data/.apptainer/tmp
mkdir -p ~/data/.apptainer/cache

export APPTAINER_TMPDIR=~/data/.apptainer/tmp
export APPTAINER_CACHEDIR=~/data/.apptainer/cache

echo $APPTAINER_TMPDIR
echo $APPTAINER_CACHEDIR
```


## Build the Container
```
apptainer build ms_pytorch_jupyter.sif ms_pytorch_jupyter.def
```

## Start the Apptainer
```
APPTAINERENV_CUDA_VISIBLE_DEVICES=<GPU numbers> \
apptainer instance start --nv \
  --bind ~/data/BA_social_media_overlay/python:/workspace \
  ~/data/BA_social_media_overlay/python/apptainer/ms_pytorch_jupyter.sif \
  mas_pytorch_jupyter
```

## Start Jupyter
Start Jupyter
```
apptainer run instance://mas_pytorch_jupyter
```

## Forward Port
Open a new shell on your local machine and bind port.
```
ssh -L 8889:localhost:8889 marcel.schubert@dgx-1
```
*Here dgx-1 corresponds to the SSH alias i configured on the local machine!
Now you can open on the local machine using `http://localhost:8889/`.
Do not close the shell until finished!
## Close
```
apptainer instance stop ms_pytorch_jupyter
```