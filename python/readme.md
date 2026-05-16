# Apptainer Tutorial

## Set Cache and Temp Dir to Data
```
mkdir -p ~/data/.apptainer/tmp
mkdir -p ~/data/.apptainer/cache

export APPTAINER_TMPDIR=~/data/.apptainer/tmp
export APPTAINER_CACHEDIR=~/data/.apptainer/cache

echo $APPTAINER_TMPDIR
echo $APPTAINER_CACHEDIR
```

```
nvidia-smi
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

## Install Deps
```
python -m pip install albumentations
# hacky fix since lib like matplot was compiled with numpy 1.x
# atleast for this apptainer!
python -m pip install --force-reinstall "numpy<1.27"
python -m pip install mlflow
python -m pip install onnx onnxscript
python -m pip install kornia
pip install timm
```

# OLD
## Create Venv in Jupyter Sess
```
python -m pip freeze > /tmp/container-requirements.txt


python -m virtualenv /workspace/.venv
source /workspace/.venv_clean/bin/activate
python -m pip install --upgrade pip setuptools wheel



python -m pip install --user virtualenv
# python -m virtualenv .venv
python -m virtualenv --system-site-packages .venv

source .venv/bin/activate
python -m pip install ipykernel
python -m ipykernel install --user --name ba-venv --display-name "BA venv"
```

## Export Requirements
```
python -m pip freeze --local > requirements.lock.txt
```
## Import:
```
python -m virtualenv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
```


# UV

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv sync
uv run python -m ipykernel install --user --name=social-media-overlay --display-name "Python social-media-overlay"
```