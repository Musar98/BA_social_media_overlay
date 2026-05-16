"""
Training entry point for the dense latent style model.

This script trains a dense latent-style model in a distributed PyTorch setup.
The main idea is similar to the regular latent style model: a pretrained MUNIT
generator is used to create or manipulate images in latent style space, while a
pretrained emotion regressor checks the emotional output of those generated
images.

The trainable part in this file is the LatentStyleDenseModel. Compared to the
simpler latent style model, this dense version uses a deeper direction network
to learn how the style representation should be changed. The model learns a
mapping from an emotion attribute value to a direction in the MUNIT style latent
space. This direction is then used to modify the generated image so that the
emotion regressor predicts a result closer to the target emotion.

The script does the following:
1. Starts Distributed Data Parallel (DDP), so the same training job can run on
   one or more GPUs.
2. Loads COCO images and applies the preprocessing expected by the generator and
   regressor.
3. Loads a pretrained MUNIT generator checkpoint from Gebhardt et al.
4. Loads a pretrained affect/emotion regressor from Gebhardt et al.
5. Freezes the generator and regressor, because they are used as fixed
   pretrained components and should not be updated during this training.
6. Probes the generator once with a random image to find the size of the MUNIT
   style latent vector.
7. Builds the LatentStyleDenseModel, optimizer, loss function, distributed
   sampler, and dataloader.
8. Optionally resumes from a checkpoint.
9. Runs training and optionally writes intermediate and final checkpoints.

Most values below are configuration or hyperparameter values. They are grouped
by purpose, and each variable has a comment directly above it explaining what it
controls and when it should or should not be changed.
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

from torch.nn import MSELoss
from torch.optim import Adam

from social_media_overlay.ext.gebhardt.dataset.CocoCaptions import CocoCaptions
from social_media_overlay.ext.imaginaire.config import Config
from social_media_overlay.ext.imaginaire.generators.munit import Generator
from social_media_overlay.ext.gebhardt.utils import get_relevant_states
from social_media_overlay.models.regressor import EmotionRegressor
from social_media_overlay.models.latent_style_dense import LatentStyleDenseModel
from social_media_overlay.ddp import setup_ddp, load_checkpoint_if_requested, cleanup_ddp, safe_checkpoint
from social_media_overlay.ddp_train_latent_dense import train


# Root folder of the COCO dataset on disk. The CocoCaptions dataset class expects
# this directory to contain the relevant COCO split folders/files. Change this if
# your local dataset is stored somewhere else.
COCO_DATA_PATH = "../data/coco"

# COCO split to load. This usually corresponds to a dataset directory or split
# name such as "train", "val", or "test". Use "train" for real training, "val"
# for validation-style experiments, and "test" only if your dataset class
# supports it and you intentionally want the test split.
COCO_SPLIT = "val"

# Size used by transforms.Resize before cropping. Images are first resized so
# that the shorter side reaches this size, then a center crop is applied below.
# This should normally match what the pretrained generator/regressor expect.
IMAGE_RESIZE_SIZE = 1024

# Final square crop size passed into the model. A larger crop keeps more image
# detail but uses more GPU memory. This should normally be equal to or smaller
# than IMAGE_RESIZE_SIZE.
IMAGE_CROP_SIZE = 1024

# Number of images processed by each GPU at once. The effective global batch size
# is this value multiplied by the number of GPUs. Increasing this can make
# training faster, but it also increases VRAM usage. More than 2 may not fit on
# smaller GPUs such as a P100.
BATCH_SIZE_PER_GPU = 2

# Whether the DistributedSampler should shuffle the dataset every epoch. This is
# usually True for training, because it reduces ordering bias and improves
# stochastic optimization. For deterministic debugging, set it to False.
SHUFFLE_DATA = True

# Number of worker processes used by the DataLoader to load and preprocess data.
# Higher values can speed up input loading, especially when image preprocessing
# is expensive, but too many workers can overload the CPU or file system.
NUM_WORKERS = 4

# If True, DataLoader stores loaded tensors in page-locked host memory before
# moving them to the GPU. This often speeds up CPU-to-GPU transfer when training
# on CUDA. It is usually safe to keep True for GPU training.
PIN_MEMORY = True

# If True, the last incomplete batch of an epoch is discarded. This is useful in
# distributed training because each GPU then receives batches with the same size.
# Set False if keeping every sample is more important than consistent batch size.
DROP_LAST_BATCH = True

# Mean used for image normalization after converting the image to a tensor.
# Values of [0.5, 0.5, 0.5] transform RGB channels from roughly [0, 1] into
# approximately [-1, 1], which is common for GAN-style generators.
NORMALIZE_MEAN = [0.5, 0.5, 0.5]

# Standard deviation used for image normalization. Together with NORMALIZE_MEAN,
# this controls the pixel value range seen by the model. Keep this consistent
# with the preprocessing used when the pretrained models were trained.
NORMALIZE_STD = [0.5, 0.5, 0.5]

# Number of full passes over the selected COCO split. Increase this for a real
# training run. The current value of 1 is mainly useful for quick tests or
# debugging.
NUM_EPOCHS = 1

# Optimizer learning rate. This controls how large each Adam update step is. The
# value 0.0002 follows the GANalyze-style setting used in the latent model, but
# it may still need tuning for this dense latent-style setup.
LEARNING_RATE = 0.0002

# How often the training loop prints progress, measured in batches. A value of
# 100 means the code prints roughly once every 100 batches. Lower values give
# more logging but can make training output noisy.
PRINT_EVERY_N_BATCHES = 100

# Directory where checkpoints are written. Rank 0 creates this folder if it does
# not exist. Use a separate directory for different experiments to avoid mixing
# checkpoint files from unrelated runs.
CHECKPOINT_DIR = "../data/models/test-dense"

# Prefix used for checkpoint filenames. For example, with CHECKPOINT_NAME="test",
# checkpoint files will be named with "test" as the identifying part of the
# filename. Change this for each experiment so files are easy to recognize.
CHECKPOINT_NAME = "test"

# Whether to save checkpoints during/after training. Keeping this True is useful
# for long runs because training can be resumed after interruption. Set False for
# quick experiments where saving model files is not needed.
SAVE_CHECKPOINTS = True

# Optional path to a checkpoint to resume from. If this is an empty string, the
# training starts from scratch. If a valid checkpoint path is provided,
# load_checkpoint_if_requested is expected to restore the model state and, if the
# checkpoint contains it, also the optimizer state.
RESUME_CHECKPOINT_PATH = ""

# Path to the Imaginaire YAML config used to construct the MUNIT generator. This
# config must match the architecture of the generator checkpoint below; otherwise
# the checkpoint weights may not load correctly.
IMAGINAIRE_CONFIG_PATH = "../src/social_media_overlay/ext/imaginaire/imagenet2imagenet.yaml"

# Path to the pretrained MUNIT generator checkpoint from Gebhardt et al. The
# script loads this checkpoint, extracts the generator weights, and uses the
# generator as a fixed image transformation component.
GENERATOR_CHECKPOINT_PATH = "../data/gebhardt/models/imaginaire_munit_200000_s5.pt"

# Path to the pretrained affect/emotion regressor model from Gebhardt et al. This
# regressor predicts emotional values from generated images and provides the
# training signal for the dense latent style model.
REGRESSOR_MODEL_PATH = "../data/gebhardt/models/va_pred_all"

# Whether the regressor should average predictions over crops. Crop averaging can
# make predictions more stable and robust, but it uses significantly more VRAM
# because several crops may be processed instead of only one image view.
REGRESSOR_USE_AVERAGE = True

# Lower bound of the attribute value range used by the LatentStyleDenseModel.
# The model samples or receives attribute values in this range to control the
# direction and strength of the desired emotional change.
ATTRIBUTE_MIN_VALUE = -0.5

# Upper bound of the attribute value range used by the LatentStyleDenseModel.
# Keep this consistent with ATTRIBUTE_MIN_VALUE and with any assumptions inside
# the training code. A future improvement could add a validation check to make
# sure the range is sensible.
ATTRIBUTE_MAX_VALUE = 0.5

# Depth of the dense direction MLP inside LatentStyleDenseModel. This controls
# how many layers the model uses to learn the mapping from the attribute value to
# a style-space direction. A larger depth can make the model more expressive, but
# it can also make training slower or easier to overfit.
DIRECTION_MLP_DEPTH = 3

# Batch size of the random dummy input used only to probe the generator. This is
# not the training batch size. It only needs to be large enough to pass through
# the encoder and reveal the style latent shape, so 1 is enough and saves VRAM.
STYLE_DIM_PROBE_BATCH_SIZE = 1

# Number of channels in the random probe image. RGB images have 3 channels, so
# this should stay 3 unless the generator was explicitly trained for a different
# input format.
STYLE_DIM_PROBE_CHANNELS = 3

# Height and width of the square random probe image. This does not control the
# real training image size. It only needs to be a valid input size for the MUNIT
# encoder so that the code can infer the style vector dimension.
STYLE_DIM_PROBE_IMAGE_SIZE = 256


def main():
    rank, local_rank, world_size = setup_ddp()

    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    dist.barrier()

    data_transforms = transforms.Compose([
        transforms.Resize(IMAGE_RESIZE_SIZE),
        transforms.CenterCrop(IMAGE_CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=NORMALIZE_MEAN,
            std=NORMALIZE_STD,
        ),
    ])

    dataset_coco = CocoCaptions(COCO_DATA_PATH, COCO_SPLIT, data_transforms)

    cfg = Config(IMAGINAIRE_CONFIG_PATH)
    gen = Generator(cfg.gen, cfg.data)

    state_dict = torch.load(
        GENERATOR_CHECKPOINT_PATH,
        map_location=device,
    )

    gen_state_dict = get_relevant_states(state_dict["net_G"])
    gen.load_state_dict(gen_state_dict)
    gen = gen.to(device)

    regressor = EmotionRegressor(
        REGRESSOR_MODEL_PATH,
        average=REGRESSOR_USE_AVERAGE,
        normalize=False,
        device=device,
    ).to(device)

    for param in gen.parameters():
        param.requires_grad = False

    for param in regressor.model.parameters():
        param.requires_grad = False

    style_dim_probe_input = torch.randn(
        STYLE_DIM_PROBE_BATCH_SIZE,
        STYLE_DIM_PROBE_CHANNELS,
        STYLE_DIM_PROBE_IMAGE_SIZE,
        STYLE_DIM_PROBE_IMAGE_SIZE,
        device=device,
    )

    with torch.no_grad():
        content_a, style_a = gen.autoencoder_a.encode(style_dim_probe_input)

    style_dim = style_a.shape[1]

    if rank == 0:
        print("MUNIT Latent Space Info")
        print("Content shape:", content_a.shape)
        print("Style shape:", style_a.shape)
        print("Style dimension:", style_dim)
        print(f"World size: {world_size}")
        print(f"Per-GPU batch size: {BATCH_SIZE_PER_GPU}")
        print(f"Effective global batch size: {BATCH_SIZE_PER_GPU * world_size}")

    model = LatentStyleDenseModel(
        regressor=regressor,
        generator=gen,
        style_dim=style_dim,
        a_min=ATTRIBUTE_MIN_VALUE,
        a_max=ATTRIBUTE_MAX_VALUE,
        direction_depth=DIRECTION_MLP_DEPTH,
    ).to(device)

    loss_fn = MSELoss()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = Adam(trainable_params, lr=LEARNING_RATE)

    sampler = DistributedSampler(
        dataset_coco,
        num_replicas=world_size,
        rank=rank,
        shuffle=SHUFFLE_DATA,
    )

    dataloader = DataLoader(
        dataset_coco,
        batch_size=BATCH_SIZE_PER_GPU,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=DROP_LAST_BATCH,
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    start_epoch, start_batch, loss_history, _ = load_checkpoint_if_requested(
        model=model,
        optimizer=optimizer,
        device=device,
        rank=rank,
        checkpoint_path=RESUME_CHECKPOINT_PATH,
    )

    try:
        last_completed_epoch = start_epoch
        last_completed_batch = start_batch

        for epoch in range(start_epoch, NUM_EPOCHS):
            epoch_start_batch = start_batch if epoch == start_epoch else 0

            loss_history = train(
                dataloader=dataloader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                save_state=SAVE_CHECKPOINTS,
                save_path=CHECKPOINT_DIR,
                save_name=CHECKPOINT_NAME,
                rank=rank,
                loss_history=loss_history,
                epoch=epoch,
                start_batch=epoch_start_batch,
                print_every=PRINT_EVERY_N_BATCHES,
            )

            last_completed_epoch = epoch
            last_completed_batch = len(dataloader) - 1

            start_batch = 0
            dist.barrier()

        if SAVE_CHECKPOINTS:
            if rank == 0:
                safe_checkpoint(
                    CHECKPOINT_DIR,
                    CHECKPOINT_NAME,
                    last_completed_epoch,
                    last_completed_batch,
                    model.module.state_dict(),
                    optimizer.state_dict(),
                    loss_history=loss_history,
                    val_loss_history=None,
                    name_suffix="-final.pt",
                )

            dist.barrier()

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()