"""
Training entry point for the multivariate conditional parametric image model.

This script trains a multivariate parametric image-transformation model in a
distributed PyTorch setup. The model receives an input image and multiple
conditioning values, predicts interpretable image-transformation parameters, and
applies those transformations directly to the image.

Compared to the single-condition parametric model, this version uses the
MultiConditionalParametricResidualMobileNetGenerator. This means the model can
condition its edits on more than one target value, for example multiple emotion
dimensions. The generator predicts residual transformation parameters around
identity values, so neutral conditions should ideally result in little or no
change to the original image.

The pretrained emotion regressor is used as a fixed evaluator. It predicts
emotion values from the transformed image and provides the training signal for
the generator. The regressor itself is not trained.

The script does the following:
1. Starts Distributed Data Parallel (DDP), so the same training job can run on
   one or more GPUs.
2. Loads COCO training and validation images.
3. Loads a pretrained affect/emotion regressor from Gebhardt et al.
4. Freezes the regressor by default, because it is used as a fixed evaluator.
5. Builds the multivariate conditional parametric generator, optimizer, loss
   function, distributed samplers, and dataloaders.
6. Optionally resumes from a checkpoint.
7. Optionally runs validation only.
8. Runs training, evaluates on the validation split after each epoch, and writes
   checkpoints if enabled.

Most values below are configuration or hyperparameter values. They are grouped by
purpose, and each variable has a comment directly above it explaining what it
controls and when it should or should not be changed.
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from timm.data import resolve_model_data_config
from torch.nn import MSELoss
from torch.optim import Adam

from social_media_overlay.ext.gebhardt.dataset.CocoCaptions import CocoCaptions
from social_media_overlay.models.regressor import EmotionRegressor
from social_media_overlay.models.parametric_multivariate import MultiConditionalParametricResidualMobileNetGenerator
from social_media_overlay.ddp import setup_ddp, load_checkpoint_if_requested, cleanup_ddp, safe_checkpoint
from social_media_overlay.ddp_train_parametric_multivariate import train, val

# data
# Root folder of the COCO dataset on disk. The CocoCaptions dataset class expects
# this directory to contain the relevant COCO split folders/files. Change this if
# your local dataset is stored somewhere else.
COCO_DATA_PATH = "../data/coco"
# COCO split used for actual training. This should normally be "train".
TRAIN_SPLIT = "train"
# COCO split used for validation after each epoch, or for validation-only runs.
# NOT IMPLEMENTED
VAL_SPLIT = "val"

# Size used by transforms.Resize before cropping. In this setup, this value does
# not directly control the MobileNet backbone input size because the backbone
# preprocessing resizes internally. However, it still affects the image size on
# which the final parametric transformations are applied.
IMAGE_RESIZE_SIZE = 480 
# Final square crop size passed into the parametric transformation pipeline. The
# backbone and regressor handle their own resizing/normalization internally, but
# this value still determines the output resolution of the edited images.
IMAGE_CROP_SIZE = 480

# Number of images processed by each GPU at once. The effective global batch size
# is this value multiplied by the number of GPUs.
BATCH_SIZE_PER_GPU = 8
# Whether the training DistributedSampler should shuffle the data every epoch.
# This is usually True for training because it reduces ordering bias.
SHUFFLE_TRAIN_DATA = True
# Number of worker processes used by each DataLoader. Higher values can speed up
# image loading/preprocessing, but too many workers can overload the CPU or disk.
NUM_WORKERS = 8
# If True, DataLoader stores tensors in page-locked host memory before moving
# them to the GPU. This often speeds up CPU-to-GPU transfers during CUDA training.
PIN_MEMORY = True
# If True, incomplete final batches are dropped. Keeping this False preserves all
# samples, while True can make distributed batch sizes more consistent.
DROP_LAST_BATCH = False

# training
# Number of full passes over the training split. The current value is mainly
# useful for quick tests or debugging; increase it for real training runs
NUM_EPOCHS = 1
# Optimizer learning rate. This controls the size of Adam update steps.
LEARNING_RATE = 0.0001
# How often the training loop prints progress, measured in batches.
PRINT_EVERY_N_BATCHES = 100
# If True, skip training and run only the validation loop. This is useful for
# checking a checkpoint or evaluating a fixed model setup.
ONLY_RUN_VALIDATION = False

# checkpointing
# Directory where checkpoints are written. Rank 0 creates this folder if it does
# not already exist.
CHECKPOINT_DIR = "../data/models/chkp-param_mult-resi_ref-uni-480-b64-fc32_3-lr1e_4-1"
# Prefix used for checkpoint filenames. Change this for each experiment so
# checkpoints are easy to identify.
CHECKPOINT_NAME = "chkp-param_mult-resi_ref-uni-480-b64-fc32_3-lr1e_4-1"
# Optional path to a checkpoint to resume from. If this is empty, training starts
# from scratch.
RESUME_CHECKPOINT_PATH = ""
# Whether to save intermediate/final checkpoints. Keep this True for longer runs
# so training can be resumed after interruption.
SAVE_CHECKPOINTS = True
# Whether to restore the optimizer state when loading a checkpoint. Set this to
# False when starting from a checkpoint but intentionally rebuilding the optimizer,
# for example for finetuning with a changed set of trainable parameters or a new
# learning rate.
LOAD_OPTIMIZER_FROM_CHECKPOINT = False

# regressor
# Path to the pretrained affect/emotion regressor model from Gebhardt et al. This
# regressor predicts emotion values from edited images and provides the main
# training signal.
REGRESSOR_MODEL_PATH = "../data/gebhardt/models/va_pred_all"
# Whether the regressor should average predictions over crops. Crop averaging can
# make predictions more stable, but it is more memory-intensive.
REGRESSOR_USE_AVERAGE = False
# Whether the regressor should internally normalize its input. This should match
# the preprocessing expected by the pretrained regressor.
REGRESSOR_NORMALIZE_INPUT = True
# If True, the regressor weights are frozen. This should normally stay True
# because the regressor is used as a fixed evaluator, not as a trainable model.
FREEZE_REGRESSOR = True

# generator
# Width of the shared fully connected trunk inside the multivariate parametric
# generator.
GENERATOR_FC_DIM = 32
# Depth of the shared fully connected trunk. Larger values increase model
# capacity but also add parameters and training cost.
GENERATOR_FC_DEPTH = 3
# Size of the learned condition embedding used to represent the multivariate
# conditioning input.
GENERATOR_COND_DIM = 32
# Whether to initialize the MobileNet backbone with pretrained weights. This
# should normally stay True because the backbone is used as an image feature
# extractor.
GENERATOR_USE_PRETRAINED_BACKBONE = True
# If True, the MobileNet backbone is frozen and only the condition/trunk/parameter
# heads are trained. Set this to False only when intentionally finetuning the
# backbone.
GENERATOR_FREEZE_BACKBONE = True

# ddp
# DDP setting controlling whether PyTorch should search for parameters that were
# not used in a forward pass. This can be True for conditional branches or partial
# finetuning, but False is faster when all parameters are always used.
DDP_FIND_UNUSED_PARAMETERS = False


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
    ])

    train_dataset = CocoCaptions(COCO_DATA_PATH, TRAIN_SPLIT, data_transforms)
    val_dataset = CocoCaptions(COCO_DATA_PATH, VAL_SPLIT, data_transforms)

    regressor = EmotionRegressor(
        path_to_model=REGRESSOR_MODEL_PATH,
        average=REGRESSOR_USE_AVERAGE,
        device=device,
        normalize=REGRESSOR_NORMALIZE_INPUT,
    ).to(device)

    regressor.eval()
    if FREEZE_REGRESSOR:
        for param in regressor.model.parameters():
            param.requires_grad = False

    if rank == 0:
        print("MUNIT Latent Space Info")
        print(f"World size: {world_size}")
        print(f"Per-GPU batch size: {BATCH_SIZE_PER_GPU}")
        print(f"Effective global batch size: {BATCH_SIZE_PER_GPU * world_size}")
        print(f"Only validation mode: {ONLY_RUN_VALIDATION}")

    model = MultiConditionalParametricResidualMobileNetGenerator(
        fc_dim=GENERATOR_FC_DIM,
        fc_depth=GENERATOR_FC_DEPTH,
        mobilenet_pretrained=GENERATOR_USE_PRETRAINED_BACKBONE,
        freeze_backbone=GENERATOR_FREEZE_BACKBONE,
        cond_dim=GENERATOR_COND_DIM,
    ).to(device)
    model.train()

    if rank == 0:
        print("backbone_input_hw:", model.backbone_input_hw)
        print("backbone_mean:", model.backbone_mean.flatten())
        print("backbone_std:", model.backbone_std.flatten())
        print("timm data config:", resolve_model_data_config(model.backbone))

    loss_fn = MSELoss()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = Adam(trainable_params, lr=LEARNING_RATE)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=SHUFFLE_TRAIN_DATA,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_PER_GPU,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=DROP_LAST_BATCH,
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_PER_GPU,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=DROP_LAST_BATCH,
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=DDP_FIND_UNUSED_PARAMETERS,
    )

    start_epoch, start_batch, loss_history, val_loss_history = load_checkpoint_if_requested(
        model=model,
        optimizer=optimizer,
        device=device,
        rank=rank,
        checkpoint_path=RESUME_CHECKPOINT_PATH,
        load_optimizer=LOAD_OPTIMIZER_FROM_CHECKPOINT,
    )

    # only for finetuning
    # trainable_params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = Adam(trainable_params, lr=LEARNING_RATE) # only set new LR once at the start of FT!

    if rank == 0:
        for name, param in model.named_parameters():
            print(name, param.requires_grad, param.shape)

    try:
        if ONLY_RUN_VALIDATION:
            dist.barrier()

            avg_val_loss = val(
                dataloader=val_dataloader,
                model=model,
                regressor=regressor,
                device=device,
                loss_fn=loss_fn,
            )
            dist.barrier()
            if rank == 0:
                def _fmt(v):
                    if isinstance(v, float):
                        return f"{v:.6f}"
                    return str(v)

                print("\n" + "=" * 100)
                print("VALIDATION SUMMARY")
                print("=" * 100)
                print(f"Validation loss: {avg_val_loss:.6f}")
            dist.barrier()
            return

        last_completed_epoch = start_epoch
        last_completed_batch = start_batch

        for epoch in range(start_epoch, NUM_EPOCHS):
            epoch_start_batch = start_batch if epoch == start_epoch else 0

            loss_history = train(
                dataloader=train_dataloader,
                model=model,
                regressor=regressor,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                save_state=SAVE_CHECKPOINTS,
                save_path=CHECKPOINT_DIR,
                save_name=CHECKPOINT_NAME,
                rank=rank,
                loss_history=loss_history,
                val_loss_history=val_loss_history,
                epoch=epoch,
                start_batch=epoch_start_batch,
                print_every=PRINT_EVERY_N_BATCHES,
            )

            dist.barrier()
            avg_val_loss = val(
                dataloader=val_dataloader,
                model=model,
                regressor=regressor,
                device=device,
                loss_fn=loss_fn,
            )
            val_loss_history.append(avg_val_loss)

            if rank == 0:
                print(f"Epoch {epoch} validation loss: {avg_val_loss}")

            last_completed_epoch = epoch
            last_completed_batch = len(train_dataloader) - 1

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
                        val_loss_history = val_loss_history,
                        name_suffix = "-final.pt"
                )
            dist.barrier()

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()