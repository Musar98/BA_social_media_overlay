"""
Training entry point for the conditional parametric image model.

This script trains a parametric image-transformation model in a distributed
PyTorch setup. Instead of changing a latent style code inside a pretrained
generator, this model directly predicts interpretable image-transformation
parameters, such as exposure, saturation, tone, color, contrast, sharpness, and
blur.

The trainable component is the
ConditionalParametricResidualMobileNetGenerator. It receives an image and a
conditioning value alpha, predicts transformation parameters, applies the
corresponding image edits, and is trained through a pretrained emotion regressor.
The regressor checks whether the edited image moves toward the desired emotional
target.

The script does the following:
1. Starts Distributed Data Parallel (DDP), so the same training job can run on
   one or more GPUs.
2. Loads COCO training and validation images with the preprocessing expected by
   the parametric model.
3. Loads a pretrained affect/emotion regressor from Gebhardt et al.
4. Optionally loads a CLIP model, which is used as an image-preservation loss.
5. Freezes the regressor and CLIP model by default, because they are used as
   fixed pretrained evaluators and should not be updated during training.
6. Builds the conditional parametric generator, optimizer, loss function,
   distributed samplers, and dataloaders.
7. Optionally resumes from a checkpoint.
8. Optionally runs validation only.
9. Runs training, evaluates on the validation split after each epoch, and writes
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
import clip

from social_media_overlay.ext.gebhardt.dataset.CocoCaptions import CocoCaptions
from social_media_overlay.models.regressor import EmotionRegressor
from social_media_overlay.models.parametric import (
    ConditionalParametricResidualMobileNetGenerator,
)
from social_media_overlay.ddp import (
    setup_ddp,
    load_checkpoint_if_requested,
    cleanup_ddp,
    safe_checkpoint,
)
from social_media_overlay.ddp_train_parametric import train, val

# Root folder of the COCO dataset on disk. The CocoCaptions dataset class expects
# this directory to contain the relevant COCO split folders/files. Change this if
# your local dataset is stored somewhere else.
COCO_DATA_PATH = "../data/coco"
# COCO split used for actual training. This should normally be "train".
TRAIN_SPLIT = "train"
# COCO split used for validation after each epoch, or for validation-only runs.
VAL_SPLIT = "val"

# Size used by transforms.Resize before cropping. Images are resized first and
# then center-cropped below. This should match the input scale expected by the
# parametric generator and the training setup.
# The actual backbone does resizing on its own!
IMAGE_RESIZE_SIZE = 480
# Final square crop size passed into the model. A larger crop keeps more image
# detail but increases GPU memory usage.
# The actual backbone does resizing on its own!
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
# samples, but distributed training may sometimes be simpler with equal batch
# sizes across ranks.
DROP_LAST_BATCH = False

# Number of full passes over the training split. The current value is mainly
# useful for quick tests or debugging; increase it for real training runs.
NUM_EPOCHS = 1
# Optimizer learning rate. This controls the size of Adam update steps and may
# need tuning depending on whether the backbone is frozen or finetuned.
LEARNING_RATE = 0.0001
# How often the training loop prints progress, measured in batches.
PRINT_EVERY_N_BATCHES = 100
# If True, skip training and run only the validation loop. This is useful for
# checking a checkpoint or evaluating a fixed model setup.
ONLY_RUN_VALIDATION = False

# checkpointing
# Directory where checkpoints are written. Rank 0 creates this folder if it does
# not already exist.
CHECKPOINT_DIR = "../data/models/test"
# Prefix used for checkpoint filenames. Change this for each experiment so
# checkpoints are easy to identify.
CHECKPOINT_NAME = "test"
# Optional path to a checkpoint to resume from. If this is empty, training starts
# from scratch.
RESUME_CHECKPOINT_PATH = ""
# Whether to save intermediate/final checkpoints. Keep this True for longer runs
# so training can be resumed after interruption.
SAVE_CHECKPOINTS = True
# Whether to restore the optimizer state when loading a checkpoint. Set this to
# False when starting finetuning from an older checkpoint with a changed set of
# trainable parameters. If continuing from a finetuning checkpoint, set it back
# to True.
LOAD_OPTIMIZER_FROM_CHECKPOINT = (
    True  # False when starting Finetuning from old checkpoint
)
# When starting finetuning: set this to false and unfreeze backbone.
# If you then continue from a finetuning checkpoint you must set this back to true.

# regressor
# Path to the pretrained affect/emotion regressor model from Gebhardt et al. This
# regressor predicts emotional values from edited images and provides the main
# training signal
REGRESSOR_MODEL_PATH = "../data/gebhardt/models/va_pred_all"
# Whether the regressor should average predictions over crops. Crop averaging can
# make predictions more stable, but it is more memory-intensive.
REGRESSOR_USE_AVERAGE = False
# Whether the regressor should internally normalize its input. This should match
# the preprocessing expected by the pretrained regressor.
REGRESSOR_NORMALIZE_INPUT = True
# If True, use the arousal output of the regressor as part of the training target.
# Use arousal see our report!
REGRESSOR_USE_AROUSAL = True
# If True, the regressor weights are frozen. This should normally stay True
# because the regressor is used as a fixed evaluator, not as a trainable model.
FREEZE_REGRESSOR = True

# generator
# Width of the shared fully connected trunk inside the parametric generator.
GENERATOR_FC_DIM = 64
# Depth of the shared fully connected trunk. Larger values increase model
# capacity but also add parameters and training cost.
GENERATOR_FC_DEPTH = 3
# Whether to initialize the MobileNet backbone with pretrained weights. This
# should normally stay True because the backbone is used as an image feature
# extractor.
GENERATOR_USE_PRETRAINED_BACKBONE = (
    True  # Always set to true. Load the initial backbone weights.
)

# IMPORTANT for Finetuning:
# If True, the MobileNet backbone is frozen and only the parametric heads/trunk
# are trained. Set this to False when intentionally finetuning the backbone.
GENERATOR_FREEZE_BACKBONE = True  # Set to false for finetuning

# If the backbone is unfrozen, only the last N MobileNet blocks are kept
# trainable. Earlier blocks are frozen to reduce overfitting and memory usage.
GENERATOR_UNFREEZE_BACKBONE_BLOCKS = (
    1  # Define amount of blocks to unfreeze during finetuning
)

# Probability of sampling alpha=0 during training. These neutral samples are
# useful for encouraging the model to preserve the original image when no edit is
# requested.
ZERO_ALPHA_PROB = (
    0.1
)
# Weight of the additional identity loss. When enabled, this applies an L1 image
# loss for samples where alpha=0, encouraging neutral edits to leave the image
# unchanged. We do not use an identity loss
IDENTITY_LOSS_WEIGHT = 0.0

# If True, target alpha values are clipped to the valid range. This prevents
# impossible targets outside the intended conditioning interval.
CLIP_TARGET = True  # This keeps alphas in possible range [0,1]
# CLIP model variant used for the image-preservation loss.
CLIP_MODEL_NAME = "ViT-B/32"
# Strength of the CLIP-based semantic preservation loss. A value of 0 disables
# CLIP loading and removes this loss term.
CLIP_WEIGHT = (
    0.05
)
# Input size expected by the selected CLIP model. This must match the CLIP
# variant used above.
CLIP_IN_SIZE = 224
# If True, scale the CLIP loss by alpha. This makes the preservation strength
# depend on the requested edit strength. We did not use it.
CLIP_SCALE_BY_ALPHA = False


# Depth of the color-curve prediction head. Use depth=1 with no hidden dimension
# for the simplest checkpoint-compatible head.
COLOR_HEAD_DEPTH = 2
# Hidden dimension of the color-curve head when a deeper head is used.
COLOR_HEAD_DIM = 64
# Depth of the tone-curve prediction head. Use depth=1 with no hidden dimension
# for the simplest checkpoint-compatible head.
TONE_HEAD_DEPTH = 2  # Depth of the fc tone head use 1 with head dim none to disable
# Hidden dimension of the tone-curve head when a deeper head is used.
TONE_HEAD_DIM = 64


# If True, use FiLM-style conditioning, where alpha modulates the image features
# inside the shared trunk. If False, alpha is concatenated with the image feature
# vector and passed through a fully connected trunk.
USE_FILM_CONDITIONING = True
# If True, use the older residual formulation inside the transformation head. We do not recommend this setting.
USE_RES_HEAD = False  # uses a residual formulation within the transformation head
# If True, alpha directly controls the strength of the predicted transformation. We do not recommend this setting.
USE_ALPHA_HEAD = False  # adds alpha to the last transformation function that pushes the output in a range
# If True, use the shifted sigmoid mapping for bounded transformation
# parameters. If False, use the piecewise tanh mapping. The sigmoid version is
# the preferred setting here.
USE_SIDMOID_HEAD = True 

# ddp
# DDP setting controlling whether PyTorch should search for parameters that were
# not used in a forward pass. This can be True for conditional branches or partial
# finetuning, but False is faster when all parameters are always used.
# partial unfreezing + your model structure can trigger unused-parameter issues
# Finetuning: https://www.emergentmind.com/topics/domain-adapted-mobilenetv2-and-mobilenetv3
DDP_FIND_UNUSED_PARAMETERS = True


def main():
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    dist.barrier()

    data_transforms = transforms.Compose(
        [
            transforms.Resize(IMAGE_RESIZE_SIZE),
            transforms.CenterCrop(IMAGE_CROP_SIZE),
            transforms.ToTensor(),
        ]
    )

    train_dataset = CocoCaptions(COCO_DATA_PATH, TRAIN_SPLIT, data_transforms)
    val_dataset = CocoCaptions(COCO_DATA_PATH, VAL_SPLIT, data_transforms)

    regressor = EmotionRegressor(
        path_to_model=REGRESSOR_MODEL_PATH,
        average=REGRESSOR_USE_AVERAGE,
        device=device,
        normalize=REGRESSOR_NORMALIZE_INPUT,
    ).to(device)

    clip_model = None
    if CLIP_WEIGHT > 0:
        clip_model, _ = clip.load(CLIP_MODEL_NAME, device=device)
        # returns also a preprocessing function here compatible with PIL
        # maybe we may use this in the future
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

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

    model = ConditionalParametricResidualMobileNetGenerator(
        fc_dim=GENERATOR_FC_DIM,
        fc_depth=GENERATOR_FC_DEPTH,
        mobilenet_pretrained=GENERATOR_USE_PRETRAINED_BACKBONE,
        freeze_backbone=GENERATOR_FREEZE_BACKBONE,
        color_head_depth=COLOR_HEAD_DEPTH,
        color_head_dim=COLOR_HEAD_DIM,
        tone_head_depth=TONE_HEAD_DEPTH,
        tone_head_dim=TONE_HEAD_DIM,
        use_film=USE_FILM_CONDITIONING,
        use_res_head=USE_RES_HEAD,
        use_alpha_head=USE_ALPHA_HEAD,
        use_sigmoid_head=USE_SIDMOID_HEAD,
    ).to(device)
    model.train()

    if rank == 0:
        print("backbone_input_hw:", model.backbone_input_hw)
        print("backbone_mean:", model.backbone_mean.flatten())
        print("backbone_std:", model.backbone_std.flatten())
        print("timm data config:", resolve_model_data_config(model.backbone))

    loss_fn = MSELoss()

    # initial optimizer, only needed for checkpoint loader
    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
    )

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

    start_epoch, start_batch, loss_history, val_loss_history = (
        load_checkpoint_if_requested(
            model=model,
            optimizer=optimizer,
            device=device,
            rank=rank,
            checkpoint_path=RESUME_CHECKPOINT_PATH,
            load_optimizer=LOAD_OPTIMIZER_FROM_CHECKPOINT,
        )
    )

    # If resuming from a checkpoint that was saved after completing an epoch
    # (especially a "-final.pt"), continue with the NEXT epoch and start at batch 0.
    if RESUME_CHECKPOINT_PATH:
        start_epoch = start_epoch + 1
        start_batch = 0

    if rank == 0:
        print(f"[Resume] start_epoch={start_epoch}, start_batch={start_batch}")

    # Finetuning logic:
    # Start from the model's own working "backbone unfrozen" setup,
    # then refreeze earlier backbone blocks so only the last N remain trainable.
    if not GENERATOR_FREEZE_BACKBONE:
        backbone = model.module.backbone

        if not hasattr(backbone, "blocks"):
            raise AttributeError("Expected backbone to have attribute 'blocks'.")

        total_blocks = len(backbone.blocks)
        start_idx = max(0, total_blocks - GENERATOR_UNFREEZE_BACKBONE_BLOCKS)

        if rank == 0:
            print(f"\n[Finetune] backbone.blocks entries: {total_blocks}")
            print(f"[Finetune] Freezing entire backbone first")
            print(
                f"[Finetune] Unfreezing blocks[{start_idx}:{total_blocks}] plus final backbone head"
            )

        # freeze everything in backbone
        for p in backbone.parameters():
            p.requires_grad = False

        # unfreeze last N blocks
        for block_idx in range(start_idx, total_blocks):
            for p in backbone.blocks[block_idx].parameters():
                p.requires_grad = True
            if rank == 0:
                print(f"UNFREEZE BLOCK: backbone.blocks.{block_idx}")

        # unfreeze final layers after blocks
        if hasattr(backbone, "conv_head"):
            for p in backbone.conv_head.parameters():
                p.requires_grad = True
            if rank == 0:
                print("UNFREEZE: backbone.conv_head")

        if hasattr(backbone, "norm_head"):
            for p in backbone.norm_head.parameters():
                p.requires_grad = True
            if rank == 0:
                print("UNFREEZE: backbone.norm_head")

        optimizer = Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE,
        )
    else:
        optimizer = Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE,
        )

    if rank == 0:
        print("\nTrainable parameters:")
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
                use_arousal=REGRESSOR_USE_AROUSAL,
                identity_loss_weight=IDENTITY_LOSS_WEIGHT,
                zero_alpha_prob=ZERO_ALPHA_PROB,
            )

            dist.barrier()

            if rank == 0:
                print(
                    f"| loss {avg_val_loss['loss']:.6f} "
                    f"| score_loss {avg_val_loss['score_loss']:.6f} "
                    f"| id_loss {avg_val_loss['identity_loss']:.6f} "
                    f"| clip_sem_loss {avg_val_loss['clip_semantic_loss']:.6f} "
                    f"| score_mae {avg_val_loss['score_abs_error']:.6f} "
                    f"| img_l1 {avg_val_loss['image_l1']:.6f} "
                    f"| zero_alpha_l1 {avg_val_loss['zero_alpha_image_l1']:.6f} "
                    f"| clipped_target_frac {avg_val_loss['clipped_target_fraction']:.6f}"
                )

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
                use_arousal=REGRESSOR_USE_AROUSAL,
                identity_loss_weight=IDENTITY_LOSS_WEIGHT,
                zero_alpha_prob=ZERO_ALPHA_PROB,
                clip_model=clip_model,
                clip_semantic_loss_weight=CLIP_WEIGHT,
                clip_loss_scale_by_alpha=CLIP_SCALE_BY_ALPHA,
            )

            dist.barrier()

            val_metrics = val(
                dataloader=val_dataloader,
                model=model,
                regressor=regressor,
                device=device,
                loss_fn=loss_fn,
                use_arousal=REGRESSOR_USE_AROUSAL,
                identity_loss_weight=IDENTITY_LOSS_WEIGHT,
                zero_alpha_prob=ZERO_ALPHA_PROB,
                clip_model=clip_model,
                clip_semantic_loss_weight=CLIP_WEIGHT,
                clip_loss_scale_by_alpha=CLIP_SCALE_BY_ALPHA,
                val_seed=1234,
            )

            dist.barrier()
            val_loss_history.append(val_metrics)

            if rank == 0:
                print(
                    f"Epoch {epoch} validation "
                    f"| loss {val_metrics['loss']:.6f} "
                    f"| score_loss {val_metrics['score_loss']:.6f} "
                    f"| id_loss {val_metrics['identity_loss']:.6f} "
                    f"| clip_sem_loss {val_metrics['clip_semantic_loss']:.6f} "
                    f"| score_mae {val_metrics['score_abs_error']:.6f} "
                    f"| img_l1 {val_metrics['image_l1']:.6f} "
                    f"| zero_alpha_l1 {val_metrics['zero_alpha_image_l1']:.6f} "
                    f"| clipped_target_frac {val_metrics['clipped_target_fraction']:.6f}"
                )

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
                    val_loss_history=val_loss_history,
                    name_suffix="-final.pt",
                )
            dist.barrier()

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
