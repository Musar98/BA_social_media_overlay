"""
Validation metrics and reporting utilities for image transformation experiments.

Note on implementation:
    Large parts of this file, especially the less central utility logic around
    metric collection, fetching/intermediate handling, formatting, progress
    reporting, latency measurement, and distributed aggregation, were
    generated with AI.

    This is considered acceptable because these parts mainly support
    validation bookkeeping and reporting. The important part of the work is
    not the automatic collection code itself, but the resulting numbers and
    comparisons, which are interpreted and analysed manually.

Usage:
    This file is used during validation to measure how an image transformation
    model changes visual image properties, emotion-regression predictions,
    image distribution quality, latency, throughput, and VRAM usage.

    Typical usage:
        1. Call val_image_attr_and_distribution(...) with only_original=True
           to collect baseline metrics for the original validation images.
        2. Call val_image_attr_and_distribution(...) again with transformed
           images, usually once with alpha=-0.1 and once with alpha=+0.1.
        3. Pass the returned metric dictionaries into print_validation_summary(...)
           to print a readable comparison table.


    The validation loss alone does not show whether transformed images are
    visually stable, emotionally meaningful, distributionally similar, or fast
    enough to generate. This file collects additional validation metrics so that
    different model versions, alpha values, training runs, and hardware setups
    can be compared more reliably.

    In particular, this file measures:
        - low-level image properties such as brightness, saturation, contrast,
          blur, lighting diversity, and colorfulness
        - valence/arousal prediction changes using an external emotion regressor
        - pixel-level image distance between original and transformed images
        - distribution similarity using FID and KID
        - model latency, regressor latency, throughput, and VRAM usage
"""

import time
from typing import Any, Callable
import torch
import torch.distributed as dist
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from social_media_overlay.low_level_image_metrics import val_image_properties
from social_media_overlay.utils import to_tensor
from tqdm.auto import tqdm


def print_validation_summary(
    avg_val_loss: float,
    val_attr_original: dict[str, Any],
    val_attr_neg: dict[str, Any],
    val_attr_pos: dict[str, Any],
) -> None:
    """
    Print a formatted validation summary for original and transformed images.

    This function expects metric dictionaries returned by
    val_image_attr_and_distribution(...). It compares the original-image
    baseline with two transformed-image runs, normally alpha=-0.1 and alpha=+0.1.

    Args:
        avg_val_loss:
            Average validation loss from the validation loop.
        val_attr_original:
            Metrics for original images only, usually collected with
            only_original=True.
        val_attr_neg:
            Metrics for transformed images with negative alpha, usually alpha=-0.1.
        val_attr_pos:
            Metrics for transformed images with positive alpha, usually alpha=+0.1.

    Returns:
        None. The function prints the summary directly to stdout.
    """

    def _fmt(v: Any) -> str:
        """
        Format metric values for printing.

        Floats are shown with six decimal places. Other values are converted
        directly to strings.
        """
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    # Image-property metrics computed for original and transformed images.
    original_metric_names = [
        "colorfulness",
        "mean_brightness",
        "mean_saturation",
        "rms_contrast",
        "lighting_diversity",
        "blur_effect",
    ]

    # Prediction-distance and pixel-distance metrics.
    distance_metric_names = [
        "valence_mse",
        "valence_mae",
        "arousal_mse",
        "arousal_mae",
        "image_l1",
    ]

    # Runtime, throughput, and memory metrics.
    latency_metric_names = [
        "model_latency_ms",
        "regressor_original_latency_ms",
        "regressor_transformed_latency_ms",
        "total_batch_latency_ms",
        "samples_per_second",
        "used_vram_mb",
        "peak_vram_mb",
    ]

    print("\n" + "=" * 100)
    print("VALIDATION SUMMARY")
    print("=" * 100)
    print(f"Validation loss: {avg_val_loss:.6f}")

    print("\n" + "=" * 100)
    print("ORIGINAL IMAGES")
    print("=" * 100)
    print(f"{'metric':<30} {'mean':>14} {'std':>14}")
    print("-" * 60)

    for name in original_metric_names:
        mean_key = f"original_{name}_mean"
        std_key = f"original_{name}_std"
        print(
            f"{name:<30} "
            f"{_fmt(val_attr_original[mean_key]):>14} "
            f"{_fmt(val_attr_original[std_key]):>14}"
        )

    print(f"\nnum_samples: {val_attr_original['num_samples']}")
    print(f"only_original: {val_attr_original['only_original']}")

    print("\n" + "=" * 100)
    print("TRANSFORMED IMAGE METRICS COMPARISON")
    print("=" * 100)
    print(
        f"{'metric':<30} "
        f"{'alpha=-0.1 mean':>18} {'alpha=-0.1 std':>18} "
        f"{'alpha=+0.1 mean':>18} {'alpha=+0.1 std':>18}"
    )
    print("-" * 105)

    for name in original_metric_names:
        neg_mean_key = f"transformed_{name}_mean"
        neg_std_key = f"transformed_{name}_std"
        pos_mean_key = f"transformed_{name}_mean"
        pos_std_key = f"transformed_{name}_std"

        print(
            f"{name:<30} "
            f"{_fmt(val_attr_neg[neg_mean_key]):>18} "
            f"{_fmt(val_attr_neg[neg_std_key]):>18} "
            f"{_fmt(val_attr_pos[pos_mean_key]):>18} "
            f"{_fmt(val_attr_pos[pos_std_key]):>18}"
        )

    print("\n" + "=" * 100)
    print("PREDICTION / DISTANCE METRICS COMPARISON")
    print("=" * 100)
    print(
        f"{'metric':<30} "
        f"{'alpha=-0.1 mean':>18} {'alpha=-0.1 std':>18} "
        f"{'alpha=+0.1 mean':>18} {'alpha=+0.1 std':>18}"
    )
    print("-" * 105)

    for name in distance_metric_names:
        neg_mean_key = f"{name}_mean"
        neg_std_key = f"{name}_std"
        pos_mean_key = f"{name}_mean"
        pos_std_key = f"{name}_std"

        print(
            f"{name:<30} "
            f"{_fmt(val_attr_neg[neg_mean_key]):>18} "
            f"{_fmt(val_attr_neg[neg_std_key]):>18} "
            f"{_fmt(val_attr_pos[pos_mean_key]):>18} "
            f"{_fmt(val_attr_pos[pos_std_key]):>18}"
        )

    print("\n" + "=" * 100)
    print("DISTRIBUTION METRICS")
    print("=" * 100)
    print(f"{'metric':<30} {'alpha=-0.1':>18} {'alpha=+0.1':>18}")
    print("-" * 70)

    distribution_keys = ["fid", "kid_mean", "kid_std"]
    for key in distribution_keys:
        print(
            f"{key:<30} "
            f"{_fmt(val_attr_neg[key]):>18} "
            f"{_fmt(val_attr_pos[key]):>18}"
        )

    print("\n" + "=" * 100)
    print("LATENCY / THROUGHPUT / VRAM METRICS")
    print("=" * 100)
    print(
        f"{'metric':<35} "
        f"{'alpha=-0.1 mean':>16} {'p50':>12} {'p95':>12} {'p99':>12} "
        f"{'alpha=+0.1 mean':>16} {'p50':>12} {'p95':>12} {'p99':>12}"
    )
    print("-" * 135)

    for name in latency_metric_names:
        neg_mean_key = f"{name}_mean"
        neg_p50_key = f"{name}_p50"
        neg_p95_key = f"{name}_p95"
        neg_p99_key = f"{name}_p99"

        pos_mean_key = f"{name}_mean"
        pos_p50_key = f"{name}_p50"
        pos_p95_key = f"{name}_p95"
        pos_p99_key = f"{name}_p99"

        print(
            f"{name:<35} "
            f"{_fmt(val_attr_neg.get(neg_mean_key, 0.0)):>16} "
            f"{_fmt(val_attr_neg.get(neg_p50_key, 0.0)):>12} "
            f"{_fmt(val_attr_neg.get(neg_p95_key, 0.0)):>12} "
            f"{_fmt(val_attr_neg.get(neg_p99_key, 0.0)):>12} "
            f"{_fmt(val_attr_pos.get(pos_mean_key, 0.0)):>16} "
            f"{_fmt(val_attr_pos.get(pos_p50_key, 0.0)):>12} "
            f"{_fmt(val_attr_pos.get(pos_p95_key, 0.0)):>12} "
            f"{_fmt(val_attr_pos.get(pos_p99_key, 0.0)):>12}"
        )

    print("\nMeasured latency/VRAM batches:")
    print(f"alpha=-0.1: {val_attr_neg.get('model_latency_ms_num_batches_measured', 0)}")
    print(f"alpha=+0.1: {val_attr_pos.get('model_latency_ms_num_batches_measured', 0)}")

    print("\n" + "=" * 100)
    print("SANITY CHECK, ORIGINAL METRICS INSIDE ALL RUNS")
    print("=" * 100)
    print(
        f"{'metric':<30} "
        f"{'original only':>18} {'from alpha=-0.1':>18} {'from alpha=+0.1':>18}"
    )
    print("-" * 90)

    for name in original_metric_names:
        key = f"original_{name}_mean"
        print(
            f"{name:<30} "
            f"{_fmt(val_attr_original[key]):>18} "
            f"{_fmt(val_attr_neg[key]):>18} "
            f"{_fmt(val_attr_pos[key]):>18}"
        )


def val_image_attr_and_distribution(
    dataloader: Any,
    model: torch.nn.Module,
    regressor: Any,
    device: torch.device,
    alpha: float,
    only_original: bool = False,
    measure_latency: bool = True,
    latency_warmup_batches: int = 3,
    show_progress: bool = False,
    status_update_freq: int = 10,
) -> dict[str, Any]:
    """
    Compute validation metrics for original and transformed image batches.

    In original-only mode, this function only computes low-level image metrics
    for the original validation images.

    In full mode, this function applies the transformation model with the given
    alpha value and computes:
        - original and transformed image-property metrics
        - valence/arousal prediction differences
        - pixel-level L1 image distance
        - FID and KID distribution metrics
        - latency, throughput, and VRAM metrics

    Args:
        dataloader:
            Validation dataloader yielding batches in the form `(X, y)`.
            The labels are ignored.
        model:
            Image transformation model. Expected to be callable as
            `model(X, alphas)` and return transformed images as the first output.
        regressor:
            Emotion regressor with a `.predict(images)` method.
        device:
            Torch device used for validation.
        alpha:
            Transformation strength/direction passed to the model.
        only_original:
            If True, skip transformation and only measure original image metrics.
        measure_latency:
            If True, measure model/regressor latency and VRAM usage.
        latency_warmup_batches:
            Number of initial batches skipped before recording latency metrics.
        show_progress:
            If True, show a tqdm progress bar on the main process.
        status_update_freq:
            Number of batches between progress updates.

    Returns:
        A dictionary containing validation metrics and metadata.
    """
    model.eval()

    # Detect whether the validation is running under torch.distributed.
    # In distributed mode, counts and sums are reduced across all workers.
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1
    is_main_process = rank == 0

    # Low-level image metrics computed from image tensors.
    metric_names = [
        "colorfulness",
        "mean_brightness",
        "mean_saturation",
        "rms_contrast",
        "lighting_diversity",
        "blur_effect",
    ]

    # In original-only mode, only original images are evaluated.
    # Otherwise, both original and transformed images are measured.
    image_prefixes = ["original"] if only_original else ["original", "transformed"]
    # Distance metrics are only available when transformed images exist.
    distance_names = (
        []
        if only_original
        else [
            "valence_mse",
            "valence_mae",
            "arousal_mse",
            "arousal_mae",
            "image_l1",
        ]
    )
    # Latency metrics are skipped in original-only mode or when disabled.
    latency_names = (
        []
        if only_original or not measure_latency
        else [
            "model_latency_ms",
            "regressor_original_latency_ms",
            "regressor_transformed_latency_ms",
            "total_batch_latency_ms",
            "samples_per_second",
            "used_vram_mb",
            "peak_vram_mb",
        ]
    )

    def _infer_total_samples() -> int | None:
        """
        Infer the total number of validation samples from the dataloader.

        Returns:
            Dataset length if available, otherwise None.
        """
        dataset = getattr(dataloader, "dataset", None)
        if dataset is not None:
            return int(len(dataset))
        return None

    def _init_running_stats() -> dict[str, torch.Tensor]:
        """
        Create a running-statistics dictionary.

        The dictionary stores count, sum, and squared sum. These values are enough
        to compute mean and standard deviation without storing every sample value.
        """
        return {
            "count": torch.tensor([0.0], device=device, dtype=torch.float64),
            "sum": torch.tensor([0.0], device=device, dtype=torch.float64),
            "sum_sq": torch.tensor([0.0], device=device, dtype=torch.float64),
        }

    def _update_running_stats(
        running_stats: dict[str, torch.Tensor],
        values: torch.Tensor,
    ) -> None:
        """
        Update running statistics with a tensor of values.

        Args:
            running_stats:
                Dictionary containing count, sum, and sum_sq tensors.
            values:
                Tensor containing values to add to the running statistics.
        """
        values = values.detach().reshape(-1).to(torch.float64)

        running_stats["count"] += torch.tensor(
            [values.numel()], device=device, dtype=torch.float64
        )
        running_stats["sum"] += values.sum().unsqueeze(0)
        running_stats["sum_sq"] += (values**2).sum().unsqueeze(0)

    def _finalize_stats(
        running_stats: dict[str, torch.Tensor],
    ) -> tuple[float, float, int]:
        """
        Convert running statistics into mean, standard deviation, and count.

        Returns:
            Tuple of `(mean, std, count)`.
        """
        count = running_stats["count"].item()

        if count <= 0:
            return 0.0, 0.0, 0

        mean = running_stats["sum"].item() / count
        var = running_stats["sum_sq"].item() / count - mean * mean
        var = max(var, 0.0)
        std = var**0.5
        return float(mean), float(std), int(count)

    def _to_uint8_for_inception(images: torch.Tensor) -> torch.Tensor:
        """
        Convert image tensors from float `[0, 1]` format to uint8 `[0, 255]`.

        This format is used before passing images to FID and KID metrics.
        """
        images = images.detach().clamp(0.0, 1.0)
        images = (images * 255.0).round().to(torch.uint8)
        return images

    def _sync_if_needed() -> None:
        """
        Synchronize CUDA operations when running on GPU.

        CUDA operations are asynchronous, so synchronization is needed for more
        accurate timing measurements.
        """
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _timed_call(fn: Callable[[], Any]) -> tuple[Any, float]:
        """
        Run a function and measure its execution time in milliseconds.

        Args:
            fn:
                Zero-argument callable to execute.

        Returns:
            Tuple containing the callable output and elapsed milliseconds.
        """
        _sync_if_needed()
        start = time.perf_counter()
        out = fn()
        _sync_if_needed()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return out, elapsed_ms

    def _compute_percentiles(values: list[float]) -> tuple[float, float, float]:
        """
        Compute p50, p95, and p99 percentiles from recorded latency values.

        Returns:
            Tuple of `(p50, p95, p99)`.
        """
        if len(values) == 0:
            return 0.0, 0.0, 0.0

        t = torch.tensor(values, dtype=torch.float64)
        p50 = torch.quantile(t, 0.50).item()
        p95 = torch.quantile(t, 0.95).item()
        p99 = torch.quantile(t, 0.99).item()
        return float(p50), float(p95), float(p99)

    def _get_vram_stats_mb() -> tuple[float, float]:
        """
        Return current and peak CUDA memory usage in megabytes.

        Returns:
            Tuple of `(used_vram_mb, peak_vram_mb)`. On CPU, both values are 0.
        """
        if device.type != "cuda":
            return 0.0, 0.0

        used_bytes = torch.cuda.memory_allocated(device)
        peak_bytes = torch.cuda.max_memory_allocated(device)

        used_mb = used_bytes / (1024**2)
        peak_mb = peak_bytes / (1024**2)
        return float(used_mb), float(peak_mb)

    def _ddp_sum_scalar(value: float | int) -> float:
        """
        Sum a scalar across all distributed workers.

        Args:
            value:
                Local scalar value.

        Returns:
            Globally summed scalar if distributed validation is active,
            otherwise the original local value.
        """
        t = torch.tensor([value], device=device, dtype=torch.float64)
        if is_distributed:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.item()
    # Running statistics for image-property metrics.
    running = {
        prefix: {name: _init_running_stats() for name in metric_names}
        for prefix in image_prefixes
    }
    # Add running statistics for distance and latency metrics.
    running.update({name: _init_running_stats() for name in distance_names})
    running.update({name: _init_running_stats() for name in latency_names})
    # Store raw latency values so percentiles can be computed later.
    latency_samples = {name: [] for name in latency_names}
    # FID/KID are only computed when original and transformed images are compared.
    if not only_original:
        fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        kid_metric = KernelInceptionDistance(
            feature=2048,
            subset_size=50,
            normalize=False,
        ).to(device)

    total_samples = _infer_total_samples()

    pbar = None
    if show_progress and is_main_process:
        desc = f"Validation alpha={alpha:.3f}"
        if only_original:
            desc += " original-only"
        pbar = tqdm(
            total=total_samples,
            desc=desc,
            dynamic_ncols=True,
            leave=True,
        )

    global_seen = 0
    local_seen_since_report = 0
    report_start_time = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (X, _) in enumerate(dataloader):
            X = X.to(device, non_blocking=True)

            batch_size_local = X.size(0)
            local_seen_since_report += batch_size_local
            # Always compute metrics for the original images.
            original_metrics = val_image_properties(X)
            for name in metric_names:
                _update_running_stats(running["original"][name], original_metrics[name])

            if only_original:
                should_report = show_progress and (
                    (batch_idx + 1) % status_update_freq == 0
                    or (batch_idx + 1) == len(dataloader)
                )

                if should_report:
                    elapsed = max(time.perf_counter() - report_start_time, 1e-8)
                    global_processed_since_report = _ddp_sum_scalar(
                        local_seen_since_report
                    )
                    global_sps = global_processed_since_report / elapsed

                    if is_main_process and pbar is not None:
                        delta = int(global_processed_since_report)
                        global_seen += delta

                        if pbar.total is not None:
                            new_n = min(global_seen, pbar.total)
                            pbar.update(new_n - pbar.n)
                        else:
                            pbar.update(delta)

                        postfix = {
                            "img/s": f"{global_sps:.1f}",
                            "seen": int(global_seen),
                        }

                        if device.type == "cuda":
                            used_vram_mb, peak_vram_mb = _get_vram_stats_mb()
                            postfix["gpu_mb"] = f"{used_vram_mb:.0f}"
                            postfix["peak_mb"] = f"{peak_vram_mb:.0f}"

                        pbar.set_postfix(postfix)

                    local_seen_since_report = 0
                    report_start_time = time.perf_counter()

                continue
            # Create one alpha value per input image.
            alpha_value = float(alpha)
            alphas = torch.full(
                (X.size(0),),
                fill_value=alpha_value,
                device=device,
                dtype=X.dtype,
            )

            should_record_latency = (
                measure_latency and batch_idx >= latency_warmup_batches
            )

            if should_record_latency and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            if should_record_latency:
                _sync_if_needed()
                batch_start = time.perf_counter()

                (transformed_images, _), model_latency_ms = _timed_call(
                    lambda: model(X, alphas)
                )

                original_preds_raw, regressor_original_latency_ms = _timed_call(
                    lambda: regressor.predict(X)
                )

                transformed_preds_raw, regressor_transformed_latency_ms = _timed_call(
                    lambda: regressor.predict(transformed_images)
                )

                _sync_if_needed()
                total_batch_latency_ms = (time.perf_counter() - batch_start) * 1000.0

                batch_size = X.size(0)
                sps = batch_size / (total_batch_latency_ms / 1000.0)

                used_vram_mb, peak_vram_mb = _get_vram_stats_mb()

                latency_values = {
                    "model_latency_ms": model_latency_ms,
                    "regressor_original_latency_ms": regressor_original_latency_ms,
                    "regressor_transformed_latency_ms": regressor_transformed_latency_ms,
                    "total_batch_latency_ms": total_batch_latency_ms,
                    "samples_per_second": sps,
                    "used_vram_mb": used_vram_mb,
                    "peak_vram_mb": peak_vram_mb,
                }

                for name, value in latency_values.items():
                    _update_running_stats(
                        running[name],
                        torch.tensor([value], device=device, dtype=torch.float64),
                    )
                    latency_samples[name].append(float(value))
            else:
                transformed_images, _ = model(X, alphas)
                original_preds_raw = regressor.predict(X)
                transformed_preds_raw = regressor.predict(transformed_images)

            transformed_metrics = val_image_properties(transformed_images)
            for name in metric_names:
                _update_running_stats(
                    running["transformed"][name], transformed_metrics[name]
                )

            original_preds = to_tensor(original_preds_raw, device, X.dtype)
            transformed_preds = to_tensor(transformed_preds_raw, device, X.dtype)

            if original_preds.ndim == 1:
                original_preds = original_preds.unsqueeze(1)
            if transformed_preds.ndim == 1:
                transformed_preds = transformed_preds.unsqueeze(1)

            original_valence = original_preds[:, :1]
            original_arousal = original_preds[:, 1:2]

            transformed_valence = transformed_preds[:, :1]
            transformed_arousal = transformed_preds[:, 1:2]

            valence_diff = transformed_valence - original_valence
            valence_mse = (valence_diff**2).mean(dim=1)
            valence_mae = valence_diff.abs().mean(dim=1)

            arousal_diff = transformed_arousal - original_arousal
            arousal_mse = (arousal_diff**2).mean(dim=1)
            arousal_mae = arousal_diff.abs().mean(dim=1)

            image_l1 = (transformed_images - X).abs().flatten(1).mean(dim=1)

            _update_running_stats(running["valence_mse"], valence_mse)
            _update_running_stats(running["valence_mae"], valence_mae)
            _update_running_stats(running["arousal_mse"], arousal_mse)
            _update_running_stats(running["arousal_mae"], arousal_mae)
            _update_running_stats(running["image_l1"], image_l1)

            original_uint8 = _to_uint8_for_inception(X)
            transformed_uint8 = _to_uint8_for_inception(transformed_images)

            fid_metric.update(original_uint8, real=True)
            fid_metric.update(transformed_uint8, real=False)

            kid_metric.update(original_uint8, real=True)
            kid_metric.update(transformed_uint8, real=False)

            should_report = show_progress and (
                (batch_idx + 1) % status_update_freq == 0
                or (batch_idx + 1) == len(dataloader)
            )

            if should_report:
                elapsed = max(time.perf_counter() - report_start_time, 1e-8)
                global_processed_since_report = _ddp_sum_scalar(local_seen_since_report)
                global_sps = global_processed_since_report / elapsed

                if is_main_process and pbar is not None:
                    delta = int(global_processed_since_report)
                    global_seen += delta

                    if pbar.total is not None:
                        new_n = min(global_seen, pbar.total)
                        pbar.update(new_n - pbar.n)
                    else:
                        pbar.update(delta)

                    postfix = {
                        "img/s": f"{global_sps:.1f}",
                        "seen": int(global_seen),
                    }

                    if device.type == "cuda":
                        used_vram_mb, peak_vram_mb = _get_vram_stats_mb()
                        postfix["gpu_mb"] = f"{used_vram_mb:.0f}"
                        postfix["peak_mb"] = f"{peak_vram_mb:.0f}"

                    pbar.set_postfix(postfix)

                local_seen_since_report = 0
                report_start_time = time.perf_counter()

    if is_main_process and pbar is not None:
        pbar.close()

    if is_distributed:
        for prefix in image_prefixes:
            for name in metric_names:
                packed = torch.cat(
                    [
                        running[prefix][name]["count"],
                        running[prefix][name]["sum"],
                        running[prefix][name]["sum_sq"],
                    ],
                    dim=0,
                )
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                running[prefix][name]["count"] = packed[0:1]
                running[prefix][name]["sum"] = packed[1:2]
                running[prefix][name]["sum_sq"] = packed[2:3]

        for name in distance_names + latency_names:
            packed = torch.cat(
                [
                    running[name]["count"],
                    running[name]["sum"],
                    running[name]["sum_sq"],
                ],
                dim=0,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            running[name]["count"] = packed[0:1]
            running[name]["sum"] = packed[1:2]
            running[name]["sum_sq"] = packed[2:3]

        if latency_names:
            for name in latency_names:
                local_tensor = torch.tensor(
                    latency_samples[name], device=device, dtype=torch.float64
                )

                local_len = torch.tensor(
                    [local_tensor.numel()], device=device, dtype=torch.long
                )
                gathered_lens = [torch.zeros_like(local_len) for _ in range(world_size)]
                dist.all_gather(gathered_lens, local_len)

                max_len = int(max(x.item() for x in gathered_lens))

                if local_tensor.numel() < max_len:
                    pad = torch.zeros(
                        max_len - local_tensor.numel(),
                        device=device,
                        dtype=torch.float64,
                    )
                    local_tensor = torch.cat([local_tensor, pad], dim=0)

                gathered_vals = [
                    torch.zeros_like(local_tensor) for _ in range(world_size)
                ]
                dist.all_gather(gathered_vals, local_tensor)

                merged = []
                for gathered_tensor, gathered_len in zip(gathered_vals, gathered_lens):
                    valid_len = int(gathered_len.item())
                    if valid_len > 0:
                        merged.extend(gathered_tensor[:valid_len].cpu().tolist())

                latency_samples[name] = merged

    result = {}
    num_samples = None

    for prefix in image_prefixes:
        for name in metric_names:
            mean, std, count = _finalize_stats(running[prefix][name])
            result[f"{prefix}_{name}_mean"] = mean
            result[f"{prefix}_{name}_std"] = std

            if num_samples is None:
                num_samples = count

    for name in distance_names:
        mean, std, count = _finalize_stats(running[name])
        result[f"{name}_mean"] = mean
        result[f"{name}_std"] = std

        if num_samples is None:
            num_samples = count

    for name in latency_names:
        mean, std, count = _finalize_stats(running[name])
        p50, p95, p99 = _compute_percentiles(latency_samples[name])

        result[f"{name}_mean"] = mean
        result[f"{name}_std"] = std
        result[f"{name}_p50"] = p50
        result[f"{name}_p95"] = p95
        result[f"{name}_p99"] = p99
        result[f"{name}_num_batches_measured"] = count

    if not only_original:
        fid_value = fid_metric.compute()
        kid_mean, kid_std = kid_metric.compute()

        result["fid"] = float(
            fid_value.item() if torch.is_tensor(fid_value) else fid_value
        )
        result["kid_mean"] = float(
            kid_mean.item() if torch.is_tensor(kid_mean) else kid_mean
        )
        result["kid_std"] = float(
            kid_std.item() if torch.is_tensor(kid_std) else kid_std
        )

    result["num_samples"] = 0 if num_samples is None else num_samples
    result["alpha"] = float(alpha)
    result["only_original"] = bool(only_original)
    result["measure_latency"] = bool(measure_latency)
    result["latency_warmup_batches"] = int(latency_warmup_batches)

    return result
