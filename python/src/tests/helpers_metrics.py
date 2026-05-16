# tests/helpers_metrics.py
from __future__ import annotations

import math
import numpy as np
import torch


def compare_images(
    a: torch.Tensor, b: torch.Tensor, rel_eps: float = 1e-6
) -> dict[str, float]:
    diff = (a - b).abs()
    mse = torch.mean((a - b) ** 2).item()
    rmse = math.sqrt(mse)
    psnr = float("inf") if rmse == 0.0 else 20.0 * math.log10(1.0 / rmse)

    rel = diff / torch.clamp(b.abs(), min=rel_eps)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "mean_rel": rel.mean().item(),
        "rmse": rmse,
        "psnr": psnr,
    }


def assert_equivalent(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    max_abs_tol: float,
    mean_abs_tol: float | None = None,
    min_psnr: float | None = None,
) -> None:
    m = compare_images(a, b)
    assert m["max_abs"] <= max_abs_tol, m
    if mean_abs_tol is not None:
        assert m["mean_abs"] <= mean_abs_tol, m
    if min_psnr is not None:
        assert m["psnr"] >= min_psnr, m


def bootstrap_mean_ci(
    values: list[float], *, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boot[i] = v[rng.integers(0, n, size=n)].mean()
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)
