import pytest
import torch
from torch import Tensor

from social_media_overlay.ext.gebhardt.image_transformations.kornia_monkey_patch import safe_blend

def original_blend_one(input1: Tensor, input2: Tensor, factor: Tensor) -> Tensor:
    if not isinstance(input1, Tensor):
        raise AssertionError(f"`input1` must be a tensor. Got {input1}.")
    if not isinstance(input2, Tensor):
        raise AssertionError(f"`input1` must be a tensor. Got {input2}.")

    if isinstance(factor, Tensor) and len(factor.size()) != 0:
        raise AssertionError(f"Factor shall be a float or single element tensor. Got {factor}.")
    if factor == 0.0:
        return input1
    if factor == 1.0:
        return input2
    diff = (input2 - input1) * factor
    res = input1 + diff
    if factor > 0.0 and factor < 1.0:
        return res
    return torch.clamp(res, 0, 1)


@pytest.mark.parametrize("shape", [(4, 5), (3, 4, 5)])
@pytest.mark.parametrize("factor_value", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_safe_blend_matches_original_for_scalar_factor_in_unit_interval(shape, factor_value):
    input1 = torch.rand(*shape)
    input2 = torch.rand(*shape)
    factor = torch.tensor(factor_value)

    expected = original_blend_one(input1, input2, factor)
    actual = safe_blend(input1, input2, factor)

    assert torch.allclose(actual, expected)


def test_safe_blend_matches_original_exactly_for_factor_zero():
    input1 = torch.rand(3, 4)
    input2 = torch.rand(3, 4)
    factor = torch.tensor(0.0)

    expected = original_blend_one(input1, input2, factor)
    actual = safe_blend(input1, input2, factor)

    assert torch.equal(actual, expected)
    assert torch.equal(actual, input1)


def test_safe_blend_matches_original_exactly_for_factor_one():
    input1 = torch.rand(3, 4)
    input2 = torch.rand(3, 4)
    factor = torch.tensor(1.0)

    expected = original_blend_one(input1, input2, factor)
    actual = safe_blend(input1, input2, factor)

    assert torch.equal(actual, expected)
    assert torch.equal(actual, input2)


@pytest.mark.parametrize("factor_value", [0.1, 0.3, 0.9])
def test_safe_blend_matches_original_for_nontrivial_interpolation(factor_value):
    input1 = torch.tensor([[0.2, 0.4], [0.6, 0.8]])
    input2 = torch.tensor([[0.8, 0.6], [0.4, 0.2]])
    factor = torch.tensor(factor_value)

    expected = original_blend_one(input1, input2, factor)
    actual = safe_blend(input1, input2, factor)

    assert torch.allclose(actual, expected)


def test_safe_blend_preserves_dtype_like_original_for_scalar_factor():
    input1 = torch.rand(2, 3, dtype=torch.float32)
    input2 = torch.rand(2, 3, dtype=torch.float32)
    factor = torch.tensor(0.5, dtype=torch.float64)

    expected = original_blend_one(input1, input2, factor.to(dtype=input1.dtype))
    actual = safe_blend(input1, input2, factor)

    assert actual.dtype == expected.dtype == input1.dtype
    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("factor_value", [-1.0, -0.5, 1.5, 2.0])
def test_safe_blend_differs_from_original_outside_unit_interval_due_to_missing_clamp(factor_value):
    input1 = torch.tensor([0.2, 0.4, 0.6])
    input2 = torch.tensor([0.8, 0.6, 0.4])
    factor = torch.tensor(factor_value)

    expected = original_blend_one(input1, input2, factor)
    actual = safe_blend(input1, input2, factor)

    assert not torch.allclose(actual, expected)


def test_original_rejects_non_scalar_factor_but_safe_blend_accepts_it():
    input1 = torch.zeros(2, 3, 4, 4)
    input2 = torch.ones(2, 3, 4, 4)
    factor = torch.tensor([0.0, 1.0])

    with pytest.raises(AssertionError, match="Factor shall be a float or single element tensor"):
        original_blend_one(input1, input2, factor)

    actual = safe_blend(input1, input2, factor)

    assert torch.allclose(actual[0], torch.zeros(3, 4, 4))
    assert torch.allclose(actual[1], torch.ones(3, 4, 4))