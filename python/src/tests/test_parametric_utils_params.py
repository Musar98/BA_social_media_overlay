import pytest
import torch

from social_media_overlay.models.parametric_utils import (
    PARAMS,
    flatten_output_params,
    unflatten_output_params,
)


def apply_params_copy(im: list[str], params: dict[str, object]) -> list[list[str]]:
    """
    Copy implementation of the ordering structure used by Gebhardt et al.'s
    `apply_params`.

    This intentionally does not perform real image transformations. Instead, it
    records the order in which transformations are executed. The goal is to test
    that the effective execution order follows the insertion order of the
    `params` dictionary.
    """
    param_names = list(params.keys())
    im_list: list[list[str]] = []

    for i in range(len(param_names)):
        if "gamma" == param_names[i]:
            im = im + ["gamma"]
        elif "sharp" == param_names[i]:
            im = im + ["sharp"]
        elif "wb" == param_names[i]:
            im = im + ["wb"]
        elif "exposure" == param_names[i]:
            im = im + ["exposure"]
        elif "bright" == param_names[i]:
            im = im + ["bright"]
        elif "contrast" == param_names[i]:
            im = im + ["contrast"]
        elif "saturation" == param_names[i]:
            im = im + ["saturation"]
        elif "bw" == param_names[i]:
            im = im + ["bw"]
        elif "tone" == param_names[i]:
            im = im + ["tone"]
        elif "hue" == param_names[i]:
            im = im + ["hue"]
        elif "color" == param_names[i]:
            im = im + ["color"]
        elif "blur" == param_names[i]:
            im = im + ["blur"]
        elif "affine" == param_names[i]:
            im = im + ["affine"]
        elif "scale" == param_names[i]:
            im = im + ["scale"]

        # Keep this line to mirror the original structure.
        # In the real implementation this clamps the image tensor.
        _ = torch.clamp(torch.tensor([0.0]), min=0.0, max=1.0)

        if i == len(param_names) - 1:
            im_list.append(im)
        else:
            im_list.append(im.copy())

    return im_list


def make_params(batch_size: int = 2) -> dict[str, torch.Tensor]:
    """
    Create transformation parameters with identifiable values.

    The values are chosen so that flattening and unflattening can be checked
    exactly.
    """
    return {
        "exposure": torch.tensor([1.0, 2.0]),
        "saturation": torch.tensor([3.0, 4.0]),
        "tone": torch.arange(batch_size * 8, dtype=torch.float32).reshape(
            batch_size, 1, 8, 1
        )
        + 10.0,
        "color": torch.arange(batch_size * 24, dtype=torch.float32).reshape(
            batch_size, 3, 8, 1
        )
        + 100.0,
        "contrast": torch.tensor([5.0, 6.0]),
        "sharp": torch.tensor([7.0, 8.0]),
        "blur": torch.tensor([9.0, 10.0]),
    }


def test_apply_params_copy_follows_param_insertion_order() -> None:
    params = {name: object() for name in PARAMS}

    outputs = apply_params_copy([], params)

    assert outputs[-1] == PARAMS


def test_apply_params_copy_does_not_follow_if_elif_code_order() -> None:
    custom_order = [
        "blur",
        "sharp",
        "exposure",
        "tone",
        "contrast",
        "color",
        "saturation",
    ]
    params = {name: object() for name in custom_order}

    outputs = apply_params_copy([], params)

    assert outputs[-1] == custom_order
    assert outputs[-1] != PARAMS


def test_intermediate_outputs_preserve_progressive_execution_order() -> None:
    params = {name: object() for name in PARAMS}

    outputs = apply_params_copy([], params)

    assert outputs == [
        ["exposure"],
        ["exposure", "saturation"],
        ["exposure", "saturation", "tone"],
        ["exposure", "saturation", "tone", "color"],
        ["exposure", "saturation", "tone", "color", "contrast"],
        ["exposure", "saturation", "tone", "color", "contrast", "sharp"],
        ["exposure", "saturation", "tone", "color", "contrast", "sharp", "blur"],
    ]


def test_flatten_output_params_has_expected_shape() -> None:
    batch_size = 2
    params = make_params(batch_size)

    flat = flatten_output_params(params, batch_size)

    assert flat.shape == (batch_size, 37)


def test_flatten_output_params_uses_param_order() -> None:
    batch_size = 2
    params = make_params(batch_size)

    flat = flatten_output_params(params, batch_size)

    expected = torch.cat(
        [
            params["exposure"].reshape(batch_size, -1),
            params["saturation"].reshape(batch_size, -1),
            params["tone"].reshape(batch_size, -1),
            params["color"].reshape(batch_size, -1),
            params["contrast"].reshape(batch_size, -1),
            params["sharp"].reshape(batch_size, -1),
            params["blur"].reshape(batch_size, -1),
        ],
        dim=1,
    )

    torch.testing.assert_close(flat, expected)


def test_flatten_output_params_ignores_unknown_keys() -> None:
    batch_size = 2
    params = make_params(batch_size)
    params["unknown"] = torch.ones(batch_size, 99)

    flat = flatten_output_params(params, batch_size)

    assert flat.shape == (batch_size, 37)


def test_flatten_output_params_returns_empty_tensor_for_empty_params() -> None:
    batch_size = 2

    flat = flatten_output_params({}, batch_size)

    assert flat.shape == (batch_size, 0)


def test_unflatten_output_params_reconstructs_expected_shapes() -> None:
    batch_size = 2
    params = make_params(batch_size)
    flat = flatten_output_params(params, batch_size)

    reconstructed = unflatten_output_params(flat)

    assert reconstructed["exposure"].shape == (batch_size,)
    assert reconstructed["saturation"].shape == (batch_size,)
    assert reconstructed["tone"].shape == (batch_size, 1, 8, 1)
    assert reconstructed["color"].shape == (batch_size, 3, 8, 1)
    assert reconstructed["contrast"].shape == (batch_size,)
    assert reconstructed["sharp"].shape == (batch_size,)
    assert reconstructed["blur"].shape == (batch_size,)


def test_flatten_and_unflatten_roundtrip() -> None:
    batch_size = 2
    params = make_params(batch_size)

    flat = flatten_output_params(params, batch_size)
    reconstructed = unflatten_output_params(flat)

    torch.testing.assert_close(reconstructed["exposure"], params["exposure"])
    torch.testing.assert_close(reconstructed["saturation"], params["saturation"])
    torch.testing.assert_close(reconstructed["tone"], params["tone"])
    torch.testing.assert_close(reconstructed["color"], params["color"])
    torch.testing.assert_close(reconstructed["contrast"], params["contrast"])
    torch.testing.assert_close(reconstructed["sharp"], params["sharp"])
    torch.testing.assert_close(reconstructed["blur"], params["blur"])


def test_unflatten_output_params_rejects_non_2d_tensor() -> None:
    flat = torch.zeros(37)

    with pytest.raises(ValueError, match="must be a 2D tensor"):
        unflatten_output_params(flat)


@pytest.mark.parametrize("num_params", [0, 1, 10, 36, 38])
def test_unflatten_output_params_rejects_wrong_number_of_params(
    num_params: int,
) -> None:
    flat = torch.zeros(2, num_params)

    with pytest.raises(ValueError, match="must contain exactly 37 parameters"):
        unflatten_output_params(flat)