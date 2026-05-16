import torch
from torch.utils.data import DataLoader, TensorDataset

from social_media_overlay.utils import get_first_n_images, to_tensor, zero_scalar

def test_to_tensor_with_tensor_changes_device_and_dtype():
    x = torch.ones(2, 2, dtype=torch.float32)

    result = to_tensor(x, device="cpu", dtype=torch.float64)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float64
    assert result.device.type == "cpu"
    assert torch.allclose(result, x.to(torch.float64))


def test_to_tensor_with_list_creates_tensor():
    x = [1, 2, 3]

    result = to_tensor(x, device="cpu", dtype=torch.float32)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))


def test_to_tensor_with_numpy_array():
    import numpy as np

    x = np.array([1, 2, 3])

    result = to_tensor(x, device="cpu", dtype=torch.float32)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))


def test_zero_scalar_returns_scalar_tensor():
    result = zero_scalar(device="cpu", dtype=torch.float32)

    assert isinstance(result, torch.Tensor)
    assert result.shape == ()  # scalar tensor
    assert result.item() == 0.0
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"



def test_get_first_n_images_returns_first_n_from_tensor_dataloader():
    images = torch.arange(10 * 3 * 4 * 4, dtype=torch.float32).reshape(10, 3, 4, 4)
    dataloader = DataLoader(images, batch_size=4)

    result = get_first_n_images(dataloader, n=5)

    assert result.shape == (5, 3, 4, 4)
    assert torch.equal(result, images[:5])


def test_get_first_n_images_supports_tuple_batches():
    images = torch.arange(6 * 3 * 2 * 2, dtype=torch.float32).reshape(6, 3, 2, 2)
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=2)

    result = get_first_n_images(dataloader, n=4)

    assert result.shape == (4, 3, 2, 2)
    assert torch.equal(result, images[:4])


def test_get_first_n_images_returns_all_if_less_than_n_available():
    images = torch.arange(3 * 1 * 2 * 2, dtype=torch.float32).reshape(3, 1, 2, 2)
    dataloader = DataLoader(images, batch_size=2)

    result = get_first_n_images(dataloader, n=8)

    assert result.shape == (3, 1, 2, 2)
    assert torch.equal(result, images)


def test_get_first_n_images_raises_for_empty_dataloader():
    images = torch.empty((0, 3, 4, 4))
    dataloader = DataLoader(images, batch_size=2)

    try:
        get_first_n_images(dataloader, n=4)
        assert False, "Expected RuntimeError to be raised"
    except RuntimeError as e:
        assert str(e) == "No images found in dataloader."