import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

from social_media_overlay.ext.gebhardt.dataset.CocoCaptions import CocoCaptions


def _make_fake_coco_root(tmp_path, split="train"):
    root = tmp_path / "coco"
    ann_dir = root / "annotations"
    img_dir = root / f"{split}2017"

    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    annotations = {
        "annotations": [
            {"image_id": 1, "caption": "a cat on a mat"},
            {"image_id": 1, "caption": "another cat caption"},
            {"image_id": 2, "caption": "a dog in the park"},
        ]
    }

    ann_file = ann_dir / f"captions_{split}2017.json"
    ann_file.write_text(json.dumps(annotations))

    img1 = Image.new("RGB", (10, 10), color=(255, 0, 0))
    img1.save(img_dir / "000000000001.jpg")

    img2 = Image.new("L", (8, 8), color=128)  # grayscale on purpose
    img2.save(img_dir / "000000000002.jpg")

    return root


def test_init_and_len(tmp_path):
    root = _make_fake_coco_root(tmp_path)

    dataset = CocoCaptions(str(root), split="train")

    assert len(dataset) == 2


def test_getitem_returns_tensor_and_metadata(tmp_path):
    root = _make_fake_coco_root(tmp_path)

    dataset = CocoCaptions(str(root), split="train")
    image, metadata = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.ndim == 3
    assert image.shape[0] == 3  # channels first, RGB

    assert isinstance(metadata, list)
    assert len(metadata) == 3
    assert metadata[0] == "000000000001.jpg"
    assert metadata[1].endswith("train2017/000000000001.jpg")
    assert metadata[2] == "a cat on a mat/another cat caption"


def test_getitem_converts_grayscale_to_rgb(tmp_path):
    root = _make_fake_coco_root(tmp_path)

    dataset = CocoCaptions(str(root), split="train")
    image, metadata = dataset[1]

    assert isinstance(image, torch.Tensor)
    assert image.shape[0] == 3  # grayscale image should be converted to RGB
    assert metadata[0] == "000000000002.jpg"


def test_custom_transform_is_used(tmp_path):
    root = _make_fake_coco_root(tmp_path)
    transform = transforms.Compose(
        [
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
        ]
    )

    dataset = CocoCaptions(str(root), split="train", transform=transform)
    image, _ = dataset[0]

    assert image.shape == (3, 16, 16)


def test_invalid_split_raises_assertion():
    with pytest.raises(AssertionError, match="Invalid split"):
        CocoCaptions("/some/path", split="wrong")


def test_missing_root_raises_file_not_found(tmp_path):
    missing_root = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError, match="Dataset root does not exist"):
        CocoCaptions(str(missing_root), split="train")


def test_missing_annotation_file_raises_file_not_found(tmp_path):
    root = tmp_path / "coco"
    root.mkdir()
    (root / "train2017").mkdir()

    with pytest.raises(FileNotFoundError, match="Annotation file missing"):
        CocoCaptions(str(root), split="train")


def test_missing_image_directory_raises_file_not_found(tmp_path):
    root = tmp_path / "coco"
    ann_dir = root / "annotations"
    ann_dir.mkdir(parents=True)

    ann_file = ann_dir / "captions_train2017.json"
    ann_file.write_text(json.dumps({"annotations": []}))

    with pytest.raises(FileNotFoundError, match="Image directory missing"):
        CocoCaptions(str(root), split="train")
