import json
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


# https://arxiv.org/pdf/2501.12289
class CocoCaptions(Dataset):

    def __init__(self, root, split, transform=None, use_annotations=True):
        """
        Constructor

        :param root: Path to COCO root folder (e.g. ".../coco")
        :param split: Dataset split, must be "train", "val", or "test"
        :param transform: torchvision transform pipeline (default: ToTensor())
        :param use_annotations: If True, load captions when available.
                                For test split, this is ignored because COCO test has no captions.
        """
        assert split in ["train", "val", "test"], f"Invalid split: {split}"

        self.split = split
        self.root_dir = os.path.join(root, f"{split}2017")
        self.transform = transforms.ToTensor() if transform is None else transform
        self.use_annotations = use_annotations and split in ["train", "val"]

        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset root does not exist: {root}")

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Image directory missing: {self.root_dir}")

        # Select correct annotation / metadata file
        if self.split in ["train", "val"] and self.use_annotations:
            ann_file = os.path.join(root, "annotations", f"captions_{split}2017.json")
            if not os.path.exists(ann_file):
                raise FileNotFoundError(f"Annotation file missing: {ann_file}")
            self.image_list = self.get_images_with_captions(ann_file)

        else:
            # test2017 has no caption annotations
            # COCO usually provides image_info_test2017.json
            info_file = os.path.join(root, "annotations", f"image_info_{split}2017.json")

            if os.path.exists(info_file):
                self.image_list = self.get_images_without_captions(info_file)
            else:
                # Fallback: scan image directory directly
                self.image_list = self.get_images_from_folder(self.root_dir)

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_list)

    def __getitem__(self, index):
        """
        Loads and returns one sample from the dataset.

        :param index: Index of the image
        :return: (image_tensor, metadata)
        """
        image_id = self.image_list[index]["id"]
        image_name = f"{str(image_id).zfill(12)}.jpg"
        image_path = os.path.join(self.root_dir, image_name)

        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        captions = self.image_list[index].get("captions", [])
        joined_captions = "/".join(captions).replace("\n", "") if captions else ""

        return image, [
            image_name,
            image_path,
            joined_captions
        ]

    @staticmethod
    def get_images_with_captions(ann_file):
        """
        Reads a COCO captions annotation file and groups captions by image_id.

        :param ann_file: Path to COCO captions JSON file
        :return: List of dicts:
                 {"id": image_id, "captions": [caption1, caption2, ...]}
        """
        with open(ann_file, "r") as f:
            data = json.load(f)["annotations"]

        captions = {}
        for item in data:
            image_id = item["image_id"]
            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(item["caption"])

        images = []
        for image_id, caps in captions.items():
            images.append({
                "id": image_id,
                "captions": caps
            })

        return images

    @staticmethod
    def get_images_without_captions(info_file):
        """
        Reads a COCO image info JSON file for splits without captions, e.g. test2017.

        :param info_file: Path to COCO image info JSON file
        :return: List of dicts:
                 {"id": image_id, "captions": []}
        """
        with open(info_file, "r") as f:
            data = json.load(f)["images"]

        images = []
        for item in data:
            images.append({
                "id": item["id"],
                "captions": []
            })

        return images

    @staticmethod
    def get_images_from_folder(root_dir):
        """
        Fallback if no annotation/info JSON is available.
        Scans the image directory and extracts ids from filenames.

        :param root_dir: Directory containing COCO images
        :return: List of dicts:
                 {"id": image_id, "captions": []}
        """
        images = []

        for file_name in sorted(os.listdir(root_dir)):
            if not file_name.lower().endswith(".jpg"):
                continue

            stem = os.path.splitext(file_name)[0]

            try:
                image_id = int(stem)
            except ValueError:
                continue

            images.append({
                "id": image_id,
                "captions": []
            })

        return images