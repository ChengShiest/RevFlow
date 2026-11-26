from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def center_crop(image, size=256):
    while min(image.size) >= size * 2:
        image = image.resize(tuple(dim // 2 for dim in image.size), Image.Resampling.BOX)
    scale = size / min(image.size)
    image = image.resize(tuple(round(dim * scale) for dim in image.size), Image.Resampling.BICUBIC)
    left, top = (image.width - size) // 2, (image.height - size) // 2
    return image.crop((left, top, left + size, top + size))


class ImageNet(Dataset):
    """ImageFolder layout, with a stable alphabetical class-to-index mapping."""
    def __init__(self, root, augment=False):
        self.samples = []
        self.augment = augment
        self.epoch_seed = 0
        self.classes = sorted(path for path in Path(root).iterdir() if path.is_dir() and not path.name.startswith("."))
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".ppm", ".pgm"}
        for label, folder in enumerate(self.classes):
            paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
            self.samples.extend((p, label) for p in paths)
        if not self.samples:
            raise ValueError(f"Expected class subdirectories containing images in {root}")
        self.mean = torch.tensor([.485, .456, .406])[:, None, None]
        self.std = torch.tensor([.229, .224, .225])[:, None, None]

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, seed, epoch):
        self.epoch_seed = seed + epoch * len(self)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = center_crop(image.convert("RGB"))
        if self.augment:
            rng = torch.Generator().manual_seed(self.epoch_seed + index)
            if torch.rand((), generator=rng) < .5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
        return (pixels - self.mean) / self.std, label
