"""Paired underwater image dataset (raw degraded -> clean reference)."""
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class UIEBPairs(Dataset):
    def __init__(self, split="train", size=128, val_fraction=0.1, seed=42):
        raw_dir = DATA_DIR / "raw-890"
        ref_dir = DATA_DIR / "reference-890"
        names = sorted(p.name for p in raw_dir.glob("*.png") if (ref_dir / p.name).exists())
        rng = random.Random(seed)
        rng.shuffle(names)
        n_val = max(1, int(len(names) * val_fraction))
        self.names = names[:n_val] if split == "val" else names[n_val:]
        self.raw_dir, self.ref_dir = raw_dir, ref_dir
        self.split = split
        self.size = size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        raw = Image.open(self.raw_dir / name).convert("RGB")
        ref = Image.open(self.ref_dir / name).convert("RGB")

        if self.split == "train":
            # random resized crop applied identically to both images
            scale = self.size / min(raw.size)
            new = (max(self.size, round(raw.size[0] * scale)),
                   max(self.size, round(raw.size[1] * scale)))
            raw, ref = raw.resize(new, Image.BICUBIC), ref.resize(new, Image.BICUBIC)
            x = random.randint(0, new[0] - self.size)
            y = random.randint(0, new[1] - self.size)
            box = (x, y, x + self.size, y + self.size)
            raw, ref = raw.crop(box), ref.crop(box)
            if random.random() < 0.5:
                raw = raw.transpose(Image.FLIP_LEFT_RIGHT)
                ref = ref.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            raw = raw.resize((self.size, self.size), Image.BICUBIC)
            ref = ref.resize((self.size, self.size), Image.BICUBIC)

        return self.to_tensor(raw), self.to_tensor(ref)
