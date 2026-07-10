"""Shared inference helpers: load the enhancer once, enhance a PIL image."""
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import UNetEnhancer

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "best.pt"

_to_tensor = transforms.ToTensor()
_to_pil = transforms.ToPILImage()


@lru_cache(maxsize=1)
def load_model():
    model = UNetEnhancer(base=32)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    return model


def enhance_image(img: Image.Image) -> Image.Image:
    """Enhance a PIL RGB image and return a PIL RGB image at the original size."""
    model = load_model()
    img = img.convert("RGB")
    orig_size = img.size
    # network needs dimensions divisible by 8 (three 2x pooling levels)
    w, h = (max(8, d - d % 8) for d in orig_size)
    x = _to_tensor(img.resize((w, h), Image.BICUBIC)).unsqueeze(0)
    with torch.no_grad():
        y = model(x)[0].clamp(0, 1)
    return _to_pil(y).resize(orig_size, Image.BICUBIC)
