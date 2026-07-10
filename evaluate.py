"""Enhance a single underwater image with the trained model.
Usage: python enhance.py input.jpg [output.png]
"""
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import UNetEnhancer

ROOT = Path(__file__).resolve().parent.parent


def main():
    inp = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else inp.with_name(inp.stem + "_enhanced.png")

    model = UNetEnhancer(base=32)
    model.load_state_dict(torch.load(ROOT / "checkpoints" / "best.pt", map_location="cpu"))
    model.eval()

    img = Image.open(inp).convert("RGB")
    # round size down to a multiple of 8 (three pooling levels)
    w, h = (d - d % 8 for d in img.size)
    img_r = img.resize((w, h), Image.BICUBIC)
    x = transforms.ToTensor()(img_r).unsqueeze(0)
    with torch.no_grad():
        y = model(x)[0].clamp(0, 1)
    out = transforms.ToPILImage()(y).resize(img.size, Image.BICUBIC)
    out.save(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
