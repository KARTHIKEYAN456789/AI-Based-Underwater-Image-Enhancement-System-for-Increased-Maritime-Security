"""Evaluate the trained model vs a CLAHE baseline on the validation split.
Reports PSNR/SSIM and saves side-by-side comparison images to results/.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage import color, exposure
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dataset import UIEBPairs
from model import UNetEnhancer

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def clahe_baseline(img: np.ndarray) -> np.ndarray:
    """Classical baseline: CLAHE on the L channel in LAB space."""
    lab = color.rgb2lab(img)
    L = lab[..., 0] / 100.0
    lab[..., 0] = exposure.equalize_adapthist(L, clip_limit=0.02) * 100.0
    return np.clip(color.lab2rgb(lab), 0, 1)


def to_np(t: torch.Tensor) -> np.ndarray:
    return t.permute(1, 2, 0).cpu().numpy()


def main():
    RESULTS.mkdir(exist_ok=True)
    ds = UIEBPairs("val", size=256)
    model = UNetEnhancer(base=32)
    model.load_state_dict(torch.load(ROOT / "checkpoints" / "best.pt", map_location="cpu"))
    model.eval()

    stats = {"raw": [], "clahe": [], "unet": []}
    for i in range(len(ds)):
        raw_t, ref_t = ds[i]
        raw, ref = to_np(raw_t), to_np(ref_t)
        with torch.no_grad():
            out = to_np(model(raw_t.unsqueeze(0))[0]).clip(0, 1)
        cl = clahe_baseline(raw)

        for key, img in (("raw", raw), ("clahe", cl), ("unet", out)):
            stats[key].append((
                peak_signal_noise_ratio(ref, img, data_range=1.0),
                structural_similarity(ref, img, channel_axis=2, data_range=1.0),
            ))

        if i < 8:  # save first few visual comparisons
            panel = (np.concatenate([raw, cl, out, ref], axis=1) * 255).astype(np.uint8)
            Image.fromarray(panel).save(RESULTS / f"compare_{i}.png")

    print(f"{'method':8s} {'PSNR (dB)':>10s} {'SSIM':>8s}")
    for key in ("raw", "clahe", "unet"):
        arr = np.array(stats[key])
        print(f"{key:8s} {arr[:,0].mean():10.2f} {arr[:,1].mean():8.3f}")
    print(f"\nComparison panels (raw | CLAHE | U-Net | reference) in {RESULTS}")


if __name__ == "__main__":
    main()
