"""Download a subset of the UIEB dataset (paired raw/reference underwater images)
from Hugging Face. UIEB: 890 real underwater images with expert-picked references.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "Edddddd8787/UIEB"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main(n_pairs: int):
    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")
    raw = sorted(f for f in files if f.startswith("raw-890/"))
    ref = set(f for f in files if f.startswith("reference-890/"))

    # keep only images that exist in both folders, take first n_pairs
    pairs = [f for f in raw if "reference-890/" + f.split("/", 1)[1] in ref][:n_pairs]
    print(f"Downloading {len(pairs)} image pairs from {REPO} ...")

    for i, rawfile in enumerate(pairs, 1):
        name = rawfile.split("/", 1)[1]
        for sub in (rawfile, "reference-890/" + name):
            hf_hub_download(REPO, sub, repo_type="dataset",
                            local_dir=DATA_DIR, local_dir_use_symlinks=False)
        if i % 25 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)} pairs done")

    print(f"Done. Data in {DATA_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=int, default=250, help="number of image pairs to download")
    main(p.parse_args().pairs)
