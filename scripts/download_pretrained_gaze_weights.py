#!/usr/bin/env python
"""
Download official MPIIGaze ResNet-preact weights (ptgaze release) and install
into models/mpiigaze/resnet_preact/fold00 … fold14.

Source (MIT, hysts):
  https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.1.0/mpiigaze_resnet_preact.pth

Note: This is a single pretrained model for demo/inference, NOT 15 separate
leave-one-person-out folds. It is much better than training on CPU for days,
but strict MPIIGaze evaluation normally trains one model per held-out person.
For an MSc comparing LLIE methods with a fixed gaze backbone, this is acceptable
if you state it in the methods section.

Run from project root:
  python scripts/download_pretrained_gaze_weights.py
"""

import os
import shutil
import sys
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT_URL = (
    "https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/"
    "v0.1.0/mpiigaze_resnet_preact.pth"
)
CACHE_PATH = os.path.join(PROJECT_ROOT, "models", "mpiigaze", "mpiigaze_resnet_preact.pth")
OUT_ROOT = os.path.join(PROJECT_ROOT, "models", "mpiigaze", "resnet_preact")
CHECKPOINT_NAME = "checkpoint_0040.pth"


def download(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(dest):
        print(f"[skip] already downloaded: {dest}")
        return

    print(f"Downloading {url}")
    print(f"  -> {dest}")

    def progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 / total_size)
            print(f"\r  {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()


def main():
    try:
        import torch
    except ImportError:
        print("Install torch first: pip install torch")
        sys.exit(1)

    download(WEIGHT_URL, CACHE_PATH)

    ckpt = torch.load(CACHE_PATH, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        payload = ckpt
    else:
        payload = {"model": ckpt}
        print("Wrapped raw state_dict as checkpoint['model']")

    for fold in range(15):
        fold_dir = os.path.join(OUT_ROOT, f"fold{fold:02d}")
        os.makedirs(fold_dir, exist_ok=True)
        out_path = os.path.join(fold_dir, CHECKPOINT_NAME)
        torch.save(payload, out_path)
        print(f"Installed {out_path}")

    print(
        "\nDone. Next run:\n"
        "  python evaluation/run_all.py\n"
    )


if __name__ == "__main__":
    main()
