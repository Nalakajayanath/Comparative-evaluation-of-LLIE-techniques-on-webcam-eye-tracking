#!/usr/bin/env python
"""Download ptgaze MPIIGaze weights into models/mpiigaze/resnet_preact/fold00-14/."""

import os
import sys
import urllib.request

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = (
    "https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/"
    "v0.1.0/mpiigaze_resnet_preact.pth"
)
CACHE = os.path.join(ROOT, "models", "mpiigaze", "mpiigaze_resnet_preact.pth")
OUT_ROOT = os.path.join(ROOT, "models", "mpiigaze", "resnet_preact")


def main():
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    if not os.path.isfile(CACHE):
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, CACHE)

    ckpt = torch.load(CACHE, map_location="cpu")
    payload = ckpt if isinstance(ckpt, dict) and "model" in ckpt else {"model": ckpt}

    for fold in range(15):
        fold_dir = os.path.join(OUT_ROOT, f"fold{fold:02d}")
        os.makedirs(fold_dir, exist_ok=True)
        path = os.path.join(fold_dir, "checkpoint_0040.pth")
        torch.save(payload, path)
        print(path)


if __name__ == "__main__":
    main()
