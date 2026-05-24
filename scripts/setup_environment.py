#!/usr/bin/env python
"""Install pip deps, clone LLIE vendor repos, download gaze weights."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR_DIR = os.path.join(PROJECT_ROOT, "vendor")
NOTICES_PATH = os.path.join(PROJECT_ROOT, "THIRD_PARTY_NOTICES.md")
REQUIREMENTS = os.path.join(PROJECT_ROOT, "requirements.txt")

VENDOR_REPOS = [
    {
        "dir": "Zero-DCE",
        "url": "https://github.com/Li-Chongyi/Zero-DCE.git",
        "license": "Apache License 2.0",
        "purpose": "Zero-DCE low-light enhancement.",
    },
    {
        "dir": "EnlightenGAN-inference",
        "url": "https://github.com/arsenyinfo/EnlightenGAN-inference.git",
        "license": "See repository LICENSE",
        "purpose": "EnlightenGAN inference.",
    },
    {
        "dir": "MIRNet",
        "url": "https://github.com/swz30/MIRNet.git",
        "license": "Apache License 2.0",
        "purpose": "MIRNet low-light enhancement.",
    },
]

PRETRAINED_GAZE = {
    "name": "mpiigaze_resnet_preact.pth",
    "url": "https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.1.0/mpiigaze_resnet_preact.pth",
    "license": "MIT (hysts/pytorch_mpiigaze_demo)",
    "purpose": "MPIIGaze ResNet-preact weights (gaze_model/mpiigaze_resnet.py).",
}


def run(cmd, cwd=None):
    print(f"\n>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd or PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pip", action="store_true")
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--skip-weights", action="store_true")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    if not args.skip_pip:
        run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS])

    if not args.skip_clone:
        os.makedirs(VENDOR_DIR, exist_ok=True)
        for repo in VENDOR_REPOS:
            dest = os.path.join(VENDOR_DIR, repo["dir"])
            if os.path.isdir(os.path.join(dest, ".git")):
                print(f"[skip] {repo['dir']}")
                continue
            run(["git", "clone", "--depth", "1", repo["url"], dest])

    lines = [
        "# Third-party notices",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
        "## vendor/",
        "",
        "| Folder | Purpose | License |",
        "|--------|---------|---------|",
    ]
    for repo in VENDOR_REPOS:
        lines.append(f"| `{repo['dir']}` | {repo['purpose']} | {repo['license']} |")
    lines.extend([
        "",
        "## Gaze weights",
        "",
        f"| {PRETRAINED_GAZE['name']} | {PRETRAINED_GAZE['purpose']} | {PRETRAINED_GAZE['license']} |",
        "",
    ])
    with open(NOTICES_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    if not args.skip_weights:
        run([sys.executable, os.path.join("scripts", "download_pretrained_gaze_weights.py")])

    print(
        "\nDone. Next:\n"
        "  python data/extract_normalized_images.py\n"
        "  python data/simulate_low_light.py\n"
        "  python evaluation/run_all.py\n"
    )


if __name__ == "__main__":
    main()
