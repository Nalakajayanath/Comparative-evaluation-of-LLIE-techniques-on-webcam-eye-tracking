#!/usr/bin/env python
"""One-time fix: replace np.float with np.float64 in vendor/pytorch_mpiigaze (NumPy 2.x)."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(PROJECT_ROOT, "vendor", "pytorch_mpiigaze")

if not os.path.isdir(VENDOR):
    print(f"Not found: {VENDOR}")
    sys.exit(1)

replacements = [
    ("dtype=np.float)", "dtype=np.float64)"),
    ("dtype=np.float,", "dtype=np.float64,"),
    ("np.float)", "np.float64)"),
]
patched = 0
for root, _, files in os.walk(VENDOR):
    for name in files:
        if not name.endswith(".py"):
            continue
        path = os.path.join(root, name)
        text = open(path, encoding="utf-8").read()
        new = text
        for old, new_val in replacements:
            new = new.replace(old, new_val)
        if new != text:
            open(path, "w", encoding="utf-8").write(new)
            patched += 1
            print("patched:", path)

print(f"Done. {patched} file(s) updated.")
