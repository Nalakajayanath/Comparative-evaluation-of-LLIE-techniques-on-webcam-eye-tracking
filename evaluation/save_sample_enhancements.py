import os
import sys
import cv2
import matplotlib.pyplot as plt

# ---------------------------------------
# Add parent directory to import path
# ---------------------------------------

current_dir = os.path.dirname(
    os.path.abspath(__file__)
)

parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ---------------------------------------
# Imports
# ---------------------------------------

from LLIEs.llie_enum import LLIEMethod
from LLIEs.llie_factory import apply_llie

# ---------------------------------------
# Input image
# ---------------------------------------

INPUT_IMAGE = (
    "data/low_light_simulated/"
    "low_i_g_left/"
    "p00/day01/left_0030.jpg"
)

# ---------------------------------------
# Output directories
# ---------------------------------------

OUTPUT_DIR = "sample_images"
RESULTS_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------
# Read image
# ---------------------------------------

image = cv2.imread(INPUT_IMAGE)

if image is None:
    raise Exception(
        f"Could not read image: {INPUT_IMAGE}"
    )

# ---------------------------------------
# Save original
# ---------------------------------------

original_path = os.path.join(
    OUTPUT_DIR,
    "original.jpg"
)

cv2.imwrite(original_path, image)

# ---------------------------------------
# Methods to generate
# ---------------------------------------

methods = [

    # Histogram-based
    (LLIEMethod.HE, {}),
    (LLIEMethod.CLAHE, {"clip_limit": 2.0}),

    # Retinex-based
    (LLIEMethod.SSR, {"sigma": 30}),
    (LLIEMethod.MSR, {"sigmas": [15,80,250]}),
    (LLIEMethod.MSRCR, {}),

    # Deep learning-based
    (LLIEMethod.ZERODCE, {}),
    (LLIEMethod.ENLIGHTENGAN, {}),
    (LLIEMethod.MIRNET, {})
]

# ---------------------------------------
# Generate enhancements
# ---------------------------------------

saved_images = [("Original", original_path)]

for method, params in methods:

    print(f"Processing {method.value}")

    enhanced = apply_llie(
        image,
        method,
        params
    )

    save_path = os.path.join(
        OUTPUT_DIR,
        f"{method.value}.jpg"
    )

    cv2.imwrite(save_path, enhanced)

    saved_images.append(
        (method.value, save_path)
    )

print("Enhancement generation complete.")

# ---------------------------------------
# Create comparison figure
# ---------------------------------------

plt.figure(figsize=(14, 8))

for i, (title, path) in enumerate(saved_images):

    img = cv2.imread(path)

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    plt.subplot(3, 3, i + 1)

    plt.imshow(img)

    plt.title(title)

    plt.axis("off")

plt.tight_layout()

# ---------------------------------------
# Save final figure
# ---------------------------------------

figure_path = os.path.join(
    RESULTS_DIR,
    "feature_comparison.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Figure saved to: {figure_path}")