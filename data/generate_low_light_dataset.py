import os
import sys
import cv2

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.load_eval_samples import load_evaluation_samples
from data.simulate_low_light import (
    simulate_low_intensity,
    simulate_gamma_dark,
    simulate_directional_dark,
)

DATASET_ROOT = os.path.normpath(
    "data/MPIIGaze_Normalized_Images"
)
# EVAL_ROOT = os.path.normpath(
#     "data/original/MPIIGaze/Evaluation Subset/sample list for eye image"
# )
OUTPUT_ROOT = os.path.normpath(
    "data/low_light_simulated"
)


MODES = {
    "low_i": lambda img: simulate_low_intensity(img, reduction=70),

    "low_g": lambda img: simulate_gamma_dark(img, gamma=0.3),

    "low_left": lambda img: simulate_directional_dark(img, "left"),

    "low_right": lambda img: simulate_directional_dark(img, "right"),

    "low_i_g_left": lambda img: simulate_directional_dark(
        simulate_gamma_dark(
            simulate_low_intensity(img, reduction=70),
            gamma=0.3
        ),
        "left"
    ),

    "low_i_g_right": lambda img: simulate_directional_dark(
        simulate_gamma_dark(
            simulate_low_intensity(img, reduction=70),
            gamma=0.3
        ),
        "right"
    ),
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    samples = load_evaluation_samples()
    print(f"Total samples to process: {len(samples)}")

    for i, sample in enumerate(samples):
        src_path = os.path.join(DATASET_ROOT, sample["path"])
        image = cv2.imread(src_path)

        if image is None:
            print(f"[WARN] Could not load {src_path}")
            continue

        for mode, transform in MODES.items():
            out_path = os.path.join(OUTPUT_ROOT, mode, sample["path"])
            ensure_dir(os.path.dirname(out_path))

            dark_image = transform(image)
            cv2.imwrite(out_path, dark_image)

        if i % 500 == 0:
            print(f"Processed {i}/{len(samples)} samples")

    print("Low-light dataset generation completed.")


if __name__ == "__main__":
    main()
