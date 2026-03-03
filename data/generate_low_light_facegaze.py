import os
import cv2
from tqdm import tqdm

from load_facegaze_samples import load_facegaze_samples
from simulate_low_light import (
    simulate_low_intensity,
    simulate_gamma_dark,
    simulate_directional_dark
)


FACEGAZE_ROOT = "data/original/MPIIFaceGaze"
OUTPUT_ROOT = "data/MPIIFaceGaze_LowLight_Simulated"


MODES = {
    "low_intensity": lambda img: simulate_low_intensity(img, reduction=70),

    "gamma_dark": lambda img: simulate_gamma_dark(img, gamma=0.3),

    "directional_left": lambda img: simulate_directional_dark(img, "left"),

    "directional_right": lambda img: simulate_directional_dark(img, "right"),

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

    samples = load_facegaze_samples()

    print(f"Total samples found: {len(samples)}")

    for mode_name, transform in MODES.items():

        print("\n===================================")
        print(f"Generating mode: {mode_name}")
        print("===================================")

        current_subject = None
        subject_count = 0

        for i, sample in enumerate(samples):

            subject = sample["subject"]

            if subject != current_subject:
                current_subject = subject
                subject_count += 1
                print(f"\nProcessing subject: {subject}")

            src_path = os.path.join(FACEGAZE_ROOT, sample["path"])
            image = cv2.imread(src_path)

            if image is None:
                print(f"[WARNING] Could not load: {src_path}")
                continue

            dark_image = transform(image)

            out_path = os.path.join(
                OUTPUT_ROOT,
                mode_name,
                sample["path"]
            )

            ensure_dir(os.path.dirname(out_path))
            cv2.imwrite(out_path, dark_image)

            # Print every 500 images
            if i % 500 == 0:
                print(f"[{mode_name}] Processed {i}/{len(samples)} images")

        print(f"\nFinished mode: {mode_name}")


if __name__ == "__main__":
    main()