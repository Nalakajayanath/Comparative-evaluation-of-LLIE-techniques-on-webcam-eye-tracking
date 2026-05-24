import sys
import os
import time
import cv2
import numpy as np
import scipy.io as sio

# Add current directory to path so we can import local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import local modules first
from data.load_eval_samples import load_evaluation_samples
from data.sample_paths import parse_sample_path
from gaze_model.l2cs_blackbox import L2CSBlackBox
from evaluation.angular_error import angular_error_vectors

# Then import LLIEs modules
from LLIEs.llie_enum import LLIEMethod
from LLIEs.llie_factory import apply_llie

NORMALIZED_ROOT = "data/original/MPIIGaze/Data/Normalized"
LOW_LIGHT_ROOT = "data/low_light_simulated/low_i_g_left"  # Change this to test different low-light versions


def load_gaze_vector_from_mat(subject, day, frame_index, eye):
    mat_path = os.path.join(NORMALIZED_ROOT, subject, day + ".mat")
    mat = sio.loadmat(mat_path)
    gaze_3d = mat["data"][0][0][eye][0][0]["gaze"][frame_index]
    return np.array([gaze_3d[0], gaze_3d[1], gaze_3d[2]], dtype=np.float64)


def load_eye_image_from_mat(subject, day, frame_index, eye):
    mat_path = os.path.join(NORMALIZED_ROOT, subject, day + ".mat")
    mat = sio.loadmat(mat_path)
    img = mat["data"][0][0][eye][0][0]["image"][frame_index]

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def load_eye_image(root, rel_path, subject, day, frame_index, eye):
    path = os.path.join(root, rel_path)
    image = cv2.imread(path)
    if image is not None:
        return image

    is_low_light = "low_light" in root.replace("\\", "/")
    if is_low_light:
        print(
            f"[WARN] Missing low-light image: {path}. "
            "Re-run data/extract_normalized_images.py and data/generate_low_light_dataset.py."
        )
        return None

    # Well-lit eval: fall back to normalized .mat when JPEGs are not extracted yet
    return load_eye_image_from_mat(subject, day, frame_index, eye)


def evaluate_dataset(root, llie_method=LLIEMethod.NONE, max_samples=None):

    samples = load_evaluation_samples()
    errors = []
    enhancement_times = []
    total_times = []

    current_subject = None
    predictor = None

    for i, sample in enumerate(samples):

        if max_samples is not None and i >= max_samples:
            break

        subject, day, path_eye, frame_index = parse_sample_path(sample["path"])
        eye = sample["eye"]
        if path_eye is not None and path_eye != eye:
            print(f"[WARN] Eye mismatch for {sample['path']}: path={path_eye}, list={eye}")
        eye = path_eye or eye

        image = load_eye_image(root, sample["path"], subject, day, frame_index, eye)
        if image is None:
            print(f"[WARN] Could not load image for {sample['path']}")
            continue

        if subject != current_subject:
            predictor = L2CSBlackBox(subject)
            current_subject = subject

        gaze_gt = load_gaze_vector_from_mat(subject, day, frame_index, eye)

        # -------- TIMING START --------
        start_total = time.perf_counter()

        start_enhance = time.perf_counter()
        enhanced = apply_llie(image, llie_method)
        end_enhance = time.perf_counter()

        yaw_pred, pitch_pred = predictor.predict(enhanced)

        end_total = time.perf_counter()
        # -------- TIMING END --------

        err = angular_error_vectors(gaze_gt, yaw_pred, pitch_pred)
        errors.append(err)

        enhancement_times.append((end_enhance - start_enhance) * 1000)
        total_times.append((end_total - start_total) * 1000)

    errors = np.array(errors)
    enhancement_times = np.array(enhancement_times)
    total_times = np.array(total_times)

    print("---- Evaluation:", llie_method.value, "----")
    print("Mean Angular Error:", errors.mean())
    print("Enhancement Time (ms):", enhancement_times.mean())
    print("Total Time (ms):", total_times.mean())
    print("FPS:", 1000 / total_times.mean())

if __name__ == "__main__":
    evaluate_dataset(LOW_LIGHT_ROOT, LLIEMethod.NONE, max_samples=100)
    evaluate_dataset(LOW_LIGHT_ROOT, LLIEMethod.HE, max_samples=100)
    # evaluate_dataset(LOW_LIGHT_ROOT, LLIEMethod.CLAHE, max_samples=100)
    # evaluate_dataset(LOW_LIGHT_ROOT, LLIEMethod.SSR, max_samples=100)
    # evaluate_dataset(LOW_LIGHT_ROOT, LLIEMethod.MSR, max_samples=100)
    evaluate_dataset(LOW_LIGHT_ROOT, LLIEMethod.MSRCR, max_samples=100)
    # evaluate_dataset(LOW_LIGHT_ROOT, LLIEMethod.ZERODCE, max_samples=100)
    # evaluate_dataset(LOW_LIGHT_ROOT, LLIEMethod.ENLIGHTENGAN, max_samples=100)
    # evaluate_dataset(LOW_LIGHT_ROOT, LLIEMethod.MIRNET, max_samples=100)
