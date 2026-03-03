import sys
import os
import time
import cv2
import numpy as np

from data.load_facegaze_samples import load_facegaze_samples

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from data.load_facegaze_gt import load_facegaze_gt
from gaze_model.l2cs_blackbox import L2CSBlackBox
from evaluation.angular_error import angular_error

from LLIEs.llie_enum import LLIEMethod
from LLIEs.llie_factory import apply_llie


LOW_LIGHT_ROOT = "data/MPIIFaceGaze_LowLight_Simulated/gamma_dark"


def load_image(rel_path):
    path = os.path.join(LOW_LIGHT_ROOT, rel_path)
    return cv2.imread(path)


def evaluate_dataset(llie_method=LLIEMethod.NONE, max_samples=None):

    samples = load_facegaze_samples()

    errors = []
    enhancement_times = []
    total_times = []

    predictor = None

    for i, sample in enumerate(samples):

        if max_samples is not None and i >= max_samples:
            break

        image = load_image(sample["path"])
        if image is None:
            continue

        if predictor is None:
            predictor = L2CSBlackBox()

        yaw_gt, pitch_gt = load_facegaze_gt(sample["annotation"])

        # -------- TIMING START --------
        start_total = time.perf_counter()

        start_enhance = time.perf_counter()
        enhanced = apply_llie(image, llie_method)
        end_enhance = time.perf_counter()

        yaw_pred, pitch_pred = predictor.predict(enhanced)

        end_total = time.perf_counter()
        # -------- TIMING END --------

        err = angular_error(yaw_gt, pitch_gt, yaw_pred, pitch_pred)
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

    evaluate_dataset(LLIEMethod.NONE, max_samples=100)
    #evaluate_dataset(LLIEMethod.HE, max_samples=100)
    #evaluate_dataset(LLIEMethod.CLAHE, max_samples=100)
    #evaluate_dataset(LLIEMethod.SSR, max_samples=100)
    #evaluate_dataset(LLIEMethod.MSR, max_samples=100)
    #evaluate_dataset(LLIEMethod.MSRCR, max_samples=100)
    #evaluate_dataset(LLIEMethod.ZERODCE, max_samples=100)
    #evaluate_dataset(LLIEMethod.ENLIGHTENGAN, max_samples=100)
    #evaluate_dataset(LLIEMethod.MIRNET, max_samples=100)