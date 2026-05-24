import os
import time
import numpy as np
import cv2

from data.load_eval_samples import load_evaluation_samples
from gaze_model.mpiigaze_predictor import MPIIGazePredictor
from evaluation.angular_error import angular_error_vectors
from LLIEs.llie_factory import apply_llie

DATA_ROOT = "data"


def load_image(root, rel_path):
    return cv2.imread(os.path.join(root, rel_path))


def extract_subject(sample):
    parts = os.path.normpath(sample["path"]).split(os.sep)
    return parts[-3]


def run_single_eval(root, method, params, max_samples):

    samples = load_evaluation_samples()

    errors = []
    times = []

    current_subject = None
    predictor = None

    for i, sample in enumerate(samples):

        if max_samples is not None and i >= max_samples:
            break

        image = load_image(root, sample["path"])
        if image is None:
            continue

        subject = extract_subject(sample)

        if subject != current_subject:
            predictor = MPIIGazePredictor(subject)
            current_subject = subject

        start = time.perf_counter()

        enhanced = apply_llie(image, method, params)

        yaw_pred, pitch_pred = predictor.predict(
            enhanced,
            sample["pose"],
            eye=sample["eye"],
        )

        end = time.perf_counter()

        err = angular_error_vectors(sample["gaze"], yaw_pred, pitch_pred)

        errors.append(err)
        times.append((end - start) * 1000)

    if len(errors) == 0:
        return None

    mean_err = np.mean(errors)
    std_err = np.std(errors)
    avg_time = np.mean(times)
    fps = 1000 / avg_time if avg_time > 0 else 0

    return mean_err, std_err, avg_time, fps
