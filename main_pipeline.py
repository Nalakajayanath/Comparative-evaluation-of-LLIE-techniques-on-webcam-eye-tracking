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
from gaze_model.l2cs_blackbox import L2CSBlackBox
from evaluation.angular_error import angular_error

# Then import LLIEs modules
from LLIEs.llie_enum import LLIEMethod
from LLIEs.llie_factory import apply_llie
from LLIEs.classical.clahe import clahe_enhancement

NORMALIZED_ROOT = "data/original/MPIIGaze/Data/Normalized"
LOW_LIGHT_ROOT = "data/low_light_simulated/low_i_g_left" # Change this to test different low-light versions

def load_image(root, rel_path):
    path = os.path.join(root, rel_path)
    image = cv2.imread(path)
    return image

def load_gt_from_mat(subject, day, frame_index, eye):
    mat_path = os.path.join(
        NORMALIZED_ROOT,
        subject,
        day + ".mat"
    )
    mat = sio.loadmat(mat_path)
    
    # Extract the 3D gaze vector [x, y, z] for the specific frame
    gaze_3d = mat['data'][0][0][eye][0][0]['gaze'][frame_index]
    
    x, y, z = gaze_3d[0], gaze_3d[1], gaze_3d[2]
    
    # Mathematical conversion from 3D vector to 2D angles
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    
    return np.rad2deg(yaw), np.rad2deg(pitch)

def extract_frame_index(filename):
    return int(os.path.splitext(filename)[0]) - 1

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

        image = load_image(root, sample["path"])
        if image is None:
            continue

        parts = os.path.normpath(sample["path"]).split(os.sep)
        subject = parts[-3]

        if subject != current_subject:
            predictor = L2CSBlackBox(subject)
            current_subject = subject

        day = parts[-2]
        filename = parts[-1]
        frame_index = extract_frame_index(filename)

        yaw_gt, pitch_gt = load_gt_from_mat(subject, day, frame_index, sample["eye"])

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
    evaluate_dataset(LOW_LIGHT_ROOT, LLIEMethod.NONE, max_samples=100)
    evaluate_dataset(LOW_LIGHT_ROOT, LLIEMethod.HE, max_samples=100)
    # evaluate_dataset(LOW_LIGHT_ROOT, LLIEMethod.CLAHE, max_samples=100)
    # evaluate_dataset(LOW_LIGHT_ROOT, LLIEMethod.SSR, max_samples=100)
    # evaluate_dataset(LOW_LIGHT_ROOT, LLIEMethod.MSR, max_samples=100)
    # evaluate_dataset(LOW_LIGHT_ROOT, LLIEMethod.ZERODCE, max_samples=100)
    #evaluate_dataset(LOW_LIGHT_ROOT, LLIEMethod.ENLIGHTENGAN, max_samples=100)
    evaluate_dataset(LOW_LIGHT_ROOT, LLIEMethod.MIRNET, max_samples=100)