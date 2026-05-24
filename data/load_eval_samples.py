import os
import sys
import scipy.io as sio
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.sample_paths import build_eye_image_rel_path
from data.gaze_pose import pose_for_eye

EVAL_ROOT = os.path.normpath(
    "data/original/MPIIGaze/Evaluation Subset/sample list for eye image"
)

MAT_ROOT = os.path.normpath(
    "data/original/MPIIGaze/Data/Normalized"
)


def load_evaluation_samples():

    samples = []
    mat_cache = {}

    for subject_file in sorted(os.listdir(EVAL_ROOT)):
        if not subject_file.endswith(".txt"):
            continue

        subject_id = subject_file.replace(".txt", "")
        txt_path = os.path.join(EVAL_ROOT, subject_file)

        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                image_rel = parts[0]
                eye = parts[1]

                day = image_rel.split("/")[0]
                frame_name = image_rel.split("/")[1]
                frame_idx = int(frame_name.split(".")[0]) - 1

                mat_path = os.path.join(MAT_ROOT, subject_id, f"{day}.mat")

                if mat_path not in mat_cache:
                    if not os.path.exists(mat_path):
                        continue
                    mat_cache[mat_path] = sio.loadmat(mat_path)

                mat = mat_cache[mat_path]

                try:
                    eye_data = mat["data"][0][0][eye][0][0]
                    gaze_3d = eye_data["gaze"][frame_idx]
                    pose_raw = eye_data["pose"][frame_idx]
                    x, y, z = gaze_3d
                    gaze_vector = np.array([x, y, z], dtype=np.float64)
                    pose_rad = pose_for_eye(pose_raw, eye)

                    pitch = np.rad2deg(np.arcsin(-y))
                    yaw = np.rad2deg(np.arctan2(-x, -z))

                except (KeyError, IndexError, ValueError):
                    continue

                eye_image_rel = build_eye_image_rel_path(image_rel, eye)
                full_rel_path = os.path.normpath(
                    os.path.join(subject_id, eye_image_rel)
                )

                samples.append({
                    "path": full_rel_path,
                    "eye": eye,
                    "yaw": yaw,
                    "pitch": pitch,
                    "gaze": gaze_vector,
                    "pose": pose_rad,
                })

    return samples
