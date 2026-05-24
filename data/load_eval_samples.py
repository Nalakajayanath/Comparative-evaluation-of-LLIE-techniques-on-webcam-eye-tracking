import os
import cv2
import numpy as np
import scipy.io as sio

EVAL_ROOT = "data/original/MPIIGaze/Evaluation Subset/sample list for eye image"
MAT_ROOT = "data/original/MPIIGaze/Data/Normalized"


def _pose_rad(pose_vector, eye):
    rot = cv2.Rodrigues(np.array(pose_vector, dtype=np.float32))[0]
    vec = rot[:, 2]
    pitch = np.arcsin(vec[1])
    yaw = np.arctan2(vec[0], vec[2])
    pose = np.array([pitch, yaw], dtype=np.float32)
    if eye == "right":
        pose[1] *= -1
    return pose


def _eye_image_path(image_rel, eye):
    day_dir, filename = os.path.split(image_rel.replace("\\", "/"))
    stem, ext = os.path.splitext(filename)
    return os.path.normpath(os.path.join(day_dir, f"{eye}_{stem}{ext}"))


def load_evaluation_samples():
    samples = []
    mat_cache = {}

    for subject_file in sorted(os.listdir(EVAL_ROOT)):
        if not subject_file.endswith(".txt"):
            continue

        subject_id = subject_file.replace(".txt", "")
        with open(os.path.join(EVAL_ROOT, subject_file)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                image_rel, eye = line.split()[:2]
                day, frame_name = image_rel.split("/")
                frame_idx = int(os.path.splitext(frame_name)[0]) - 1

                mat_path = os.path.join(MAT_ROOT, subject_id, f"{day}.mat")
                if mat_path not in mat_cache:
                    if not os.path.exists(mat_path):
                        continue
                    mat_cache[mat_path] = sio.loadmat(mat_path)

                try:
                    eye_data = mat_cache[mat_path]["data"][0][0][eye][0][0]
                    gaze = np.asarray(eye_data["gaze"][frame_idx], dtype=np.float64)
                    pose = _pose_rad(eye_data["pose"][frame_idx], eye)
                except (KeyError, IndexError, ValueError):
                    continue

                rel_path = os.path.normpath(
                    os.path.join(subject_id, _eye_image_path(image_rel, eye))
                )
                samples.append({
                    "path": rel_path,
                    "eye": eye,
                    "gaze": gaze,
                    "pose": pose,
                })

    return samples
