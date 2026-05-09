import os
import scipy.io as sio
import numpy as np

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
                    gaze_3d = mat['data'][0][0][eye][0][0]['gaze'][frame_idx]
                    x, y, z = gaze_3d

                    pitch = np.arcsin(-y)
                    yaw = np.arctan2(-x, -z)

                    yaw = np.rad2deg(yaw)
                    pitch = np.rad2deg(pitch)

                except:
                    continue
                
                frame_file = image_rel.split("/")[1]
                new_filename = f"{eye}_{frame_file}"

                full_rel_path = os.path.normpath(
                    os.path.join(subject_id, day, new_filename)
                )

                samples.append({
                    "path": full_rel_path,
                    "eye": eye,
                    "yaw": yaw,
                    "pitch": pitch
                })

    return samples