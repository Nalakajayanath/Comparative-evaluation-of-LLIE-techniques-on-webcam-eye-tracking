import numpy as np


def load_facegaze_gt(annotation_parts):

    # Columns:
    # 22–24 (index 21–23) = fc
    # 25–27 (index 24–26) = gt

    fc = np.array([
        float(annotation_parts[21]),
        float(annotation_parts[22]),
        float(annotation_parts[23])
    ])

    gt = np.array([
        float(annotation_parts[24]),
        float(annotation_parts[25]),
        float(annotation_parts[26])
    ])

    gaze_vector = gt - fc

    x, y, z = gaze_vector

    norm = np.linalg.norm(gaze_vector)

    # Robust yaw / pitch conversion
    yaw = np.arctan2(x, z)
    pitch = np.arcsin(-y / norm)

    return np.rad2deg(yaw), np.rad2deg(pitch)