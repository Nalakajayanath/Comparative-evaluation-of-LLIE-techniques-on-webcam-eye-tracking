import numpy as np


def angular_error(gaze_gt, yaw_pred, pitch_pred):
    """Degrees between GT gaze vector (.mat) and predicted yaw/pitch (MPIIGaze convention)."""
    v_gt = np.asarray(gaze_gt, dtype=np.float64).flatten()
    v_gt /= np.linalg.norm(v_gt)

    pitch = np.deg2rad(pitch_pred)
    yaw = np.deg2rad(yaw_pred)
    v_pred = np.array([
        -np.cos(pitch) * np.sin(yaw),
        -np.sin(pitch),
        -np.cos(pitch) * np.cos(yaw),
    ])
    v_pred /= np.linalg.norm(v_pred)

    dot = np.clip(np.dot(v_gt, v_pred), -1.0, 1.0)
    return np.rad2deg(np.arccos(dot))
