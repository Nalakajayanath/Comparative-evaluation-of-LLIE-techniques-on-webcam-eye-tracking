import numpy as np


def angular_error_vectors(v_gt, yaw_pred, pitch_pred):
    """Angular error (degrees) between a 3D GT gaze vector and predicted yaw/pitch."""
    v_gt = np.asarray(v_gt, dtype=np.float64).flatten()
    v_pred = gaze_to_vector(yaw_pred, pitch_pred)

    v_gt /= np.linalg.norm(v_gt)
    v_pred /= np.linalg.norm(v_pred)

    dot = np.clip(np.dot(v_gt, v_pred), -1.0, 1.0)
    return np.rad2deg(np.arccos(dot))


def angular_error(yaw_gt, pitch_gt, yaw_pred, pitch_pred):
    v_gt = gaze_to_vector(yaw_gt, pitch_gt)
    v_pred = gaze_to_vector(yaw_pred, pitch_pred)

    dot = np.dot(v_gt, v_pred)
    dot = np.clip(dot, -1.0, 1.0)

    angle = np.arccos(dot)
    return np.rad2deg(angle)


def gaze_to_vector(yaw, pitch):
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)

    x = np.cos(pitch) * np.sin(yaw)
    y = np.sin(pitch)
    z = np.cos(pitch) * np.cos(yaw)

    return np.array([x, y, z])
