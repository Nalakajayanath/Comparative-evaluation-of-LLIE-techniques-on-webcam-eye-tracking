import numpy as np


def mpiigaze_angles_to_vector(pitch_deg, yaw_deg):
    """
    Convert pitch/yaw (degrees) to a unit gaze vector using the MPIIGaze / hysts convention.

    Matches gaze_estimation.utils.convert_to_unit_vector in pytorch_mpiigaze.
    """
    pitch = np.deg2rad(float(pitch_deg))
    yaw = np.deg2rad(float(yaw_deg))

    x = -np.cos(pitch) * np.sin(yaw)
    y = -np.sin(pitch)
    z = -np.cos(pitch) * np.cos(yaw)

    v = np.array([x, y, z], dtype=np.float64)
    return v / np.linalg.norm(v)


def angular_error_vectors(v_gt, yaw_pred, pitch_pred):
    """Angular error (degrees) between 3D GT gaze (from .mat) and predicted yaw/pitch."""
    v_gt = np.asarray(v_gt, dtype=np.float64).flatten()
    v_gt /= np.linalg.norm(v_gt)

    # Model outputs angles in the same convention as MPIIGaze labels (pitch, yaw)
    v_pred = mpiigaze_angles_to_vector(pitch_pred, yaw_pred)

    dot = np.clip(np.dot(v_gt, v_pred), -1.0, 1.0)
    return np.rad2deg(np.arccos(dot))


def angular_error(yaw_gt, pitch_gt, yaw_pred, pitch_pred):
    v_gt = mpiigaze_angles_to_vector(pitch_gt, yaw_gt)
    v_pred = mpiigaze_angles_to_vector(pitch_pred, yaw_pred)

    dot = np.clip(np.dot(v_gt, v_pred), -1.0, 1.0)
    return np.rad2deg(np.arccos(dot))


def gaze_to_vector(yaw, pitch):
    """Legacy helper (non-MPIIGaze sign). Prefer mpiigaze_angles_to_vector."""
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)

    x = np.cos(pitch) * np.sin(yaw)
    y = np.sin(pitch)
    z = np.cos(pitch) * np.cos(yaw)

    return np.array([x, y, z])
