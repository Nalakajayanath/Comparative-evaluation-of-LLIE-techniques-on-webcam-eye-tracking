"""Head pose helpers aligned with hysts/pytorch_mpiigaze preprocessing."""

import cv2
import numpy as np


def convert_pose(pose_vector) -> np.ndarray:
    """Rodrigues rotation vector -> (pitch, yaw) in radians."""
    rot = cv2.Rodrigues(np.array(pose_vector, dtype=np.float32))[0]
    vec = rot[:, 2]
    pitch = np.arcsin(vec[1])
    yaw = np.arctan2(vec[0], vec[2])
    return np.array([pitch, yaw], dtype=np.float32)


def pose_for_eye(pose_vector, eye: str) -> np.ndarray:
    """Pose in radians; right eye uses the same flip convention as hysts."""
    pose = convert_pose(pose_vector)
    if eye == "right":
        pose = pose * np.array([1.0, -1.0], dtype=np.float32)
    return pose
