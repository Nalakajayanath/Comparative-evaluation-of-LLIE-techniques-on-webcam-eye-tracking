"""
MPIIGaze eye-patch gaze estimator (hysts/pytorch_mpiigaze ResNet-preact).

Requires vendor/pytorch_mpiigaze (scripts/setup_environment.py) and weights from
scripts/download_pretrained_gaze_weights.py
"""

import os
import sys

import cv2
import numpy as np
import torch
import yacs.config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR_ROOT = os.path.join(PROJECT_ROOT, "vendor", "pytorch_mpiigaze")
CHECKPOINT_NAME = os.environ.get("MPIIGAZE_CHECKPOINT", "checkpoint_0040.pth")
EYE_SIZE = (60, 36)  # width, height — matches MPIIGaze / hysts (36 x 60)


def _ensure_vendor_on_path():
    if not os.path.isdir(VENDOR_ROOT):
        raise FileNotFoundError(
            f"Missing {VENDOR_ROOT}. Run: python scripts/setup_environment.py"
        )
    if VENDOR_ROOT not in sys.path:
        sys.path.insert(0, VENDOR_ROOT)


def _checkpoint_path(subject_id: str) -> str:
    fold = int(subject_id[1:])
    path = os.path.join(
        PROJECT_ROOT,
        "models",
        "mpiigaze",
        "resnet_preact",
        f"fold{fold:02d}",
        CHECKPOINT_NAME,
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing gaze weights: {path}\n"
            "Run: python scripts/download_pretrained_gaze_weights.py"
        )
    return path


def _build_config() -> yacs.config.CfgNode:
    _ensure_vendor_on_path()
    from gaze_estimation.config import get_default_config

    config = get_default_config()
    config.defrost()
    config.mode = "MPIIGaze"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.model.name = "resnet_preact"
    config.freeze()
    return config


def prepare_eye_tensor(image_bgr, eye: str, pose_rad: np.ndarray, device: torch.device):
    """BGR/gray eye image -> (1,1,36,60) tensor + pose (1,2), hysts conventions."""
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = np.asarray(image_bgr)

    if gray.shape[1] != EYE_SIZE[0] or gray.shape[0] != EYE_SIZE[1]:
        gray = cv2.resize(gray, EYE_SIZE)

    if eye == "right":
        gray = gray[:, ::-1]

    tensor = gray.astype(np.float32) / 255.0
    tensor = torch.from_numpy(tensor).unsqueeze(0).unsqueeze(0).to(device)
    pose = torch.from_numpy(np.asarray(pose_rad, dtype=np.float32)).unsqueeze(0).to(device)
    return tensor, pose


class MPIIGazePredictor:
    def __init__(self, subject_id: str):
        _ensure_vendor_on_path()
        from gaze_estimation import create_model

        self.subject_id = subject_id
        self.config = _build_config()
        self.device = torch.device(self.config.device)

        self.model = create_model(self.config)
        checkpoint = torch.load(_checkpoint_path(subject_id), map_location="cpu")
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_bgr, pose_rad: np.ndarray, eye: str = "left"):
        """
        Returns (yaw_deg, pitch_deg) for compatibility with evaluation.angular_error_vectors.
        Model internally uses radians (pitch, yaw).
        """
        tensor, pose = prepare_eye_tensor(image_bgr, eye, pose_rad, self.device)

        with torch.no_grad():
            angles_rad = self.model(tensor, pose)

        pitch_rad = float(angles_rad[0, 0].cpu())
        yaw_rad = float(angles_rad[0, 1].cpu())
        return np.rad2deg(yaw_rad), np.rad2deg(pitch_rad)
