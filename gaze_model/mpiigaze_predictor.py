"""
MPIIGaze eye-patch gaze estimator (ResNet-preact, hysts/pytorch_mpiigaze weights).

Weights: scripts/download_pretrained_gaze_weights.py
  -> models/mpiigaze/resnet_preact/foldXX/checkpoint_0040.pth
"""

import os

import cv2
import numpy as np
import torch
import yacs.config

from gaze_model.mpiigaze_resnet import MPIIGazeResNetPreact

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_NAME = os.environ.get("MPIIGAZE_CHECKPOINT", "checkpoint_0040.pth")
EYE_SIZE = (60, 36)  # width, height — MPIIGaze / hysts (36 x 60)


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
            "Run: python scripts/setup_environment.py"
        )
    return path


def _build_config() -> yacs.config.CfgNode:
    cfg = yacs.config.CfgNode()
    cfg.mode = "MPIIGaze"
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.model = yacs.config.CfgNode()
    cfg.model.name = "resnet_preact"
    return cfg


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
        self.subject_id = subject_id
        self.config = _build_config()
        self.device = torch.device(self.config.device)

        self.model = MPIIGazeResNetPreact(self.config)
        checkpoint = torch.load(_checkpoint_path(subject_id), map_location="cpu")
        state = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_bgr, pose_rad: np.ndarray, eye: str = "left"):
        """
        Returns (yaw_deg, pitch_deg) for evaluation.angular_error_vectors.
        Model outputs (pitch, yaw) in radians.
        """
        tensor, pose = prepare_eye_tensor(image_bgr, eye, pose_rad, self.device)

        with torch.no_grad():
            angles_rad = self.model(tensor, pose)

        pitch_rad = float(angles_rad[0, 0].cpu())
        yaw_rad = float(angles_rad[0, 1].cpu())
        return np.rad2deg(yaw_rad), np.rad2deg(pitch_rad)
