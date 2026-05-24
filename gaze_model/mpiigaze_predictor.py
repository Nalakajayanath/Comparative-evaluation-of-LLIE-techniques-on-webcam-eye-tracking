import os

import cv2
import numpy as np
import torch

from gaze_model.mpiigaze_resnet import MPIIGazeResNetPreact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EYE_SIZE = (60, 36)  # width, height


class MPIIGazePredictor:
    def __init__(self, subject_id: str):
        fold = int(subject_id[1:])
        ckpt = os.path.join(
            ROOT, "models", "mpiigaze", "resnet_preact",
            f"fold{fold:02d}", "checkpoint_0040.pth",
        )
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Missing weights: {ckpt}\nRun: python scripts/setup_environment.py")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MPIIGazeResNetPreact()
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def predict(self, image_bgr, pose_rad, eye="left"):
        if image_bgr.ndim == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr
        if gray.shape[1] != EYE_SIZE[0] or gray.shape[0] != EYE_SIZE[1]:
            gray = cv2.resize(gray, EYE_SIZE)
        if eye == "right":
            gray = gray[:, ::-1]

        x = torch.from_numpy(gray.astype(np.float32) / 255.0)
        x = x.unsqueeze(0).unsqueeze(0).to(self.device)
        pose = torch.from_numpy(np.asarray(pose_rad, dtype=np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(x, pose)

        pitch = float(out[0, 0].cpu())
        yaw = float(out[0, 1].cpu())
        return np.rad2deg(yaw), np.rad2deg(pitch)
