from l2cs import getArch
import torch
import torch.nn as nn
import numpy as np
import cv2
import os


class L2CSBlackBox:
    """
    Black-box wrapper for official L2CS model.
    Adapted for MPIIFaceGaze (no fold-specific weights).
    """

    def __init__(self):

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # L2CS configuration
        self.num_bins = 90
        self.bin_width = 2  # Gaze360 uses 2-degree bins
        self.angle_offset = -90

        # Build official architecture
        self.model = getArch("ResNet50", self.num_bins)

        # ------------------------------------------------------------------
        # IMPORTANT:
        # For MPIIFaceGaze, use a SINGLE pretrained L2CS model.
        # Do NOT use fold-specific weights.
        # ------------------------------------------------------------------

        weights_path = os.path.join(
            os.path.dirname(__file__),
            "../models/l2cs/L2CSNet_gaze360.pkl"
        )

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"L2CS weight file not found at: {weights_path}"
            )

        # Load weights
        state_dict = torch.load(weights_path, map_location=self.device)

        # Remove 'module.' prefix if exists
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                clean_state_dict[k[7:]] = v
            else:
                clean_state_dict[k] = v

        self.model.load_state_dict(clean_state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.softmax = nn.Softmax(dim=1)
        self.idx_tensor = torch.arange(self.num_bins).float().to(self.device)

        print(f"[L2CS] Model loaded on device: {self.device}")

    def predict(self, image):
        """
        Input:
            image: BGR uint8 (full face image)

        Output:
            yaw, pitch (degrees)
        """

        # Convert OpenCV BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        image = cv2.resize(image, (448, 448))

        # Normalize to 0–1
        image = image / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # HWC → CHW
        image = np.transpose(image, (2, 0, 1))

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pitch_logits, yaw_logits = self.model(image)

            pitch_prob = self.softmax(pitch_logits)
            yaw_prob = self.softmax(yaw_logits)

            pitch = (
                torch.sum(pitch_prob * self.idx_tensor, dim=1)
                * self.bin_width
                + self.angle_offset
            )

            yaw = (
                torch.sum(yaw_prob * self.idx_tensor, dim=1)
                * self.bin_width
                + self.angle_offset
            )

        return yaw.item(), pitch.item()