from l2cs import getArch
import torch
import torch.nn as nn
import numpy as np
import cv2
import os

class L2CSBlackBox:
    def __init__(self, subject_id):
        # OPTIMIZATION: Use GPU if available to speed up processing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_bins = 28
        self.bin_width = 3
        self.angle_offset = -42

        # Build official architecture
        self.model = getArch("ResNet50", self.num_bins)

        # Extract fold index from subject string (e.g., "p06" → 6)
        fold_index = int(subject_id[1:])
        weights_path = os.path.join(os.path.dirname(__file__), "../", "models", "l2cs", "weights", f"fold{fold_index}.pkl")
        
        # Load weights mapping to the correct device
        state_dict = torch.load(weights_path, map_location=self.device)
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

    def predict(self, image):
        # CRITICAL FIX: Convert OpenCV BGR to RGB before feeding to the model
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = cv2.resize(image, (448, 448))
        image = image / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pitch_logits, yaw_logits = self.model(image)

            pitch_prob = self.softmax(pitch_logits)
            yaw_prob = self.softmax(yaw_logits)

            pitch = torch.sum(pitch_prob * self.idx_tensor, dim=1) * self.bin_width + self.angle_offset
            yaw = torch.sum(yaw_prob * self.idx_tensor, dim=1) * self.bin_width + self.angle_offset

        return yaw.item(), pitch.item()