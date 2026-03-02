import torch
import cv2
import numpy as np
import sys
import os

# -----------------------------
# Add MIRNet root to sys.path
# -----------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "../../../")
)

MIRNET_ROOT = os.path.join(PROJECT_ROOT, "MIRNet")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if MIRNET_ROOT not in sys.path:
    sys.path.insert(0, MIRNET_ROOT)

# Now import internal modules
#from MIRNet.networks import MIRNet_model
#from networks import MIRNet_model
from MIRNet.networks.MIRNet_model import MIRNet as MIRNetModel

class MIRNet:

    def __init__(self, weights_path=os.path.join(os.path.dirname(__file__), "../../../", "models", "mirnet", "model_lol.pth")):

        self.device = torch.device("cpu")

        self.model = MIRNetModel().to(self.device)

        checkpoint = torch.load(weights_path, map_location=self.device)

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v  # remove 'module.'
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict)
        self.model.eval()

    def enhance(self, image):
        """
        Input: BGR uint8
        Output: BGR uint8
        """

        # Convert BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize 0–1
        image = image.astype(np.float32) / 255.0

        # HWC → CHW
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            restored = self.model(image)

        restored = restored.squeeze().permute(1, 2, 0).cpu().numpy()
        restored = np.clip(restored, 0, 1)

        restored = (restored * 255).astype(np.uint8)

        # Convert back RGB → BGR
        restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)

        return restored