import torch
import cv2
import numpy as np
import sys
import os

# Add Zero-DCE path
sys.path.append("Zero-DCE/Zero-DCE_code")

from model import enhance_net_nopool

class ZeroDCE:
    def __init__(self, weights_path=os.path.join(os.path.dirname(__file__), "../../", "models", "zerodce", "Epoch99.pth")):

        self.device = torch.device("cpu")

        self.model = enhance_net_nopool().to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def enhance(self, image):

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device)
        
        with torch.no_grad():
            output = self.model(image)

        # If model returns multiple outputs, take first one
        if isinstance(output, tuple) or isinstance(output, list):
            enhanced_image = output[0]
        else:
            enhanced_image = output

        enhanced_image = enhanced_image.squeeze().permute(1, 2, 0).cpu().numpy()
        enhanced_image = np.clip(enhanced_image, 0, 1)

        enhanced_image = (enhanced_image * 255).astype(np.uint8)

        # Convert back to BGR
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)

        return enhanced_image