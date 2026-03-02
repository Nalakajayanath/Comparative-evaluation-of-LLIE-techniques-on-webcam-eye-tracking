import cv2
import numpy as np
from enlighten_inference import EnlightenOnnxModel


class EnlightenGAN:

    def __init__(self):
        # Automatically loads included ONNX model
        self.model = EnlightenOnnxModel()

    def enhance(self, image):
        """
        image: BGR numpy image (uint8)
        return: enhanced BGR image
        """

        # Convert BGR → RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize to 0-1
        image_rgb = image_rgb.astype(np.float32) / 255.0

        # Run inference
        enhanced = self.model.predict(image_rgb)

        # Convert back to uint8
        enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)

        # RGB → BGR
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

        return enhanced