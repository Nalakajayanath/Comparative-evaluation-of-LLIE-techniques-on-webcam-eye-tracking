import cv2
from enlighten_inference import EnlightenOnnxModel

class EnlightenGAN:

    def __init__(self):
        # Explicitly declare the CPU provider to suppress the CUDA warning
        self.model = EnlightenOnnxModel(providers=["CPUExecutionProvider"])

    def enhance(self, image):
        """
        image: BGR numpy image (uint8)
        return: enhanced BGR image
        """
        
        # The enlighten_inference wrapper natively handles:
        # - BGR to RGB conversion
        # - Normalization to 0-1 and tensor formatting
        # - Model Inference
        # - Converting the output tensor back to a uint8 BGR image
        
        enhanced = self.model.predict(image)

        return enhanced