import cv2
import numpy as np

# TODO: try different sigma values (15, 30, 60) to see which gives best results for gaze estimation.
def single_scale_retinex(image, sigma=30):
    """
    Single Scale Retinex (SSR)

    Parameters:
        image : BGR image (uint8)
        sigma : Gaussian blur standard deviation

    Returns:
        Enhanced BGR image (uint8)
    """

    # Convert to float and avoid log(0)
    img = image.astype(np.float32) + 1.0

    # Apply Gaussian blur (illumination estimation)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)

    # Retinex formula
    retinex = np.log(img) - np.log(blur)

    # Normalize each channel separately to 0–255
    for i in range(3):
        channel = retinex[:, :, i]
        channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel)) * 255
        retinex[:, :, i] = channel

    retinex = np.clip(retinex, 0, 255)

    return retinex.astype(np.uint8)