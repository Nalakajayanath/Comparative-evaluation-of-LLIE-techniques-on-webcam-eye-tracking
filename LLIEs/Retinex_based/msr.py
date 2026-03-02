import cv2
import numpy as np

def multi_scale_retinex(image, sigmas=[15, 80, 250]):
    """
    Multi-Scale Retinex (MSR)

    Parameters:
        image  : BGR uint8 image
        sigmas : list of Gaussian blur sigmas

    Returns:
        Enhanced BGR uint8 image
    """

    img = image.astype(np.float32) + 1.0
    retinex = np.zeros_like(img)

    # Compute SSR for each sigma and accumulate
    for sigma in sigmas:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex += np.log(img) - np.log(blur)

    # Average across scales
    retinex = retinex / len(sigmas)

    # Normalize each channel independently
    for i in range(3):
        channel = retinex[:, :, i]
        channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel)) * 255
        retinex[:, :, i] = channel

    retinex = np.clip(retinex, 0, 255)

    return retinex.astype(np.uint8)