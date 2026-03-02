import cv2
import numpy as np


def msrcr(
    image,
    sigmas=[15, 80, 250],
    alpha=125,
    beta=46,
    gain=1,
    offset=0
):
    """
    Multi-Scale Retinex with Color Restoration (MSRCR)

    Parameters:
        image  : BGR uint8 image
        sigmas : list of Gaussian blur sigmas
        alpha  : color restoration parameter
        beta   : color restoration scaling
        gain   : final gain
        offset : final offset

    Returns:
        Enhanced BGR uint8 image
    """

    # Convert to float and avoid log(0)
    img = image.astype(np.float32) + 1.0

    # ---------- Multi-Scale Retinex ----------
    retinex = np.zeros_like(img)

    for sigma in sigmas:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex += np.log(img) - np.log(blur + 1e-6)

    retinex = retinex / len(sigmas)

    # ---------- Color Restoration ----------
    sum_channels = np.sum(img, axis=2, keepdims=True)

    color_restoration = beta * (
        np.log(alpha * img) - np.log(sum_channels + 1e-6)
    )

    # ---------- Combine ----------
    msrcr = gain * (retinex * color_restoration) + offset

    # ---------- Normalize each channel ----------
    for i in range(3):
        channel = msrcr[:, :, i]
        channel = (channel - np.min(channel)) / (
            np.max(channel) - np.min(channel) + 1e-6
        ) * 255
        msrcr[:, :, i] = channel

    msrcr = np.clip(msrcr, 0, 255)

    return msrcr.astype(np.uint8)