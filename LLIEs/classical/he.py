import cv2
import numpy as np

def histogram_equalization(image):
    """
    Apply Histogram Equalization on luminance channel.
    """
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # Equalize Y channel
    y_eq = cv2.equalizeHist(y)

    # Merge back
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    enhanced = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    return enhanced
