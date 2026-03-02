import cv2
import numpy as np

def clahe_enhancement(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Split image into small tiles
    Apply histogram equalization inside each tile
    Limit contrast so noise doesn't explode
    Smoothly blend tiles together
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    y_clahe = clahe.apply(y)

    ycrcb_clahe = cv2.merge((y_clahe, cr, cb))
    enhanced = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)

    return enhanced
