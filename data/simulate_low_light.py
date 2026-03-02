import cv2
import numpy as np

def simulate_low_intensity(image, reduction=70):
    """
    Uniform intensity reduction. 
    This does NOT create shadows
    It does NOT change contrast

    What does this simulate: 
        Dim room (Everything looks darker), 
        Low camera exposure (All pixels lose brightness), 
        Weak lighting (No shadows, just darker)

    Mathematical Idea: I_dark = max(I_original - k, 0)
    """
    image = image.astype(np.int16)  # Subtraction can produce negative values, and uint8 cannot handle negative numbers properly.
    dark = image - reduction
    dark = np.clip(dark, 0, 255)
    return dark.astype(np.uint8)


def simulate_gamma_dark(image, gamma=0.3):
    """
    Gamma-based darkening.
    gamma < 1 makes image darker.
    """
    image_norm = image / 255.0
    dark = np.power(image_norm, 1 / gamma)
    dark = np.clip(dark * 255, 0, 255)
    return dark.astype(np.uint8)


def simulate_directional_dark(image, direction="left", strength=0.2): # strength: How dark the far side becomes
    """
    Simulates side lighting using a horizontal gradient.
    """
    h, w, _ = image.shape

    if direction == "left":
        gradient = np.linspace(1.0, strength, w)
    else:  # right
        gradient = np.linspace(strength, 1.0, w)

    mask = np.tile(gradient, (h, 1)) # Create a 2D mask from the gradient
    mask = np.expand_dims(mask, axis=2) # adds a new dimension at position 2. This is necessary to make the mask compatible for element-wise multiplication with the 3-channel image.

    dark = image * mask
    return dark.astype(np.uint8)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    test_image_path = "data/original/MPIIGaze/Data/Original/p00/day01/0001.jpg"

    image = cv2.imread(test_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    low_i = simulate_low_intensity(image)
    low_g = simulate_gamma_dark(image)
    low_left = simulate_directional_dark(image, "left")
    low_right = simulate_directional_dark(image, "right")

    titles = ["Original", "Low Intensity", "Gamma Dark", "Left Light", "Right Light"]
    images = [image, low_i, low_g, low_left, low_right]

    plt.figure(figsize=(12, 4))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()
