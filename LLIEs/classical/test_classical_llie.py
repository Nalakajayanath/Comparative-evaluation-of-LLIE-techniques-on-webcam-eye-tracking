import cv2
import matplotlib.pyplot as plt
from he import histogram_equalization
from clahe import clahe_enhancement

# Test image (use low-light image)
test_path = "data/low_light_simulated/low_i/p00/day08/0069.jpg"

image = cv2.imread(test_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

he_img = histogram_equalization(image)
clahe_img = clahe_enhancement(image)

he_rgb = cv2.cvtColor(he_img, cv2.COLOR_BGR2RGB)
clahe_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Low-Light Input")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(he_rgb)
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(clahe_rgb)
plt.title("CLAHE")
plt.axis("off")

plt.tight_layout()
plt.show()
