import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def filterGaussian(image, kernel_size, kernel_sigma, border_type, separable):
    print("I am here")
    for w in width:
        for h in height:

def apply_gaussian_filter(image, sigma):
    if len(image.shape) == 3:
        filtered_image = np.zeros_like(image)
        for i in range(3):  # Assuming the image has 3 channels (RGB)
            filtered_image[:, :, i] = gaussian_filter(image[:, :, i], sigma=sigma)
    else:
        # If it's a grayscale image (2D array), apply the filter directly
        filtered_image = gaussian_filter(image, sigma=sigma)

    return filtered_image

img = cv2.imread("task1.jpg", cv2.IMREAD_COLOR)
image_np = np.array(img)

width = img.shape[0]
height = img.shape[1]

# Apply Gaussian filter with sigma=2
sigma_value = 2
filtered_image = apply_gaussian_filter(image_np, sigma=sigma_value)

    # Plot the original and filtered images
    # plt.figure(figsize=(10, 5))
    #
    # plt.subplot(1, 2, 1)
    # plt.title("Original Image")
    # plt.imshow(image_np)
    #
    # plt.subplot(1, 2, 2)
    # plt.title(f"Filtered Image (sigma={sigma_value})")
    # plt.imshow(filtered_image)

    # plt.show()

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()