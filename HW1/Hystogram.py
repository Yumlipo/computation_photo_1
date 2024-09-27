import numpy as np
import matplotlib.pyplot as plt


def histogram_equalization(image):
    # Flatten the image to a 1D array
    image_flat = image.flatten()

    # Calculate the histogram
    histogram, _ = np.histogram(image_flat, bins=256, range=(0, 256))

    # Calculate the cumulative distribution function (CDF)
    cdf = histogram.cumsum()

    # Normalize the CDF
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)

    # Map the original pixel values to equalized values
    image_equalized = cdf_normalized[image_flat]

    # Reshape the equalized image back to the original shape
    image_equalized = image_equalized.reshape(image.shape)

    return image_equalized


# Load an image
image = plt.imread('gray3.jpg')

if len(image.shape) == 3:  # Color image (3D array)
    equalized_image = np.zeros_like(image)
    for i in range(3):  # Apply the filter to each channel separately
        # Apply histogram equalization
        equalized_image[:, :, i] = histogram_equalization(image[:, :, i])
else:  # Grayscale image (2D array)
    equalized_image = histogram_equalization(image)


# Display original and equalized images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Equalized Image")
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.show()
