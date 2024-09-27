import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def gaussian_kernel_1d(size, sigma):
    """
    Creates a 1D Gaussian kernel.

    Parameters:
    - size: The size of the kernel (must be odd).
    - sigma: The standard deviation of the Gaussian.

    Returns:
    - A 1D numpy array representing the Gaussian kernel.
    """
    kernel = np.fromfunction(
        lambda x: (1 / (np.sqrt(2 * np.pi) * sigma)) *
                  np.exp(-((x - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size,)
    )
    return kernel / np.sum(kernel)  # Normalize the kernel so the sum is 1


def apply_gaussian_filter_1d(image, sigma, kernel_size):
    """
    Apply a Gaussian filter to an image using a 1D Gaussian kernel.

    Parameters:
    - image: A 2D or 3D numpy array representing the image.
    - sigma: The standard deviation of the Gaussian kernel.
    - kernel_size: The size of the Gaussian kernel (should be odd).

    Returns:
    - filtered_image: The image after applying the Gaussian filter.
    """
    # Create the 1D Gaussian kernel
    kernel = gaussian_kernel_1d(kernel_size, sigma)

    # Initialize the output image
    if len(image.shape) == 3:  # Color image (3D array)
        filtered_image = np.zeros_like(image)
        for i in range(3):  # Apply the filter to each channel separately
            filtered_image[:, :, i] = convolve_1d(image[:, :, i], kernel)
    else:  # Grayscale image (2D array)
        filtered_image = convolve_1d(image, kernel)

    return filtered_image


def convolve_1d(image, kernel):
    """
    Perform convolution between an image and a 1D kernel (first across rows, then columns).

    Parameters:
    - image: A 2D numpy array representing the image.
    - kernel: A 1D numpy array representing the convolution kernel.

    Returns:
    - result: The result of applying the convolution.
    """
    # Convolve along rows
    result = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=image)

    # Convolve along columns
    result = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=result)

    return result


# Example usage:
if __name__ == "__main__":
    # Load image using PIL and convert to grayscale numpy array
    image = Image.open('sample_image.jpg').convert('L')  # Convert to grayscale
    image_np = np.array(image)

    # Apply 1D Gaussian filter with sigma=2 and kernel size=5
    sigma_value = 2
    kernel_size = 5
    filtered_image = apply_gaussian_filter_1d(image_np, sigma=sigma_value, kernel_size=kernel_size)

    # Plot the original and filtered images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_np, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title(f"Filtered Image (sigma={sigma_value})")
    plt.imshow(filtered_image, cmap='gray')

    plt.show()
