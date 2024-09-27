import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time

def gaussian_kernel_1d(size, sigma):
    """
    Creates a 1D Gaussian kernel.
    """
    kernel = np.fromfunction(
        lambda x: (1 / (np.sqrt(2 * np.pi) * sigma)) *
                  np.exp(-((x - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size,)
    )
    return kernel / np.sum(kernel)
    # Normalize the kernel so the sum is 1


def gaussian_kernel_2d(size, kernel_sigma):
    """
    Creates a 2D Gaussian kernel.
    """
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * kernel_sigma ** 2)) *
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * kernel_sigma ** 2)),
        (size, size)
    )
    # print(kernel / np.sum(kernel))
    return kernel / np.sum(kernel)
    # Normalize the kernel so that the sum is 1

def apply_gaussian_filter(image, kernel_size, kernel_sigma, border_type, separable):
    # Initialize the output image
    if len(image.shape) == 3:  # Color image (3D array)
        filtered_image = np.zeros_like(image)
        kernel = gaussian_kernel_2d(kernel_size, kernel_sigma)
        separable, u, s, vt = check_separable(kernel)
        # separable = False
        print(f"Is the filter separable? {separable}")

        for i in range(3):  # Apply the filter to each channel separately
            if separable:  # 1d convolution
                print("The kernel can be expressed as the outer product of two 1D vectors.")
                row_vector = u[:, 0] * np.sqrt(s[0])
                col_vector = vt[0, :] * np.sqrt(s[0])
                print(f"Row Vector: {row_vector}")
                print(f"Column Vector: {col_vector}")
                filtered_image[:, :, i] = convolve_separable(image[:, :, i], row_vector, col_vector, border_type)
            else:  # 2d convolution
                filtered_image[:, :, i] = convolve_2d(image[:, :, i], kernel, border_type)
    else:  # Grayscale image (2D array)
        kernel = gaussian_kernel_2d(kernel_size, kernel_sigma)
        separable, u, s, vt = check_separable(kernel)
        print(f"Is the filter separable? {separable}")
        if separable:  # 1d convolution
            print("The kernel can be expressed as the outer product of two 1D vectors.")
            row_vector = u[:, 0] * np.sqrt(s[0])
            col_vector = vt[0, :] * np.sqrt(s[0])
            print(f"Row Vector: {row_vector}")
            print(f"Column Vector: {col_vector}")
            filtered_image = convolve_separable(image,  row_vector, col_vector, border_type)
        else:  # 2d convolution
            filtered_image = convolve_2d(image, kernel, border_type)

    return filtered_image

def convolve_2d(image, kernel, border_type):
    """
    Perform convolution between an image and a kernel.
    """
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    # Determine the padding size
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    # Pad the image to handle borders
    # padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=border_type)
    padded_image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, border_type, value=0)

    # padded_image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, border_type, None, value=0)

    # Create an empty output array
    result = np.zeros_like(image)

    # Perform convolution
    for i in range(image_h):
        for j in range(image_w):
            region = padded_image[i:i + kernel_h, j:j + kernel_w]
            result[i, j] = np.sum(region * kernel)
    return result


def convolve_1d(signal, kernel, border_type):
    """
    Perform 1D convolution between a signal (1D array) and a kernel (1D array).
    """
    signal_len = len(signal)
    kernel_len = len(kernel)
    pad_size = kernel_len // 2

    # Pad the signal with zeros
    # padded_image = np.pad(signal, (pad_size, pad_size), mode=border_type)
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, border_type, value=0)
    # Initialize result array
    result = np.zeros(signal_len)

    # Convolution loop
    for i in range(signal_len):
        result[i] = np.sum(padded_image[i:i + kernel_len] * kernel)

    return result


def convolve_separable(image, row_kernel, col_kernel, border_type):
    """
    Apply convolution using separable filters to a 2D image.
    """
    # Step 1: Convolve along the rows
    temp_result = np.zeros_like(image)
    for i in range(image.shape[0]):
        temp_result[i, :] = convolve_1d(image[i, :], row_kernel, border_type)

    # Step 2: Convolve along the columns
    final_result = np.zeros_like(image)
    for j in range(image.shape[1]):
        final_result[:, j] = convolve_1d(temp_result[:, j], col_kernel, border_type)

    return final_result

def check_separable(kernel):
    """
    Check if a 2D filter kernel is separable.
    """
    # Perform Singular Value Decomposition (SVD)
    u, s, vt = np.linalg.svd(kernel)

    # Check the number of non-zero singular values
    rank = np.sum(s > 1e-10)  # Count singular values above a threshold

    # If the rank is 1, the filter is separable
    return rank == 1, u, s, vt



# Load image
# image = Image.open('task1.jpg').convert('L')  # Convert to grayscale
image = Image.open('task1.jpg')
img = cv2.imread("task1.jpg", cv2.IMREAD_COLOR)
image_np = np.array(image)

# Apply Gaussian filter with kernel_sigma=2 and kernel size=5
kernel_sigma_value = 5
kernel_size = 10

border_type=cv2.BORDER_CONSTANT
# cv2.BORDER_CONSTANT
# cv2.BORDER_REPLICATE
# cv2.BORDER_REFLECT
# cv2.BORDER_WRAP
# cv2.BORDER_REFLECT_101
"""
‘constant’ 
‘edge’
‘linear_ramp’
‘maximum’
‘mean’
‘median’
‘minimum’
‘reflect’
‘symmetric’
‘wrap’
‘empty’
"""

start_time = time.time()
filtered_image = apply_gaussian_filter(image_np, kernel_sigma=kernel_sigma_value, kernel_size=kernel_size, border_type=border_type, separable=True)
print("--- %s seconds ---" % (time.time() - start_time))

# Plot the original and filtered images
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_np, cmap='gray')

plt.subplot(1, 2, 2)
plt.title(f"Filtered Image (kernel_sigma={kernel_sigma_value}  \n kernel_size={kernel_size})")
plt.imshow(filtered_image, cmap='gray')

plt.show()

plt.title(f"Filtered Image (kernel_sigma={kernel_sigma_value} \n kernel_size={kernel_size}")
plt.imshow(filtered_image, cmap='gray')
plt.show()
