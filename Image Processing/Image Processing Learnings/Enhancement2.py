# Adding Salt & Pepper and Gaussian Noise to an Image

import cv2
import matplotlib.pyplot as plt
from skimage.util import random_noise
import numpy as np
from scipy.ndimage import rank_filter


def add_and_show_noise(image, noise_type, noise_param, title):
    noisy_img = random_noise(image, mode=noise_type, **noise_param)
    return noisy_img

# Step 1: Read the image
image_path = './standard_test_images/lake.tif'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 2: Display the original image
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Step 3: Add and display salt and pepper noise
noise_density = 0.03
sp_noise_img = add_and_show_noise(img, 's&p', {'amount': noise_density}, 'Salt & Pepper Noise')

# Step 4: Add and display Gaussian noise
gaussian_noise_variance = 0.02
gaussian_noise_img = add_and_show_noise(img, 'gaussian', {'var': gaussian_noise_variance}, 'Gaussian Noise')

# Create a side-by-side display of all three images
plt.subplot(1, 3, 2)
plt.imshow(sp_noise_img, cmap='gray')
plt.title('Salt & Pepper Noise')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gaussian_noise_img, cmap='gray')
plt.title('Gaussian Noise')
plt.axis('off')

# Save the images as PNG files
output_dir = './output_images/'
cv2.imwrite(output_dir + 'original_image.png', img)
cv2.imwrite(output_dir + 'salt_and_pepper_noise.png', (sp_noise_img * 255).astype('uint8'))
cv2.imwrite(output_dir + 'gaussian_noise.png', (gaussian_noise_img * 255).astype('uint8'))

plt.tight_layout()
plt.show()


# Mean Filtering

# Load the original image
original_image_path = './output_images/original_image.png'
original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

# Load the salt and pepper image
salt_and_pepper_image_path = './output_images/salt_and_pepper_noise.png'
salt_and_pepper_image = cv2.imread(salt_and_pepper_image_path, cv2.IMREAD_GRAYSCALE)

# Load the Gaussian image
gaussian_image_path = './output_images/gaussian_noise.png'
gaussian_image = cv2.imread(gaussian_image_path, cv2.IMREAD_GRAYSCALE)

# Define the mean filter kernel
k = np.ones((3, 3), dtype=np.float32) / 9

# Apply mean filtering to the original image
mean_filtered_original = cv2.filter2D(original_image, -1, k)

# Apply mean filtering to the salt and pepper image
mean_filtered_salt_and_pepper = cv2.filter2D(salt_and_pepper_image, -1, k)

# Apply mean filtering to the Gaussian image
mean_filtered_gaussian = cv2.filter2D(gaussian_image, -1, k)

# Display the images using matplotlib
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(mean_filtered_original, cmap='gray')
plt.title('Mean Filtered Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mean_filtered_salt_and_pepper, cmap='gray')
plt.title('Mean Filtered Salt and Pepper Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(mean_filtered_gaussian, cmap='gray')
plt.title('Mean Filtered Gaussian Image')
plt.axis('off')

plt.show()



# Median Filtering

# Load the original image, salt and pepper image, and image with Gaussian noise
I = cv2.imread('./output_images/original_image.png', cv2.IMREAD_GRAYSCALE)
Isp = cv2.imread('./output_images/salt_and_pepper_noise.png', cv2.IMREAD_GRAYSCALE)
Ig = cv2.imread('./output_images/gaussian_noise.png', cv2.IMREAD_GRAYSCALE)

# Apply median filter to the original image
I_med = cv2.medianBlur(I, ksize=3)

# Apply median filter to the salt and pepper image
Isp_med = cv2.medianBlur(Isp, ksize=3)

# Apply median filter to the image with Gaussian noise
Ig_med = cv2.medianBlur(Ig, ksize=3)

# Display the results using matplotlib
plt.subplot(1, 3, 1)
plt.imshow(I_med, cmap='gray')
plt.title('Filtered Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(Isp_med, cmap='gray')
plt.title('Filtered Salt and Pepper Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(Ig_med, cmap='gray')
plt.title('Filtered Image with Gaussian Noise')
plt.axis('off')

plt.show()




# Rank Filtering


# Function to apply rank filter to an image
def apply_rank_filter(image, rank, kernel_size):
    filtered_image = rank_filter(image, rank, size=kernel_size)
    return filtered_image

# Load the original image, salt and pepper image, and Gaussian image
original_image = cv2.imread('./output_images/original_image.png', cv2.IMREAD_GRAYSCALE)
salt_pepper_image = cv2.imread('./output_images/salt_and_pepper_noise.png', cv2.IMREAD_GRAYSCALE)
gaussian_image = cv2.imread('./output_images/gaussian_noise.png', cv2.IMREAD_GRAYSCALE)

# Apply rank filtering to the images
rank = 24
kernel_size = (5, 5)
filtered_original = apply_rank_filter(original_image, rank, kernel_size)
filtered_salt_pepper = apply_rank_filter(salt_pepper_image, rank, kernel_size)
filtered_gaussian = apply_rank_filter(gaussian_image, rank, kernel_size)

# Display the images side by side using matplotlib
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(filtered_original, cmap='gray')
plt.title('Filtered Original')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(filtered_salt_pepper, cmap='gray')
plt.title('Filtered Salt and Pepper Image')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(filtered_gaussian, cmap='gray')
plt.title('Filtered Gaussian Image')
plt.axis('off')

plt.show()



# Gaussian Filtering


def apply_gaussian_filter(image, kernel_size, sigma):
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel = np.outer(gaussian_kernel, gaussian_kernel)
    return cv2.filter2D(image, -1, gaussian_kernel)

# Step 1: Read the three images
original_image_path = './output_images/original_image.png'
salt_and_pepper_image_path = './output_images/salt_and_pepper_noise.png'
gaussian_noise_image_path = './output_images/gaussian_noise.png'

original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
salt_and_pepper_image = cv2.imread(salt_and_pepper_image_path, cv2.IMREAD_GRAYSCALE)
gaussian_noise_image = cv2.imread(gaussian_noise_image_path, cv2.IMREAD_GRAYSCALE)

# Step 2: Apply Gaussian filter to the three images
kernel_size = 55
sigma = 2

filtered_original_image = apply_gaussian_filter(original_image, kernel_size, sigma)
filtered_salt_and_pepper_image = apply_gaussian_filter(salt_and_pepper_image, kernel_size, sigma)
filtered_gaussian_noise_image = apply_gaussian_filter(gaussian_noise_image, kernel_size, sigma)

# Step 3: Display the four images side by side
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 4, 4)
plt.imshow(filtered_original_image, cmap='gray')
plt.title('Filtered Original')

plt.subplot(1, 4, 2)
plt.imshow(salt_and_pepper_image, cmap='gray')
plt.title('Salt and Pepper Noise')

plt.subplot(1, 4, 3)
plt.imshow(gaussian_noise_image, cmap='gray')
plt.title('Gaussian Noise')



plt.tight_layout()
plt.show()

# 4. 5 

# Filtering for edge Detection

# 1. First-order edge detection

# 