import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Load the image and convert it to grayscale
image = cv2.imread("./images/f_bg3.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Preprocess the image (optional)
# e.g., apply denoising, histogram equalization, etc.

# Flatten the pixels
pixels = gray.flatten().reshape(-1, 1)

# Determine the optimal number of components using BIC
best_bic = float('inf')
best_gmm = None
for n_components in range(1, 6):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(pixels)
    bic = gmm.bic(pixels)
    if bic < best_bic:
        best_bic = bic
        best_gmm = gmm

# Apply Gaussian Mixture Model (GMM) for segmentation with the best number of components
labels = best_gmm.predict(pixels)

# Reshape the labels to create a mask
mask = labels.reshape(gray.shape)

# Create foreground and background images using GMM means
foreground_mean = best_gmm.means_[1].astype(np.uint8)
background_mean = best_gmm.means_[0].astype(np.uint8)

# Convert mask to three-dimensional array to match the image dimensions
mask_3d = np.expand_dims(mask, axis=-1)

# Create foreground and background images using broadcasting
foreground_img = np.where(mask_3d, image, foreground_mean)
background_img = np.where(mask_3d, background_mean, image)

# Display the images
plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(132)
plt.imshow(cv2.cvtColor(foreground_img, cv2.COLOR_BGR2RGB))
plt.title("Foreground")

plt.subplot(133)
plt.imshow(cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB))
plt.title("Background")

plt.show()
