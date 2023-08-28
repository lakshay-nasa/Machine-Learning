import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./images/nature.jpg')
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(RGB_img, cv2.COLOR_RGB2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Compute Sobel gradients
sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)  # x
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)  # y

# Calculate the magnitude of gradients
gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

# Create a 1x4 grid for plotting
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Plot the original image
axes[0].imshow(RGB_img)
axes[0].set_title("Original Image")

# Plot Sobel-x edge detection
axes[1].imshow(sobelx, cmap='gray')
axes[1].set_title("Sobel-x Edge Detection")

# Plot Sobel-y edge detection
axes[2].imshow(sobely, cmap='gray')
axes[2].set_title("Sobel-y Edge Detection")

# Plot Combined Sobel edges
axes[3].imshow(gradient_magnitude, cmap='gray')
axes[3].set_title("Combined Sobel Edges")

plt.show()
