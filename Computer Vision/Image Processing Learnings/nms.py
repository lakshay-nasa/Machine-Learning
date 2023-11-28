import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./images/nature.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to the image -> This helps to reduce noise in the image and make edges smoother.
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Compute gradient magnitude and direction using Sobel filters
gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_direction = np.arctan2(gradient_y, gradient_x) * 180 / np.pi

# Apply non-maximum suppression
rows, cols = gradient_magnitude.shape
nms_result = np.zeros((rows, cols), dtype=np.float32)

for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        angle = gradient_direction[i, j]
        angle = angle % 180
        
        if (0 <= angle < 22.5) or (157.5 <= angle < 180):
            neighbors = [gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]]
        elif 22.5 <= angle < 67.5:
            neighbors = [gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]
        elif 67.5 <= angle < 112.5:
            neighbors = [gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]
        else:
            neighbors = [gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]
        
        if gradient_magnitude[i, j] >= max(neighbors):
            nms_result[i, j] = gradient_magnitude[i, j]

# Apply double thresholding for edge detection
low_threshold = 30
high_threshold = 70
edges = np.zeros_like(nms_result)
edges[(nms_result > low_threshold) & (nms_result > high_threshold)] = 255
# edges = cv2.Canny(image, threshold1=100, threshold2=200)

# Display images side by side
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(nms_result, cmap='gray')
plt.title('Non-Maximum Suppression')

plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')

plt.tight_layout()
plt.show()
