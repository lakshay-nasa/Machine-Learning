import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('./images/nature.jpg', cv2.IMREAD_GRAYSCALE)

# Define the Prewitt kernels
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# Apply the Prewitt operator
prewitt_x_result = cv2.filter2D(image, cv2.CV_64F, prewitt_x)
prewitt_y_result = cv2.filter2D(image, cv2.CV_64F, prewitt_y)

# Calculate the gradient magnitude
gradient_magnitude = np.sqrt(prewitt_x_result**2 + prewitt_y_result**2)

# Display the original image, x-gradient, y-gradient, and gradient magnitude
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(prewitt_x_result, cmap='gray')
plt.title('X-Gradient (Prewitt)')

plt.subplot(2, 2, 3)
plt.imshow(prewitt_y_result, cmap='gray')
plt.title('Y-Gradient (Prewitt)')

plt.subplot(2, 2, 4)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient Magnitude')

plt.tight_layout()
plt.show()
