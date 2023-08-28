import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('./images/nature.jpg', cv2.IMREAD_GRAYSCALE)

# Apply the Robert Cross operator to get horizontal and vertical gradients
roberts_x = cv2.filter2D(image, cv2.CV_64F, np.array([[1, 0], [0, -1]]))
roberts_y = cv2.filter2D(image, cv2.CV_64F, np.array([[0, 1], [-1, 0]]))

# Calculate gradient magnitude
gradient_magnitude = np.sqrt(roberts_x ** 2 + roberts_y ** 2)

# Plot the original image, horizontal and vertical gradients, and gradient magnitude
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(roberts_x, cmap='gray')
plt.title('Roberts Operator (Horizontal Gradient)')

plt.subplot(2, 2, 3)
plt.imshow(roberts_y, cmap='gray')
plt.title('Roberts Operator (Vertical Gradient)')

plt.subplot(2, 2, 4)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient Magnitude')

plt.tight_layout()
plt.show()
