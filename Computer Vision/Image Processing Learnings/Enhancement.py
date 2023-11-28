import cv2
import numpy as np
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt

# Image Sharpening with Minimum Filter

# Read the image
A = cv2.imread('./standard_test_images/cameraman.tif', cv2.IMREAD_GRAYSCALE)

# Display the original image using matplotlib
plt.subplot(1, 2, 1)
plt.imshow(A, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Define the filter function (min filter over a 3x3 neighborhood)
def min_filter(x):
    return np.min(x)

# Apply the filter over the 3x3 neighborhood using scipy.ndimage.generic_filter
B = generic_filter(A, min_filter, size=(3, 3))

# Display the resulting image using matplotlib
plt.subplot(1, 2, 2)
plt.imshow(B, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')

plt.show()





# Motion Blur Effect

# Read in image
A = cv2.imread('./standard_test_images/peppers_color.tif')

# Display image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(A, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Create a motion blur convolution kernel
kernel_size = 50
angle = 54
kernel = np.zeros((kernel_size, kernel_size))
kernel[:, int((kernel_size - 1) / 2)] = 1.0 / kernel_size

# Rotate the kernel using cv2.getRotationMatrix2D
M = cv2.getRotationMatrix2D((int((kernel_size - 1) / 2), int((kernel_size - 1) / 2)), angle, 1)
kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

# Apply using symmetric mirroring at edges
B = cv2.filter2D(A, -1, kernel, borderType=cv2.BORDER_REFLECT)

# Display result image B
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(B, cv2.COLOR_BGR2RGB))
plt.title('Motion Blurred Image')

plt.show()
