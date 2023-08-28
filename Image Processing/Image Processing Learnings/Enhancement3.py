# Filtering For edge Detection


# Derivative Filters For Discontinuities



    # First Order Edge Detection



        # Linearly Seperable Filtering



    # Second Order Edge Detection

        # Laplacian edge detection

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the image and convert it to grayscale
image = cv2.imread('./standard_test_images/peppers_gray.tif')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Create Laplacian filter
laplacian_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

# Step 3: Apply the Laplacian filter to the grayscale image
laplacian_edges = cv2.filter2D(gray_image, -1, laplacian_filter)

# Step 4: Display the original and edge-detected images
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(laplacian_edges, cmap='gray')
plt.title('Laplacian Edges')
plt.axis('off')

plt.show()


    # Laplacian of Gaussian

    # Zero-crossing detector

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image_path = './standard_test_images/peppers_gray.tif'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Laplacian operator to compute the second derivative
laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

# Find the zero-crossings using thresholding
threshold = 100
zero_crossings = np.zeros_like(laplacian, dtype=np.uint8)
zero_crossings[laplacian > threshold] = 255

# Display the original and edge-detected images side by side using matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(zero_crossings, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

plt.show()


# Edge enhancement

    # Laplacian edge sharpening

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the image using OpenCV
image = cv2.imread('./standard_test_images/cameraman.tif', cv2.IMREAD_GRAYSCALE)

# Step 2: Generate a 3x3 Laplacian filter
laplacian_filter = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

# Step 3: Filter the image with the Laplacian kernel
laplacian_image = cv2.filter2D(image, -1, laplacian_filter)

# Step 4: Subtract the Laplacian-filtered image from the original image
enhanced_image = image - laplacian_image

# Step 5: Display the original, Laplacian, and enhanced images
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(laplacian_image, cmap='gray')
plt.title('Laplacian Filtered Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(enhanced_image, cmap='gray')
plt.title('Enhanced Image')
plt.axis('off')

plt.show()



# Below code uses opencv function but creates a blurred iamge why?
import cv2
import matplotlib.pyplot as plt

# Step 1: Read the image using OpenCV
image = cv2.imread('./standard_test_images/cameraman.tif', cv2.IMREAD_GRAYSCALE)

# Step 2: Apply the Laplacian filter to the image
laplacian_image = cv2.Laplacian(image, cv2.CV_64F)

# Step 3: Convert the Laplacian image to uint8 (8-bit) and take its absolute value
laplacian_image = cv2.convertScaleAbs(laplacian_image)

# Step 4: Subtract the Laplacian-filtered image from the original image
enhanced_image = cv2.subtract(image, laplacian_image)

# Step 5: Display the original, Laplacian, and enhanced images
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(laplacian_image, cmap='gray')
plt.title('Laplacian Filtered Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(enhanced_image, cmap='gray')
plt.title('Enhanced Image')
plt.axis('off')

plt.show()



    # The unsharp mask filter

# Read the original image and convert it to grayscale
image_path = './standard_test_images/cameraman.tif'
Jorig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Generate Gaussian kernel
kernel_size = (55, 55)
sigma = 1.5
g = cv2.getGaussianKernel(kernel_size[0], sigma) * cv2.getGaussianKernel(kernel_size[1], sigma).T

# Apply Gaussian smoothing to the original image
Is = cv2.filter2D(Jorig, -1, g)

# Compute the difference image (unsharp mask) which can be considered as an edge-enhanced image
le = Jorig.astype(np.float32) - Is.astype(np.float32)

# Varying scaling factors
scaling_factors = [0.3, 0.5, 0.7, 2.0]

# Initialize the plot
plt.figure(figsize=(15, 10))

# Display the original image
plt.subplot(2, 3, 1)
plt.imshow(Jorig, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Display the edge-enhanced image (difference image)
plt.subplot(2, 3, 2)
plt.imshow(le, cmap='gray')
plt.title('Edge-Enhanced Image')
plt.axis('off')

# Apply the unsharp mask with different scaling factors and display the results
for i, k in enumerate(scaling_factors):
    lout = Jorig.astype(np.float32) + k * le
    lout = np.clip(lout, 0, 255).astype(np.uint8)

    plt.subplot(2, 3, i + 3)
    plt.imshow(lout, cmap='gray')
    plt.title(f"k = {k}")
    plt.axis('off')

plt.tight_layout()
plt.show()

