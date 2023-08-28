#                   Pixel Distributions: Histograms [3.4]


#       Image Display and Histogram Plotting

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read in the image using OpenCV
I = cv2.imread('./standard_test_images/pirate.tif', cv2.IMREAD_GRAYSCALE)

# Display the image using matplotlib
plt.subplot(1, 2, 1)
plt.imshow(I, cmap='gray')
plt.title('Image')
plt.axis('off')

# Compute and display the histogram using NumPy and matplotlib
hist, bins = np.histogram(I.flatten(), bins=256, range=[0, 256])

# counts = hist

# Query 60th bin value  # Bin Value Retrieval
bin_value_60 = hist[60]
print("60th bin value:", bin_value_60)

plt.subplot(1, 2, 2)
plt.plot(hist, color='b')
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.show()




#       Histograms for threshold selections

# Read the image
image = cv2.imread('./standard_test_images/pirate.tif', cv2.IMREAD_GRAYSCALE)

# Get the threshold value using Otsu's method
_, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the binary image
cv2.imshow('Thresholded Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



#       Adaptive thresholding

# Read in the image
I = cv2.imread('./standard_test_images/livingroom.tif', cv2.IMREAD_GRAYSCALE)

# Create mean image using 2D convolution (filter2D in OpenCV)
kernel = np.ones((15, 15), np.float32) / 225
Im = cv2.filter2D(I, -1, kernel, borderType=cv2.BORDER_REPLICATE)

# Subtract mean image from the original (with a constant C = 20)
constant_C = 20
It = I - (Im + constant_C)

# Threshold the result using adaptive thresholding
Ibw = cv2.adaptiveThreshold(It, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Display the original and thresholded images using matplotlib
plt.subplot(1, 2, 1)
plt.imshow(I, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(Ibw, cmap='gray')
plt.title('Thresholded Image')
plt.axis('off')

plt.show()



#       Contrasts Streching

# Read the image
image = cv2.imread('./standard_test_images/pirate.tif', cv2.IMREAD_GRAYSCALE)

# Increase contrast level (adjust the contrast stretching parameters here)
c_min, c_max = np.percentile(image, [0.1, 99.9])  # Adjust percentiles here for contrast control
Ics = cv2.normalize(image, None, alpha=c_min, beta=c_max, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Display the images
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(Ics, cmap='gray')
plt.title('Increased Contrast Image')
plt.axis('off')

plt.show()

# Display input histogram
plt.subplot(1, 2, 1)
plt.hist(image.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
plt.title('Input Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# Display output histogram
plt.subplot(1, 2, 2)
plt.hist(Ics.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
plt.title('Output Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.show()



# Histogram Equilization

# Step 1: Read the image using OpenCV
image_path = './standard_test_images/walkbridge.tif'
I = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 2: Perform histogram equalization using OpenCV
Ieq = cv2.equalizeHist(I)

# Step 3: Display the original and equalized images along with their histograms
plt.figure(figsize=(10, 8))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(I, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Equalized Image
plt.subplot(2, 2, 2)
plt.imshow(Ieq, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

# Histogram of Original Image
plt.subplot(2, 2, 3)
plt.hist(I.ravel(), bins=256, range=[0, 256], color='r')
plt.title('Histogram of Original Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# Histogram of Equalized Image
plt.subplot(2, 2, 4)
plt.hist(Ieq.ravel(), bins=256, range=[0, 256], color='r')
plt.title('Histogram of Equalized Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()




#       Histogram Matching

# Load the input image
I = cv2.imread('./standard_test_images/jetplane.tif', cv2.IMREAD_GRAYSCALE)

# Define the desired output histogram as a ramp-like pdf from 0 to 255
pz = np.arange(256)

# Perform histogram matching
Im = cv2.equalizeHist(I, pz)

# Display the images and histograms using matplotlib
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(I, cmap='gray')
plt.title('Input Image')

plt.subplot(2, 3, 2)
plt.imshow(Im, cmap='gray')
plt.title('Result after Histogram Matching')

plt.subplot(2, 3, 3)
plt.plot(pz)
plt.title('Desired Distribution')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(2, 3, 4)
plt.hist(I.ravel(), bins=256, range=(0, 256), density=True)
plt.title('Histogram of Input Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(2, 3, 5)
plt.hist(Im.ravel(), bins=256, range=(0, 256), density=True)
plt.title('Histogram of Result Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.show()





#       Histogram Operation On Color Images

# Read the image in BGR format (default format for OpenCV)
img = cv2.imread('./standard_test_images/mandril_color.tif')

# Convert BGR image to HSV color space
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Split the HSV image into separate channels
h, s, v = cv2.split(hsv_img)

# Perform histogram equalization on the V channel (the 3rd channel in HSV)
v_eq = cv2.equalizeHist(v)

# Replace the original V channel with the equalized V channel
hsv_img = cv2.merge([h, s, v_eq])

# Convert the HSV image back to BGR color space
out_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

# Plot the original and output images side by side
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
plt.title("Histogram Equalized Image")
plt.axis('off')

plt.show()
