import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read in test card image
A = cv2.imread('./standard_test_images/cameraman.tif', cv2.IMREAD_GRAYSCALE)
FA = np.fft.fft2(A)
FA = np.fft.fftshift(FA)

# Define PSF
PSF = cv2.getGaussianKernel(A.shape[0], 6) @ cv2.getGaussianKernel(A.shape[1], 6).T

# Calculate corresponding OTF
OTF = np.fft.fft2(PSF)
OTF = np.fft.fftshift(OTF)

# Calculate filtered image
Afilt = np.fft.ifft2(OTF * FA)
Afilt = np.fft.fftshift(Afilt)

# Display results
plt.subplot(1, 4, 1)
plt.imshow(A, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(np.log(1 + PSF), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(np.log(1 + np.abs(OTF)), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(np.abs(Afilt), cmap='gray')
plt.axis('off')

plt.show()

# Calculate corresponding PSF
PSF = cv2.getGaussianKernel(A.shape[0], 6) @ cv2.getGaussianKernel(A.shape[1], 6).T

# Calculate corresponding OTF
OTF = np.fft.fft2(PSF)
OTF = np.fft.fftshift(OTF)

# Calculate filtered image
Afilt = np.fft.ifft2(OTF * FA)
Afilt = np.fft.fftshift(Afilt)

# Display results
plt.subplot(1, 4, 1)
plt.imshow(A, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(np.log(1 + PSF), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(np.log(1 + np.abs(OTF)), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(np.abs(Afilt), cmap='gray')
plt.axis('off')

plt.show()