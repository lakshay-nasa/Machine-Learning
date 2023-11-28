import cv2
import numpy as np
import matplotlib.pyplot as plt

def hysteresis_thresholding(image, low_threshold, high_threshold):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges

def apply_hysteresis(image, low_threshold, high_threshold):
    edges = hysteresis_thresholding(image, low_threshold, high_threshold)
    visited = np.zeros_like(edges)
    
    def dfs(start_i, start_j):
        stack = [(start_i, start_j)]
        while stack:
            i, j = stack.pop()
            if i < 0 or i >= edges.shape[0] or j < 0 or j >= edges.shape[1]:
                continue
            if visited[i, j] == 1 or edges[i, j] == 0:
                continue
            visited[i, j] = 1
            stack.append((i + 1, j))
            stack.append((i - 1, j))
            stack.append((i, j + 1))
            stack.append((i, j - 1))
    
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i, j] == 255 and visited[i, j] == 0:
                dfs(i, j)
    
    edges = edges * visited
    return edges

image = cv2.imread('./images/nature.jpg', cv2.IMREAD_GRAYSCALE)

low_threshold = 50
high_threshold = 150
edges = apply_hysteresis(image, low_threshold, high_threshold)

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection with Hysteresis Thresholding')
plt.axis('off')

plt.show()
