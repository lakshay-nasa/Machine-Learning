import cv2
import matplotlib.pyplot as plt

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

image_path = "./images/f_bg.jpg"  # Replace with the path to your image file
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cropped_face = max(face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)), key=lambda f: f[2] * f[3], default=None)

# Display the input and the output side by side using Matplotlib
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Display the input image
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Input Image")
axes[0].axis('off')

# Display the cropped face
if cropped_face is not None:
    (x, y, w, h) = cropped_face
    axes[1].imshow(cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2RGB))
    axes[1].set_title("Face")
    axes[1].axis('off')

plt.show()
