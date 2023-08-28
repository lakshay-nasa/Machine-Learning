import cv2
import dlib

hog_detector = dlib.get_frontal_face_detector()

# Read the image for detection.
image = cv2.imread('./images/GP.png')  # Corrected the image filename

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform human detection using dlib's HOG detector.
rectangles = hog_detector(gray_image)

# Draw rectangles around the detected humans.
for rect in rectangles:
    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with detected humans using OpenCV.
cv2.imshow('Human Detection using dlib', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
