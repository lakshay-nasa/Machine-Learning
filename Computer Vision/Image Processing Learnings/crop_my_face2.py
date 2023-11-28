import cv2
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt

def detect_and_display_faces(image_path):
    # Load the MTCNN face detection model
    mtcnn = MTCNN()

    # Read the input image
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    boxes, _ = mtcnn.detect(rgb_image)

    # Crop and store the detected faces
    cropped_faces = []
    if boxes is not None:
        for box in boxes:
            x, y, w, h = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])
            # Crop the face region
            face_roi = image[y:y+h, x:x+w]
            cropped_faces.append(face_roi)

    # Display all the cropped faces using Matplotlib
    num_faces = len(cropped_faces)
    if num_faces == 0:
        print("No faces were detected in the image.")
    else:
        if num_faces == 1:
            plt.imshow(cv2.cvtColor(cropped_faces[0], cv2.COLOR_BGR2RGB))
            plt.axis('off')
        else:
            fig, axes = plt.subplots(1, num_faces, figsize=(10, 5))
            for i, face in enumerate(cropped_faces):
                axes[i].imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                axes[i].axis('off')

        plt.show()

if __name__ == "__main__":
    image_path = "./images/f_bg.jpg"  # Replace with the path to your image file
    detect_and_display_faces(image_path)
