import cv2
import numpy as np
import matplotlib.pyplot as plt
import mrcnn.config
import mrcnn.model as modellib
from mrcnn import visualize

class InferenceConfig(mrcnn.config.Config):
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81  # COCO dataset has 80 classes + 1 background class

def separate_human_background(image_path):
    # Step 1: Load the pre-trained Mask R-CNN model
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir="logs")
    model.load_weights("mask_rcnn_coco.h5", by_name=True)

    # Step 2: Read the input image
    image = cv2.imread(image_path)

    # Step 3: Run the image through the Mask R-CNN model for instance segmentation
    results = model.detect([image], verbose=0)

    # Step 4: Get the masks, bounding boxes, and class IDs of detected objects
    r = results[0]
    masks = r['masks']
    class_ids = r['class_ids']

    # Step 5: Create a binary mask for the human (class ID 1 corresponds to person in COCO dataset)
    human_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for i in range(masks.shape[2]):
        if class_ids[i] == 1:
            human_mask = np.maximum(human_mask, masks[:, :, i].astype(np.uint8) * 255)

    # Step 6: Separate the human and background using the binary mask
    foreground = cv2.bitwise_and(image, image, mask=human_mask)
    background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(human_mask))

    return foreground, background

# Replace 'input_image.jpg' with the path to your input image.
input_image_path = './images/f_bg.jpg'

# Separate the human and background
foreground_image, background_image = separate_human_background(input_image_path)

# Display the original and output images side by side using Matplotlib
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Load and display the original image
original_image = cv2.imread(input_image_path)
axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')

# Display the human image (foreground)
axes[1].imshow(cv2.cvtColor(foreground_image, cv2.COLOR_BGR2RGB))
axes[1].set_title('Human (Foreground)')
axes[1].axis('off')

# Display the background image
axes[2].imshow(cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB))
axes[2].set_title('Background')
axes[2].axis('off')

plt.show()
