import cv2
import numpy as np
import os
from imgaug import augmenters as iaa
import imageio

# Path to the dataset
dataset_path = "E:\SCC\dataset"
folders = ['101', '102']  # Folder names for labeling

# Augmented output folder
augmented_output_path = "E:\SCC\dataset"

# Create augmented output directories if they don't exist
if not os.path.exists(augmented_output_path):
    os.makedirs(augmented_output_path)

for folder in folders:
    input_folder_path = os.path.join(dataset_path, folder)
    output_folder_path = os.path.join(augmented_output_path, folder)

    # Create folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Define the augmentation sequence
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # Horizontal flip 50% of the images
        iaa.Affine(rotate=(-30, 30)),  # Rotate between -30 and 30 degrees
        iaa.Affine(scale=(0.8, 1.2)),  # Scale images between 80% and 120% of original size
        iaa.Multiply((0.8, 1.2)),  # Change brightness
        iaa.GaussianBlur(sigma=(0, 1.0))  # Apply Gaussian blur
    ])

    # Load and augment images
    for img_name in os.listdir(input_folder_path):
        img_path = os.path.join(input_folder_path, img_name)

        # Read the image
        image = cv2.imread(img_path)

        # Convert BGR to RGB for imgaug
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate multiple augmented images
        for i in range(5):  # Generate 5 augmented images per original
            augmented_image = seq(image=image_rgb)

            # Convert back to BGR for saving with OpenCV
            augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

            # Save the augmented image
            output_img_name = f"{img_name.split('.')[0]}_aug_{i}.jpg"
            output_img_path = os.path.join(output_folder_path, output_img_name)
            cv2.imwrite(output_img_path, augmented_image_bgr)

print("Augmentation complete!")
