import cv2
import albumentations as A
import os
import glob
import random

# Define the augmentation pipeline
transform = A.Compose([
    # Adjust lighting conditions
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Randomly change brightness and contrast
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),  # Adjust hue, saturation, value

    # Introduce perspective and angle variations
    A.Perspective(scale=(0.05, 0.1), p=0.3),  # Simulate different camera angles
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),  # Rotate, scale, and shift image

    # Add color variations to handle different ball colors
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.4),  # Randomly shift RGB channels
    A.ChannelShuffle(p=0.2),  # Shuffle channels to introduce color variations

    # Introduce slight blur to simulate motion
    A.MotionBlur(blur_limit=5, p=0.3),  # Motion blur
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),  # Gaussian blur for further variation

    # Randomly apply some noise to simulate real-world camera noise
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.2), p=0.3),  # Add noise to mimic low-light conditions
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.25, label_fields=['category_ids']))

def load_image_and_label(image_path, label_path):
    image = cv2.imread(image_path)
    with open(label_path, 'r') as f:
        label_data = f.readline().strip().split()
        category_id = int(label_data[0])  # Class label
        bbox = list(map(float, label_data[1:]))  # YOLO format: [x_center, y_center, width, height]
    return image, category_id, bbox

def save_augmented_image_and_label(aug_image, aug_bbox, category_id, save_path, save_label_path):
    cv2.imwrite(save_path, aug_image)
    with open(save_label_path, 'w') as f:
        # Write in YOLO format
        f.write(f"{category_id} " + " ".join(map(str, aug_bbox)) + "\n")

def augment_and_save(image_path, label_path, output_dir):
    # Define subdirectories for images and labels
    image_output_dir = os.path.join(output_dir, "images")
    label_output_dir = os.path.join(output_dir, "labels")

    # Ensure subdirectories exist
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    image, category_id, bbox = load_image_and_label(image_path, label_path)
    
    aug_count = random.randint(1, 4)  # Randomly choose the number of augmentations to apply
    for i in range(aug_count):
        augmented = transform(image=image, bboxes=[bbox], category_ids=[category_id])
        # Check if augmented contains any bboxes
        if not augmented['bboxes']:
            print(f"Skipped augmentation for {image_path} as no bounding box remains.")
            continue  # Skip this augmentation if no bbox is present

        aug_image = augmented['image']
        aug_bbox = augmented['bboxes'][0]
        # Define paths for augmented images and labels
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # save images in output_dir/pictures and labels in output_dir/labels
        save_path = os.path.join(image_output_dir, f"{base_name}_aug_{i}.jpg")
        save_label_path = os.path.join(label_output_dir, f"{base_name}_aug_{i}.txt")

        # Save augmented image and label
        save_augmented_image_and_label(aug_image, aug_bbox, category_id, save_path, save_label_path)

def get_train_path():
    """Determine the absolute path for project logs and models based on the environment."""
    current_dir = os.getcwd()
    if '/content' in current_dir:
        return '/content/COS-791---Object-tracking/data/raw/pictures/train'
    else:
        return os.path.abspath(os.path.join(current_dir, '../../data/raw/pictures/train'))
    
def make_aug_path():
    """Creates a directory for augmented images inside the data folder and returns the path."""
    current_dir = os.getcwd()
    if '/content' in current_dir:
        path = '/content/COS-791---Object-tracking/data/raw/pictures/augmented'
    else:
        path = os.path.abspath(os.path.join(current_dir, '../../data/raw/pictures/augmented'))
    
    os.makedirs(path, exist_ok=True)
    return path  # Return the path as a string

# Example usage
input_dir = os.path.join(get_train_path(), 'images')
label_dir = os.path.join(get_train_path(), 'labels')
output_dir = make_aug_path()

os.makedirs(output_dir, exist_ok=True)
random.seed(42)
for image_path in glob.glob(os.path.join(input_dir, "*.jpg")):
    label_path = os.path.join(label_dir, os.path.basename(image_path).replace('.jpg', '.txt'))
    augment_and_save(image_path, label_path, output_dir)