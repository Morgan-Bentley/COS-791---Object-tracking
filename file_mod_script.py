import os
import random

current_dir = os.getcwd()
# Define paths for train, validation, and test images and labels directories to move the images and labels to
train_images_dir = current_dir + r"\data\raw\pictures\train\images"
train_labels_dir = current_dir + r"\data\raw\pictures\train\labels"
valid_images_dir = current_dir + r"\data\raw\pictures\valid\images"
valid_labels_dir = current_dir + r"\data\raw\pictures\valid\labels"
test_images_dir = current_dir + r"\data\raw\pictures\test\images"
test_labels_dir = current_dir + r"\data\raw\pictures\test\labels"


directories = [
    (train_images_dir, train_labels_dir),
    (valid_images_dir, valid_labels_dir),
    (test_images_dir, test_labels_dir)
]

# Process each directory
for images_dir, labels_dir in directories:
    # Collect image and label file pairs
    file_pairs = [(f, f.replace('.jpg', '.txt')) for f in os.listdir(images_dir) if f.endswith('.jpg')]

    # Shuffle the file pairs
    random.shuffle(file_pairs)

    # Rename files with randomized order in place
    for i, (image_file, label_file) in enumerate(file_pairs):
        # Define new names with zero-padded numbers
        image_name = f"image_{i+1:04d}.jpg"
        label_name = f"image_{i+1:04d}.txt"

        # Paths for old and new filenames
        old_image_path = os.path.join(images_dir, image_file)
        image_path = os.path.join(images_dir, image_name)
        old_label_path = os.path.join(labels_dir, label_file)
        label_path = os.path.join(labels_dir, label_name)

        # Rename the image
        os.rename(old_image_path, image_path)
        print(f"Renamed image: {old_image_path} to {image_path}")

        # Rename the corresponding label
        if os.path.exists(old_label_path):
            os.rename(old_label_path, label_path)
            print(f"Renamed label: {old_label_path} to {label_path}")

print("All images and labels have been successfully renamed in random order.")