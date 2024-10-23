import os
import subprocess
import yaml

def train_yolo(yolo_model="yolo11n.pt", epochs=100, patience=100, imgsz=640, batch=16):
    """
    Function to automate YOLOv11 model training with adapted paths according to the project structure.

    Args:
    yolo_model (str): YOLO model name.
    epochs (int): Number of epochs.
    patience (int): Patience.
    imgsz (int): Image size.
    batch (int): Batch size.
    """
    # Installing ultralytics package if not already installed
    if "ultralytics" not in os.popen("pip freeze").read():
        os.system('pip install ultralytics')

    # Update_yaml_paths()
    update_yaml_paths()
    # Get the current working directory
    current_dir = os.getcwd()
    # Use raw string to handle spaces and backslashes correctly
    current_dir = r"{}".format(current_dir)
    
    # Construct absolute paths
    data_path = os.path.join(current_dir, r'..\..\data\raw\pictures\data.yaml')
    project_path = os.path.join(current_dir, r'..\..\modelsAndLogs')
    
    # YOLO training command
    train_command = (
        f'yolo task=detect mode=train '
        f'model={yolo_model} '  
        f'data="{data_path}" '  # Wrapped in quotes
        f'epochs={epochs} patience={patience} imgsz={imgsz} batch={batch} '
        f'project="{project_path}" '  # Wrapped in quotes
        f'name=modelnum save=True'
    )

    # Print the training command
    print(train_command)
    
    # Execute the training command
    os.system(train_command)

def test_yolo(model="model",conf=0.2, iou=0.5, max_det=1,file_type="images", video_number=-1):
    """
    Function to automate YOLOv11 model testing with adapted paths according to the project structure.

    Args:
    model (str): Model name.
    conf (float): Confidence threshold.
    iou (float): Intersection over Union threshold.
    max_det (int): Maximum detections.
    """
    # Installing ultralytics package if not already installed
    if "ultralytics" not in os.popen("pip freeze").read():
        os.system('pip install ultralytics')

    if file_type == "images":
        dir = "pictures/test/images"
        save_dir = "pictures"
        folder_name = "image_predictions"
    else:
        dir = "videos/"
        save_dir = "videos"
        folder_name = "ball_prediction" if video_number == 1 else "puck_prediction"

    video_name = f"/Hockey{video_number}.mp4" if video_number != -1 else ""

    # YOLO testing command
    test_command = (
        f"yolo task=detect mode=predict "
        f"model=../../modelsAndLogs/{model}/weights/best.pt "  
        f"source=../../data/raw/{dir}{video_name} "  
        f"conf={conf} iou={iou} max_det={max_det} "
        f"project=../../data/predictions/{save_dir} name={folder_name} save=True"
    )

    # Print the testing command
    print(test_command)

    # Execute the testing command
    os.system(test_command)

def update_yaml_paths():
    """
    Function to update the paths in the data.yaml file according to the project structure.
    """
    # Get the current working directory
    current_dir = os.getcwd()
    # Use raw string to handle spaces and backslashes correctly
    current_dir = r"{}".format(current_dir)
    
    # Construct absolute paths
    train_path = os.path.join(current_dir, r'..\..\data\raw\pictures\train')
    val_path = os.path.join(current_dir, r'..\..\data\raw\pictures\valid')
    test_path = os.path.join(current_dir, r'..\..\data\raw\pictures\test')
    
    # Load the data.yaml file
    with open(r'..\..\data\raw\pictures\data.yaml') as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        if line.startswith("train:"):
            updated_lines.append(f"train: '{train_path}'\n")
        elif line.startswith("val:"):
            updated_lines.append(f"val: '{val_path}'\n")
        elif line.startswith("test:"):
            updated_lines.append(f"test: '{test_path}'\n")
        else:
            updated_lines.append(line)

    # Save the updated data.yaml file
    with open(r'..\..\data\raw\pictures\data.yaml', 'w') as file:
        file.writelines(updated_lines)

def revert_yaml_paths():
    """
    Function to revert the paths in the data.yaml file to the original paths.
    """
    # Load the data.yaml file
    with open(r'..\..\data\raw\pictures\data.yaml') as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        if line.startswith("train:"):
            updated_lines.append("train: ../train\n")
        elif line.startswith("val:"):
            updated_lines.append("val: ../valid\n")
        elif line.startswith("test:"):
            updated_lines.append("test: ../test\n")
        else:
            updated_lines.append(line)

    # Save the updated data.yaml file
    with open(r'..\..\data\raw\pictures\data.yaml', 'w') as file:
        file.writelines(updated_lines)
    

if __name__ == "__main__":
    # ask if the user wants to train a new model or test an existing one
    answer1 = input("Do you want to train a new model? (y/n): ")
    if answer1.lower() == "y":
        yolo_model = input("Enter a pre-trained model name (i.e yolo11n.pt): ")
        epochs = int(input("Enter the number of epochs: "))
        patience = int(input("Enter the patience: "))
        imgsz = int(input("Enter the image size: "))
        batch = int(input("Enter the batch size: "))
        train_yolo(yolo_model, epochs, patience, imgsz, batch)
        revert_yaml_paths()
        

    answer2 = input("Do you want to test an existing model? (y/n): ")
    if answer2.lower() == "y":
        model = input("Enter the model name: ")
        conf = float(input("Enter the confidence threshold: "))
        iou = float(input("Enter the intersection over union threshold: "))
        max_det = int(input("Enter the maximum detections per frame: "))
        # ask if the user wants to test on images or videos, images is boolean
        file_type = input("Do you want to test on images or videos? (images/videos): ")
        if file_type.lower() == "videos":
            test_yolo(model, conf, iou, max_det, file_type="videos", video_number=0)
            test_yolo(model, conf, iou, max_det, file_type="videos", video_number=1)
        else:
            test_yolo(model, conf, iou, max_det)
        revert_yaml_paths()
