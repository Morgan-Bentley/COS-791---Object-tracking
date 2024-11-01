import os
import subprocess
import yaml
import torch
import cv2
#import optuna

torch.cuda.empty_cache()

# drop images without ball
def drop_images_without_ball():


def install_ultralytics():
    """Install ultralytics if not already installed."""
    if "ultralytics" not in subprocess.check_output("pip freeze", shell=True).decode():
        subprocess.run(["pip", "install", "ultralytics"])

def get_data_path():
    """Determine the absolute path for data.yaml based on the environment."""
    current_dir = os.getcwd()
    if '/content' in current_dir:  # Check if running on Colab
        return '/content/COS-791---Object-tracking/data/raw/pictures/data.yaml'
    else:
        return os.path.abspath(os.path.join(current_dir, '../../data/raw/pictures/data.yaml'))

def get_project_path():
    """Determine the absolute path for project logs and models based on the environment."""
    current_dir = os.getcwd()
    if '/content' in current_dir:
        return '/content/COS-791---Object-tracking/modelsAndLogs'
    else:
        return os.path.abspath(os.path.join(current_dir, '../../modelsAndLogs'))

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.
    """
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.8, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    # Train YOLO model with suggested hyperparameters
    train_yolo(epochs=100, patience=100, imgsz=640, batch=16, lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Evaluate the model
    # Return the evaluation metric (e.g. mAP) as the objective value



def train_yolo(yolo_model="yolo11n.pt", epochs=100, patience=100, imgsz=640, batch=16, momentum=0.937, lr=0.01):
    """
    Function to automate YOLOv11 model training with adapted paths according to the project structure.
    """
    install_ultralytics()
    update_yaml_paths()

    data_path = get_data_path()
    project_path = get_project_path()
    
    # YOLO training command with augmentation options
    train_command = (
        f'yolo task=detect mode=train '
        f'model={yolo_model} '
        f'data="{data_path}" '  # Wrapped in quotes
        f'epochs={epochs} patience={patience} imgsz={imgsz} batch={batch} '
        f'project="{project_path}" '  # Wrapped in quotes
        f'name=modelnum save=True momentum={momentum} lr0={lr} '
        f'augment=True hsv_h=0.2 hsv_s=0.8 hsv_v=0.5 degrees=25 scale=0.6 shear=10 erasing=0.5'
    )

    print("Executing training command:", train_command)
    os.system(train_command)

def calculate_iou(box1, box2):
    xA, yA, xB, yB = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    return inter_area / float(box1_area + box2_area - inter_area)

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
    # Train or test model based on user input
    answer1 = input("Do you want to train a new model? (y/n): ")
    if answer1.lower() == "y":
        yolo_model = input("Enter a pre-trained model name (i.e yolo11n.pt): ")
        epochs = int(input("Enter the number of epochs: "))
        patience = int(input("Enter the patience: "))
        imgsz = int(input("Enter the image size: "))
        batch = float(input("Enter the batch size: "))
        momentum = float(input("Enter the momentum (i.e 0.937): "))
        lr = float(input("Enter the learning rate (i.e 0.01): "))
        train_yolo(yolo_model, epochs, patience, imgsz, int(batch) if batch.is_integer() else batch, momentum, lr)
        revert_yaml_paths()

    answer2 = input("Do you want to test an existing model? (y/n): ")
    if answer2.lower() == "y":
        model = input("Enter the model name: ")
        conf = float(input("Enter the confidence threshold: "))
        iou = float(input("Enter the intersection over union threshold: "))
        max_det = int(input("Enter the maximum detections per frame: "))
        file_type = input("Do you want to test on images or videos? (images/videos): ")
        if file_type.lower() == "videos":
            test_yolo(model, conf, iou, max_det, file_type="videos", video_number=1)
            test_yolo(model, conf, iou, max_det, file_type="videos", video_number=2)
        else:
            test_yolo(model, conf, iou, max_det)
        revert_yaml_paths()
