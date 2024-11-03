import os
import subprocess
import yaml
import torch
import optuna
import pandas as pd

torch.cuda.empty_cache()

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

def train_yolo(model_num, yolo_model="yolo11n.pt", epochs=100, patience=100, imgsz=640, batch=16, momentum=0.937, lr=0.01):
    """
    Function to automate YOLOv11 model training with adapted paths according to the project structure.
    """
    install_ultralytics()
    update_yaml_paths()

    data_path = get_data_path()
    project_path = get_project_path()
    model_dir = f"modelnum{model_num}"
    
    train_command = (
        f'yolo task=detect mode=train '
        f'model={yolo_model} '
        f'data="{data_path}" '  # Wrapped in quotes
        f'epochs={epochs} patience={patience} imgsz={imgsz} batch={batch} '
        f'project="{project_path}" '  # Wrapped in quotes
        f'name={model_dir} save=True momentum={momentum} lr0={lr} '
    )

    print("Executing training command:", train_command)
    os.system(train_command)

    metrics_path = os.path.join(project_path, model_dir, "results.csv")
    if os.path.exists(metrics_path):
        try:
            metrics_df = pd.read_csv(metrics_path, sep=";")
            mAP_50 = metrics_df["metrics/mAP50(B)"].max()
            return mAP_50
        except Exception as e:
            print(f"Error reading metrics: {e}")
            return 0
    return 0  # Default value if metric file not found

def calculate_iou(box1, box2):
    xA, yA, xB, yB = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    return inter_area / float(box1_area + box2_area - inter_area)

def test_yolo(model="model", conf=0.2, iou=0.5, max_det=1, file_type="images", video_number=-1):
    """
    Function to automate YOLOv11 model testing with adapted paths according to the project structure.

    Args:
    model (str): Model name.
    conf (float): Confidence threshold.
    iou (float): Intersection over Union threshold.
    max_det (int): Maximum detections.
    """
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

    print(test_command)
    os.system(test_command)

def update_yaml_paths():
    current_dir = os.getcwd()
    train_path = os.path.abspath(os.path.join(current_dir, '../../data/raw/pictures/train'))
    val_path = os.path.abspath(os.path.join(current_dir, '../../data/raw/pictures/valid'))
    test_path = os.path.abspath(os.path.join(current_dir, '../../data/raw/pictures/test'))
    yaml_path = os.path.abspath(os.path.join(current_dir, '../../data/raw/pictures/data.yaml'))
    
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"{yaml_path} not found.")

    with open(yaml_path, 'r') as file:
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

    with open(yaml_path, 'w') as file:
        file.writelines(updated_lines)

def revert_yaml_paths():
    current_dir = os.getcwd()
    yaml_path = os.path.abspath(os.path.join(current_dir, '../../data/raw/pictures/data.yaml'))

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"{yaml_path} not found.")

    with open(yaml_path, 'r') as file:
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

    with open(yaml_path, 'w') as file:
        file.writelines(updated_lines)

# Global counter for model directory
model_counter = 1

def objective(trial):
    global model_counter
    model_counter += 1

    yolo_model = trial.suggest_categorical("yolo_model", ["yolo11n.pt"])
    epochs = trial.suggest_int("epochs", 50, 300)
    patience = trial.suggest_int("patience", 10, 20)
    
    imgsz = trial.suggest_int("imgsz", 1920, 2048, step=160)

    batch = trial.suggest_float("batch", 0.8, 0.95)

    momentum = trial.suggest_float("momentum", 0.85, 0.95)  

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    mAP_50 = train_yolo(model_counter, yolo_model, epochs, patience, imgsz, batch, momentum, lr)
    revert_yaml_paths()

    return mAP_50

if __name__ == "__main__":
    answer1 = input("Enter (y) for testing existing models, (n) to train new models:")
    if answer1.lower() == "y":
        model = input("Enter the model name: ")
        conf = float(input("Enter the confidence threshold: "))
        iou = float(input("Enter the intersection over union threshold: "))
        max_det = int(input("Enter the maximum detections per frame: "))
        file_type = input("Do you want to test on images or videos? (images/videos): ")
        if file_type.lower() == "videos":
            test_yolo(model, conf, iou, max_det, file_type="videos", video_number=0)
            test_yolo(model, conf, iou, max_det, file_type="videos", video_number=1)
        else:
            test_yolo(model, conf, iou, max_det)
        revert_yaml_paths()
    else:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=3) 

        print("Best trial:")
        trial = study.best_trial

        print(f"  Value: {trial.value}")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
