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
        f"yolo task=detect mode=train "
        f"model={yolo_model} "  
        f"data={data_path} "  
        f"epochs={epochs} patience={patience} imgsz={imgsz} batch={batch} "
        f"project={project_path} " 
        f"name=model save=True"
    )
    
    print("Training Command:", train_command)  # Print the training command

    # Execute the training command
    result = subprocess.run(train_command, shell=True, capture_output=True, text=True)
    
    # Print command output and errors
    print("Command Output:", result.stdout)
    print("Command Error:", result.stderr)

def test_yolo(model="model",conf=0.2, iou=0.5, max_det=1):
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

    # YOLO testing command
    test_command = (
        f"yolo task=detect mode=test "
        f"model=../../modelsAndLogs/{model}/weights/best.pt "  
        f"source=../../data/raw/pictures/test/images "  
        f"conf={conf} iou={iou} max_det={max_det}"
    )

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
    val_path = os.path.join(current_dir, r'..\..\data\raw\pictures\val')
    test_path = os.path.join(current_dir, r'..\..\data\raw\pictures\test')
    
    # Load the data.yaml file
    with open(r'..\..\data\raw\pictures\data.yaml') as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        if line.startswith("train:"):
            updated_lines.append(f"train: {train_path}\n")
        elif line.startswith("val:"):
            updated_lines.append(f"val: {val_path}\n")
        elif line.startswith("test:"):
            updated_lines.append(f"test: {test_path}\n")
        else:
            updated_lines.append(line)

    # Save the updated data.yaml file
    with open(r'..\..\data\raw\pictures\data.yaml', 'w') as file:
        file.writelines(updated_lines)

    

if __name__ == "__main__":
    # ask if the user wants to train a new model or test an existing one
    train_yolo()
    #answer1 = input("Do you want to train a new model? (y/n): ")
    #if answer1.lower() == "y":
        #yolo_model = input("Enter the model name: ")
        #epochs = int(input("Enter the number of epochs: "))
        #patience = int(input("Enter the patience: "))
        #imgsz = int(input("Enter the image size: "))
        #batch = int(input("Enter the batch size: "))
        #train_yolo(yolo_model, epochs, patience, imgsz, batch)
        

    answer2 = input("Do you want to test an existing model? (y/n): ")
    if answer2.lower() == "y":
        model = input("Enter the model name: ")
        conf = float(input("Enter the confidence threshold: "))
        iou = float(input("Enter the intersection over union threshold: "))
        max_det = int(input("Enter the maximum detections: "))
        test_yolo(model, conf, iou, max_det)
