# COS-791---Object-tracking

This project automates the training and testing of a YOLOv11 model for ball detection in hockey footage. The scripts allow you to train a new model, test an existing model, and update file paths in the data.yaml configuration file to match the project structure.

Table of Contents
1. Project Structure
2. Setting Up the Environment
3. Training a New Model
4. Testing an Existing Model

## Project Structure
The directory structure for this project is as follows:
```bash
/project_root/
  ├── data/
  │   ├── predictions/
  │   │   ├── pictures/             # Pictures with objectprediction
  │   │   └── videos/               # Videos with object prediction
  │   └── raw/
  │       ├── pictures/
  │       │   ├── train/            # Training images
  │       │   ├── valid/            # Validation images
  │       │   ├── test/             # Test images
  │       │   └── data.yaml         # YOLOv11 configuration file
  │       └── videos/               # Videos for prediction
  ├── modelsAndLogs/                # Directory for model and training results
  ├── report/                       # Final detailed report on the whole project process
  └── src/
      ├── 1. detection/
      │      └──ball_detection.py   # script for training, testing and performing ball detection
      ├── 2. tracking/
      └── 3. augmentation/
```

## Setting Up the Environment
Ensure that Python 3.12 is installed, along with the ultralytics package for YOLOv11. If not installed, run:
```bash
pip install ultralytics
```

## Training a New Model
To train a YOLOv11 model, use the provided Python script. Run:
python ball_detection.py

The script will prompt you for the following:

Do you want to train a new model? (y/n):
Type 'y' to initiate the training process.

Enter the model name (e.g., yolo11n.pt):
Provide the name of the pre-trained YOLO model you want to use.

Enter the number of epochs, patience, image size, and batch size:
Input the desired training parameters.

Once training is complete, the model will be saved in the modelsAndLogs/ directory along with logs of the training process.

## Testing an Existing Model
To test a trained YOLO model on images or videos, follow these steps in the script:

Do you want to test an existing model? (y/n):
Type 'y' to start testing.

Enter the model name, confidence threshold, IoU threshold, and max detections per frame:
Provide the necessary testing parameters.

Do you want to test on images or videos? (images/videos):
Specify whether to test on images or videos. If videos, you can test on multiple video files.
