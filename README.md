# COS-791---Object-tracking

This project automates the training and testing of a YOLOv11 model for ball detection in hockey footage. The scripts allow you to train a new model, test an existing model, and update file paths in the data.yaml configuration file to match the project structure.

Table of Contents
1. Project Structure
2. Setting Up the Environment
3. Training a New Model
4. Testing an Existing Model
5. Ball Tracking and Video Processing with YOLO

## Project Structure
The directory structure for this project is as follows:
```bash
/project_root/
  ├── data/
  │   ├── predictions/
  │   │   ├── pictures/             # Pictures with objectprediction
  │   │   └── videos/               # Videos with object prediction
  │   ├── raw/
  │   │   ├── pictures/
  │   │   │   ├── train/            # Training images
  │   │   │   ├── valid/            # Validation images
  │   │   │   ├── test/             # Test images
  │   │   │   └── data.yaml         # YOLOv11 configuration file
  │   │   └── videos/               # Videos for prediction
  │   └── trackingAndModify/        # Videos with enlarged and color changed ball
  ├── modelsAndLogs/                # Directory for model and training results
  ├── report/                       # Final detailed report on the whole project process
  └── src/
      ├── 1. augmentation/
      │      └──augmentation.py   # script for augmenting training dataset to avoid overfitting
      ├── 2. detection/
      │      └──ball_detection.py   # script for training, testing and performing ball detection
      └── 3. tracking/
             └──ball_tracking.py   # script for tracking ball, enlarging it and changing its colour
      
```

## Setting Up the Environment

### Prerequisites:
Anaconda: Download and install Anaconda for managing dependencies and creating isolated environments.
NVIDIA GPU (optional): If you plan to use GPU acceleration, ensure you have an NVIDIA GPU and the compatible CUDA version installed.

**Create a New Environment:**<br>
Open Anaconda Prompt and create a new environment for YOLO:
```bash
conda create -n yolov11_env python=<python-version>
```
**Activate the environment:**
```bash
conda activate yolov11_env
```
**Install PyTorch:**<br>
Install the appropriate version of PyTorch for your setup:<br>
CPU-only:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
GPU (CUDA-enabled): <br>
Visit PyTorch’s Get Started page for the exact command matching your CUDA version. Example:
```bash
conda install pytorch torchvision torchaudio cudatoolkit=<cuda_version> -c pytorch
```
**Install YOLO with Ultralytics:**<br>
Install the ultralytics package:
```bash
pip install ultralytics
```
**Verify Installation:**<br>
Check that YOLO is installed correctly:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov11n.pt')"
```

## Training a New Model
To train a YOLOv11 model, use the provided Python script. <br>
### Run:
```bash
python ball_detection.py
```

### The script will prompt you for the following:

Do you want to train a new model? (y/n): <br>
Type 'y' to initiate the training process.

Enter the model name (e.g., yolo11n.pt): <br>
Provide the name of the pre-trained YOLO model you want to use.

Enter the number of epochs, patience, image size, and batch size: <br>
Input the desired training parameters.

Once training is complete, the model will be saved in the modelsAndLogs/ directory along with logs of the training process.

## Testing an Existing Model
To test a trained YOLO model on images or videos, follow these steps in the script: <br>

Do you want to test an existing model? (y/n): <br>
Type 'y' to start testing.

Enter the model name, confidence threshold, IoU threshold, and max detections per frame: <br>
Provide the necessary testing parameters.

Do you want to test on images or videos? (images/videos): <br>
Specify whether to test on images or videos. If videos, you can test on multiple video files.

## Ball Tracking and Video Processing with YOLO
In addition to training and testing, the script includes a feature to track and modify the appearance of the ball in videos using the BallTracker class. This class tracks the ball’s position across frames and predicts its location if it temporarily disappears. Additionally, you can modify the ball’s hue and enlarge it for enhanced visibility.

### Run Ball Tracking:
To process videos with ball tracking, run:

```bash
python ball_detection.py
```

### The script will prompt for these additional inputs:

Enter the video number (0 or 1):<br>
Select the video file (e.g., Hockey0.mp4 or Hockey1.mp4).

Enter the hue shift value (default=0):<br>
Adjust the ball’s color by shifting the hue.

Enter the scale value (default=3.5):<br>
Set the enlargement factor for the ball in the output video.

Enter the model name (default=modelnum1):<br>
Provide the name of the YOLO model to be used for detection.

The processed video, with tracked and modified ball detections, will be saved in the data/trackingAndModify/ directory.
