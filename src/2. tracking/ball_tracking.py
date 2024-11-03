from scipy.spatial import distance as dist
import numpy as np
from collections import deque
import cv2
import os

class BallTracker:
    def __init__(self, max_disappeared=3):
        # Initialize OpenCV's Kalman Filter
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Additional attributes
        self.max_disappeared = max_disappeared
        self.positions = deque(maxlen=2)
        self.disappeared = 0
        self.initialized = False

    def initialize_filter(self, position):
        # Set the initial state [x, y, dx, dy] where (x, y) is the position
        # and (dx, dy) is the velocity
        self.kf.statePre = np.array([[position[0]], [position[1]], [0], [0]], dtype=np.float32)
        self.initialized = True

    def update(self, position):
        if position is not None:
            if not self.initialized:
                self.initialize_filter(position)
            
            # Perform a measurement update with the detected position
            measurement = np.array([[np.float32(position[0])], [np.float32(position[1])]])
            self.kf.correct(measurement)
            self.positions.append(position)
            self.disappeared = 0

        else:
            # Increment disappeared counter if no position is detected
            self.disappeared += 1
            
            # Perform a prediction step if we’re missing detections
            if self.disappeared > self.max_disappeared:
                predicted = self.kf.predict()
                predicted_position = (int(predicted[0]), int(predicted[1]))
                return predicted_position

        # Predict the next position if the ball disappears in the next frame
        predicted = self.kf.predict()
        predicted_position = (int(predicted[0]), int(predicted[1]))
        return predicted_position

def test_yolo_with_tracking(model="modelnum", conf=0.2, iou=0.5, max_det=1, video_path=None, output_path=None, hue_shift=0, scale=3.5):
    if "ultralytics" not in os.popen("pip freeze").read():
        os.system('pip install ultralytics')
    from ultralytics import YOLO

    yolo_model = YOLO(f'../../modelsAndLogs/{model}/weights/best.pt')
    video = cv2.VideoCapture(video_path)
    tracker = BallTracker()
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        results = yolo_model.predict(source=frame, conf=conf, iou=iou, max_det=max_det, save=False)
        detections = results[0].boxes.data.cpu().numpy()

        if len(detections) > 0:
            x1, y1, x2, y2, confidence, _ = detections[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Ensure coordinates are integers
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            ball_width = int((x2 - x1) * scale)
            ball_height = int((y2 - y1) * scale)
            ball_position = (center_x, center_y)
            
            # Extract the ball region and adjust hue
            ball_region = frame[y1:y2, x1:x2]
            hsv_ball = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)
            hsv_ball[:, :, 0] = (hsv_ball[:, :, 0] + hue_shift) % 180  # Adjust hue channel
            color_adjusted_ball = cv2.cvtColor(hsv_ball, cv2.COLOR_HSV2BGR)
            
            # Resize (enlarge) the detected ball region
            enlarged_ball = cv2.resize(color_adjusted_ball, (ball_width, ball_height), interpolation=cv2.INTER_LINEAR)
            
            # Calculate the new coordinates to overlay the enlarged ball
            top_left_x = max(center_x - ball_width // 2, 0)
            top_left_y = max(center_y - ball_height // 2, 0)
            bottom_right_x = min(top_left_x + ball_width, frame.shape[1])
            bottom_right_y = min(top_left_y + ball_height, frame.shape[0])
            
            # Overlay the enlarged ball region onto the frame
            frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = enlarged_ball[:bottom_right_y-top_left_y, :bottom_right_x-top_left_x]
        else:
            ball_position = None
        
        tracked_position = tracker.update(ball_position)
        
        # Draw the tracked position as a circle for reference if needed
        if tracked_position:
            cv2.circle(frame, tracked_position, 1, (0, 0, 255), 2)  # Small circle at tracked position
        
        out.write(frame)
    
    video.release()
    out.release()

def get_data_path():
    current_dir = os.getcwd()
    if '/content' in current_dir:
        return '/content/COS-791---Object-tracking/data/raw/videos'
    else:
        return os.path.abspath(os.path.join(current_dir, '../../data/raw/videos'))
    
def get_output_path():
    current_dir = os.getcwd()
    if '/content' in current_dir:
        return '/content/COS-791---Object-tracking/data/trackingAndModify'
    else:
        return os.path.abspath(os.path.join(current_dir, '../../data/trackingAndModify'))

# main function to query user for video number and parameters
def main():
    video_number = int(input("Enter the video number (0 or 1): "))
    hue_shift = float(input("Enter the hue shift value (default=0): ") or 0)
    scale = float(input("Enter the scale value (default=3.5): ") or 3.5)
    model = input("Enter the model name (default=modelnum1): ") or "modelnum1"

    # output path is in data/trackingAndModify folder
    output_path = os.path.join(get_output_path(), f'Hockey{video_number}_tracked_modified.mp4')
    video_path = os.path.join(get_data_path(), f'Hockey{video_number}.mp4')
    
    test_yolo_with_tracking(model=model, conf=0.3, iou=0.5, max_det=1, video_path=video_path, output_path=output_path, hue_shift=hue_shift, scale=scale)

main()