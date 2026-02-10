# Online-Exam-Cheating-Detection

## Introduction
This notebook implements an automated proctoring system designed to monitor online exam environments for potential cheating behaviors. It utilizes computer vision techniques to detect gaze violations (when a student looks away from the screen), the presence of multiple individuals, and the use of unauthorized objects like cell phones or books.

## Setup
Before running the core logic, it's essential to set up the environment by downloading necessary models and installing required libraries.

### 1. Download YOLOv8n Model
The YOLOv8n model is used for object detection. It is downloaded from the Ultralytics GitHub repository.

```python
import os
import urllib.request

# Create the models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# URL of the YOLOv8n model file
url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
destination = "models/yolov8n.pt"

# Download the model
print("⬇️  Downloading YOLOv8n model...")
urllib.request.urlretrieve(url, destination)
print("✅ YOLOv8n model downloaded and saved to models/yolov8n.pt")
```

### 2. Download MediaPipe Blaze Face Short Range Model
This TFLite model is part of MediaPipe's face detection solution, optimized for short-range detection.

```bash
!wget -q -O blaze_face_short_range.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
```

### 3. Download MediaPipe Face Landmarker Model
This task file is used by MediaPipe for face landmark detection, which is crucial for head pose estimation.

```bash
!wget -q -O face_landmarker_v2_with_blendshapes.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### 4. Install MediaPipe Library
MediaPipe is a cross-platform framework for building multimodal applied machine learning pipelines.

```bash
!pip install mediapipe
```

### 5. Install Ultralytics Library
Ultralytics provides the YOLO models and related tools.

```bash
!pip install -U ultralytics
```

## Core Logic
The main script orchestrates the proctoring process, combining face landmark detection for head pose estimation and YOLO for object detection.

### 1. Import Libraries
Essential libraries such as `cv2` for OpenCV operations, `mediapipe` for face detection and landmarking, `numpy` for numerical operations, `ultralytics` for YOLO models, and `google.colab.patches` for displaying images in Colab are imported.

```python
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time
from google.colab.patches import cv2_imshow
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from IPython.display import clear_output
from google.colab import files
```

### 2. File Upload and Constants
The script prompts the user to upload a video file for analysis. Key constants like `MAX_YAW_DEG` and `MAX_PITCH_DEG` define the thresholds for head pose deviation considered a 'gaze violation'.

```python
uploaded = files.upload()

YOLO_MODEL_PATH = 'yolov8n.pt'

if uploaded:

    VIDEO_PATH = list(uploaded.keys())[0]
    print(f"File uploaded successfully: {VIDEO_PATH}")
else:
    print("No File Uploaded")


MAX_YAW_DEG = 35
MAX_PITCH_DEG = 25

yolo_model = YOLO(YOLO_MODEL_PATH)
away_count = 0
phone_detected_count = 0
unauthorized_person_detected_count = 0
calibrated_pitch, calibrated_yaw = 0, 0
detected_object_types = set()
```

### 3. Model Points for Pose Estimation
Defines a 3D model of a face used as reference for head pose estimation.

```python
model_points = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)
```

### 4. MediaPipe Face Landmarker Initialization
The Face Landmarker is initialized with the downloaded model, configured for image mode and to detect up to two faces.

```python
base_options_land = python.BaseOptions(model_asset_path='/content/face_landmarker_v2_with_blendshapes.task')
options_land = vision.FaceLandmarkerOptions(
    base_options=base_options_land,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=2
)
```

### 5. `estimate_head_pose` Function
This function takes face landmarks and image dimensions to calculate the head's pitch and yaw angles using `cv2.solvePnP`.

```python
def estimate_head_pose(landmarks, width, height):

    image_points = np.array([
        (landmarks[1].x * width, landmarks[1].y * height),
        (landmarks[152].x * width, landmarks[152].y * height),
        (landmarks[33].x * width, landmarks[33].y * height),
        (landmarks[263].x * width, landmarks[263].y * height),
        (landmarks[61].x * width, landmarks[61].y * height),
        (landmarks[291].x * width, landmarks[291].y * height)
    ], dtype=np.float64)

    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    if not success: return None

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles
```

### 6. `convert_frame` Function
A utility function to convert OpenCV BGR frames to MediaPipe's SRGB image format.

```python
def convert_frame(frame):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
```

### 7. Video Processing Loop
The main loop processes each frame of the uploaded video.

-   **Calibration**: An initial phase where the user is asked to look directly at the camera to establish baseline head pose angles.
-   **Gaze Violation Detection**: Compares the current head pose (pitch and yaw) against the calibrated angles. If deviations exceed `MAX_PITCH_DEG` or `MAX_YAW_DEG`, a gaze violation is flagged.
-   **Multiple Person Detection**: Checks for more than one face detected by MediaPipe, indicating an unauthorized person.
-   **Object Detection**: Uses the YOLOv8n model to detect specific unauthorized objects (e.g., "cell phone", "book") within the frame.
-   **Visual Feedback**: Overlays text annotations on the frame to indicate detected violations.

```python
cap = cv2.VideoCapture(VIDEO_PATH)

with vision.FaceLandmarker.create_from_options(options_land) as landmarker:


    print("Calibrating... Please look directly at the camera.")
    calibration_data = []
    while len(calibration_data) < 30:
        ret, frame = cap.read()
        if not ret: break
        mp_img = convert_frame(frame)
        res = landmarker.detect(mp_img)
        if res.face_landmarks:
            angles = estimate_head_pose(res.face_landmarks[0], frame.shape[1], frame.shape[0])
            if angles: calibration_data.append(angles[:2])

    if calibration_data:
        calibrated_pitch, calibrated_yaw = np.mean(calibration_data, axis=0)
        print(f"Calibration Complete: Pitch {calibrated_pitch:.2f}, Yaw {calibrated_yaw:.2f}")


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        mp_img = convert_frame(frame)
        land_res = landmarker.detect(mp_img)


        looking_away = False
        multi_face = False


        if land_res.face_landmarks:
            num_faces = len(land_res.face_landmarks)
            if num_faces > 1:
                unauthorized_person_detected_count += 1
                multi_face = True


            angles = estimate_head_pose(land_res.face_landmarks[0], w, h)
            if angles:
                p_diff = abs(angles[0] - calibrated_pitch)
                y_diff = abs(angles[1] - calibrated_yaw)
                if p_diff > MAX_PITCH_DEG or y_diff > MAX_YAW_DEG:
                    away_count += 1
                    looking_away = True


        y_res = yolo_model(frame, verbose=False, conf=0.4)
        phone_present = False
        for r in y_res:
            for box in r.boxes:
                cls = r.names[int(box.cls[0])]
                if cls in ["cell phone", "book"]:
                    phone_detected_count += 1
                    phone_present = True
                    detected_object_types.add(cls)
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)


        if looking_away: cv2.putText(frame, "GAZE VIOLATION", (30, 50), 2, 0.8, (0, 0, 255), 2)
        if multi_face: cv2.putText(frame, "MULTIPLE PEOPLE", (30, 80), 2, 0.8, (0, 0, 255), 2)
        if phone_present: cv2.putText(frame, "OBJECT PROHIBITED", (30, 110), 2, 0.8, (0, 0, 255), 2)



```

## Final Proctoring Report
After processing the entire video, the system generates a summary report detailing any detected violations.

```python
cap.release()
print(f"\n--- Final Proctoring Report ---")

gaze_status = "Yes" if away_count > 0 else "No"
object_status = "Yes" if phone_detected_count > 0 else "No"
person_status = "Yes" if unauthorized_person_detected_count > 0 else "No"
object_list = ", ".join(detected_object_types) if detected_object_types else "N/A"

print(f"Gaze Violation Detected: {gaze_status}")
print(f"Unauthorized Object Detected: {object_status} ({object_list})")
print(f"Multiple Person Violation: {person_status}")
```

The report provides clear indicators for:
-   **Gaze Violation Detected**: Indicates if the student looked away from the screen for a significant duration.
-   **Unauthorized Object Detected**: Lists any prohibited objects (e.g., cell phone, book) identified.
-   **Multiple Person Violation**: Flags if more than one person was detected in the frame at any point.
