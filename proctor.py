import cv2
import numpy as np
import datetime
import os
import urllib.request

# Auto-download LBF model if not present
MODEL_PATH = "lbfmodel.yaml"
MODEL_URL = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading lbfmodel.yaml...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded.")

# Load classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load facial landmark model
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(MODEL_PATH)

def detect_cheating(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cheating_event = None

    if len(faces) != 1:
        cheating_event = "Multiple/No Faces Detected"
        log_event(cheating_event, frame)
        return frame, cheating_event

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) < 1:
            cheating_event = "Eyes Not Detected"
            log_event(cheating_event, frame)

        _, landmarks = facemark.fit(gray, np.array([[(x, y, x+w, y+h)]]))

        if landmarks:
            image_points = np.array([
                landmarks[0][0][30],  # Nose tip
                landmarks[0][0][8],   # Chin
                landmarks[0][0][36],  # Left eye left corner
                landmarks[0][0][45],  # Right eye right corner
                landmarks[0][0][48],  # Left Mouth corner
                landmarks[0][0][54]   # Right mouth corner
            ], dtype="double")

            model_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0)
            ])

            size = frame.shape
            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")

            dist_coeffs = np.zeros((4, 1))
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )

            rmat, _ = cv2.Rodrigues(rotation_vector)
            proj_matrix = np.hstack((rmat, translation_vector))
            _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)

            yaw = eulerAngles[1][0]   # side to side
            pitch = eulerAngles[0][0] # up and down

            if abs(yaw) > 25:
                cheating_event = f"Looking Away (Yaw: {yaw:.1f}°)"
                log_event(cheating_event, frame)
            elif abs(pitch) > 20:
                cheating_event = f"Looking Up/Down (Pitch: {pitch:.1f}°)"
                log_event(cheating_event, frame)

        # Optional: draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame, cheating_event

def log_event(event, frame=None):
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    with open("log.csv", "a") as f:
        f.write(f"{now},{event}\n")

    if frame is not None:
        cv2.imwrite(f"screenshots/{now}.jpg", frame)
