import cv2
import numpy as np
import tqdm
import os 

def extract_frames(video_path, frame_rate=2, duration=10):
    """
    Extract frames from the first `duration` seconds of a video at the specified frame rate.
    Returns a list of frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Video frame rate
    total_frames_to_process = min(int(fps * duration), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_interval = max(1, int(fps / frame_rate))  # Interval to sample frames

    frames = []
    with tqdm(total=total_frames_to_process, desc="Extracting Frames", unit="frame") as pbar:
        for i in range(total_frames_to_process):
            ret, frame = cap.read()
            if not ret:
                break
            if i % frame_interval == 0:  # Capture frames at the specified rate
                frames.append(frame)
            pbar.update(1)

    cap.release()
    return frames

# Define a function to check head pose
def detect_face_and_pose(frame, tolerance=15):
    """
    Detect the face in a frame and check if the head pose is within tolerance.
    Returns True if the face is directed at the camera.
    """
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        return False, None

    for (x, y, w, h) in faces:
        # Define 3D model points for head pose estimation
        model_points = np.array([
            (0.0, 0.0, 0.0),            # Nose tip
            (0.0, -330.0, -65.0),       # Chin
            (-225.0, 170.0, -135.0),    # Left eye corner
            (225.0, 170.0, -135.0),     # Right eye corner
            (-150.0, -150.0, -125.0),   # Left mouth corner
            (150.0, -150.0, -125.0)     # Right mouth corner
        ])

        # Define 2D image points from detected face
        image_points = np.array([
            (x + w//2, y + h//2),               # Nose tip
            (x + w//2, y + h),                  # Chin
            (x, y + h//4),                      # Left eye corner
            (x + w, y + h//4),                  # Right eye corner
            (x + w//4, y + 3*h//4),             # Left mouth corner
            (x + 3*w//4, y + 3*h//4)            # Right mouth corner
        ], dtype="double")

        # Camera matrix
        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        # Assume no lens distortion
        dist_coeffs = np.zeros((4, 1))

        # Solve for head pose
        _, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Extract yaw, pitch, and roll
        yaw = np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0]) * (180 / np.pi)
        pitch = np.arctan2(-rotation_matrix[2][0], np.sqrt(rotation_matrix[2][1]**2 + rotation_matrix[2][2]**2)) * (180 / np.pi)
        roll = np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2]) * (180 / np.pi)

        # Check if within tolerance
        if abs(yaw) <= tolerance and abs(pitch) <= tolerance:
            return True, frame

    return False, None

def process_video(video_path, output_path, frame_rate=2, tolerance=15):
    """
    Extract frames and find the first frame where the face is directed at the camera.
    """
    frames = extract_frames(video_path, frame_rate=frame_rate)
    for i, frame in enumerate(frames):
        is_direct, result_frame = detect_face_and_pose(frame, tolerance=tolerance)
        if is_direct:
            print(f"Frontal face found in frame {i}")
            cv2.imwrite(output_path, result_frame)
            return result_frame

    print("No frontal face detected in the first 10 seconds of the video.")
    return None
