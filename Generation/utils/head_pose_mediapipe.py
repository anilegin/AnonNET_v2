# import cv2
# import mediapipe as mp
# import numpy as np
# from math import radians
# import sys
# import os

# mp_face_mesh = mp.solutions.face_mesh

# def get_head_pose(frame):
#     """
#     Given a BGR image (frame), this function uses MediaPipe FaceMesh to detect facial landmarks,
#     selects a subset of landmarks, and then computes the head pose (rotation vector) using OpenCV's solvePnP.
    
#     Returns:
#         rotation_vector (np.ndarray): The rotation vector for the detected face, or None if detection fails.
#     """
#     # Convert the image to RGB as MediaPipe expects.
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     h, w, _ = frame.shape
#     # Use MediaPipe FaceMesh in static image mode.
#     with mp_face_mesh.FaceMesh(static_image_mode=True,
#                                max_num_faces=1,
#                                refine_landmarks=True,
#                                min_detection_confidence=0.5) as face_mesh:
#         results = face_mesh.process(frame_rgb)
#         if not results.multi_face_landmarks:
#             return None
#         # Get the first detected face's landmarks.
#         landmarks = results.multi_face_landmarks[0].landmark
        
#         # Convert normalized landmark coordinates to pixel coordinates.
#         # Using common indices:
#         #   Nose tip: 1, Chin: 152, Left eye outer corner: 33, Right eye outer corner: 263,
#         #   Left mouth corner: 61, Right mouth corner: 291.
#         image_points = np.array([
#             (landmarks[1].x * w, landmarks[1].y * h),     # Nose tip
#             (landmarks[152].x * w, landmarks[152].y * h),   # Chin
#             (landmarks[33].x * w, landmarks[33].y * h),     # Left eye outer corner
#             (landmarks[263].x * w, landmarks[263].y * h),   # Right eye outer corner
#             (landmarks[61].x * w, landmarks[61].y * h),     # Left mouth corner
#             (landmarks[291].x * w, landmarks[291].y * h)    # Right mouth corner
#         ], dtype="double")
        
#         # Define corresponding 3D model points.
#         model_points = np.array([
#             (0.0, 0.0, 0.0),             # Nose tip.
#             (0.0, -330.0, -65.0),        # Chin.
#             (-225.0, 170.0, -135.0),      # Left eye outer corner.
#             (225.0, 170.0, -135.0),       # Right eye outer corner.
#             (-150.0, -150.0, -125.0),     # Left mouth corner.
#             (150.0, -150.0, -125.0)       # Right mouth corner.
#         ])
        
#         # Camera internals.
#         focal_length = w
#         center = (w / 2, h / 2)
#         camera_matrix = np.array([
#             [focal_length, 0, center[0]],
#             [0, focal_length, center[1]],
#             [0, 0, 1]
#         ], dtype="double")
        
#         dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion.
        
#         success, rotation_vector, translation_vector = cv2.solvePnP(
#             model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
#         if not success:
#             return None
#         return rotation_vector

# def rotation_vector_to_euler_angles(rotation_vector):
#     """
#     Converts a rotation vector to Euler angles (pitch, yaw, roll) in degrees.
#     """
#     rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
#     sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
#     singular = sy < 1e-6
#     if not singular:
#         pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
#         yaw = np.arctan2(-rotation_matrix[2, 0], sy)
#         roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
#     else:
#         pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
#         yaw = np.arctan2(-rotation_matrix[2, 0], sy)
#         roll = 0
#     return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

# def get_frontal_frame(video_path, start, end, frontal_threshold=10, max_attempts=10, best=False):
#     """
#     Opens the video at the specified start time and checks up to max_attempts frames 
#     (or until the current time exceeds `end`) to find one where the detected face is roughly frontal.
    
#     A face is considered frontal if the absolute values of yaw and pitch are both below frontal_threshold.
#     If best is False, the function returns the first frame that meets the criteria.
#     If best is True, it examines up to max_attempts frames and returns the one with the smallest
#     sum of absolute yaw and pitch values.
#     The function will stop reading if the current video time exceeds `end`.
#     If no qualifying frame is found, the first extracted frame is returned.
    
#     Args:
#         video_path (str): Path to the video file.
#         start (float): Start time in seconds.
#         end (float): End time in seconds. Frames beyond this time will not be considered.
#         frontal_threshold (float): Maximum acceptable absolute angle for yaw and pitch (in degrees).
#         max_attempts (int): Maximum number of frames to try.
#         best (bool): If True, return the most frontal frame (lowest error) over max_attempts.
    
#     Returns:
#         np.ndarray or None: The selected frame (BGR) or None if no frame is available.
#     """
#     cap = cv2.VideoCapture(video_path)
#     cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)  # Set start position (ms)
    
#     first_frame = None
#     frontal_frame = None
#     best_candidate = None
#     best_error = float('inf')
#     attempt = 0

#     while attempt < max_attempts:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # in seconds
#         if current_time > end:
#             print("Reached the specified end time.")
#             break
#         if first_frame is None:
#             first_frame = frame.copy()
#         head_pose = get_head_pose(frame)
#         if head_pose is not None:
#             pitch, yaw, roll = rotation_vector_to_euler_angles(head_pose)
#             error = abs(yaw) + abs(pitch)
#             if best:
#                 if error < best_error:
#                     best_error = error
#                     best_candidate = frame.copy()
#             else:
#                 if abs(yaw) < frontal_threshold and abs(pitch) < frontal_threshold:
#                     frontal_frame = frame.copy()
#                     print(f"Frontal frame found at attempt {attempt+1}: pitch={pitch:.2f}, yaw={yaw:.2f}, roll={roll:.2f}")
#                     break
#         attempt += 1

#     cap.release()
#     if best:
#         if best_candidate is not None:
#             print(f"Best frame found with error {best_error:.2f}")
#             return best_candidate
#         else:
#             print("No face detected in any frame; using the first extracted frame.")
#             return first_frame
#     else:
#         if frontal_frame is not None:
#             return frontal_frame
#         else:
#             print("No frontal frame found; using the first extracted frame.")
#             return first_frame

# def main():
#     video_path = "your_video.mp4"  # Replace with your video file
#     start_time = 10.0  # For example, start at 10 seconds
#     end_time = 20.0    # Only consider frames until 20 seconds
#     # Set best=True if you want to search for the most frontal frame, otherwise False.
#     frame = get_frontal_frame(video_path, start_time, end_time, frontal_threshold=10, max_attempts=10, best=True)
#     if frame is None:
#         print("No frame could be extracted.")
#         sys.exit(1)
    
#     # Display the selected frame
#     cv2.imshow("Selected Frontal Frame", frame)
#     print("Press any key to exit...")
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     # Optionally, save the frame.
#     cv2.imwrite("frontal_frame.png", frame)
#     print("Frontal frame saved as 'frontal_frame.png'.")

# if __name__ == "__main__":
#     main()
