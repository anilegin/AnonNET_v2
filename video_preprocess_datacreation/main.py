
import numpy as np 
import cv2
import torch
from PIL import Image
import argparse
import os
import argparse
import logging
from scene_detection import detect_scenes_and_assign_ids
from video_utils import extract_video_segment_as_array

from deepface import DeepFace
import cv2

def _convert(face_data):
    """
    Convert DeepFace face bounding box (x, y, w, h) to (x1, y1, x2, y2).
    
    :param face_data: Dictionary containing 'facial_area' with x, y, w, h.
    :return: Dictionary with x1, y1, x2, y2 coordinates.
    """
    x, y, w, h = face_data['facial_area']['x'], face_data['facial_area']['y']. face_data['facial_area']['w']. face_data['facial_area']['h']
    x1, x2, y1, y2 = x, x+w, y, y+h
    return [x1, x2, y1, y2]

def _apply_margin(loc, rate = 0.1):
    
    x1, x2, y1, y2 = loc
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    
    # Ensure box is valid
    if x2 <= x1 or y2 <= y1:
        print("Skipping invalid bounding box:", box)

    # Compute margin
    box_width = x2 - x1
    box_height = y2 - y1
    margin_w = int(rate * box_width)
    margin_h = int(rate * box_height)
    
    # Adjust bounding box with margins and clamp within image bounds
    x1 = max(0, x1 - margin_w)
    y1 = max(0, y1 - margin_h)
    x2 = min(w, x2 + margin_w)
    y2 = min(h, y2 + margin_h)
    
    return [x1,x2,y1,y2]
    

logging.basicConfig(level=logging.INFO)

def main(video_path, threshold=0.2, similarity_threshold=5.0):
    """
    1) Detect scenes using script.py
    2) For each scene:
       a) Extract frames via extract_video_segment_as_array
       b) Use DeepFace.extract to detect faces in the first frame
       c) Verify new faces against known faces (if any)
       d) Keep track of how many faces are found
    3) Print progress along the way
    4) At the end, report # of scenes with 0, 1, or multiple faces
    """

    # 1) Detect scenes
    logging.info("Detecting scenes...")
    scenes = detect_scenes_and_assign_ids(
        input_video=video_path,
        threshold=threshold,
        similarity_threshold=similarity_threshold
    )
    logging.info(f"Found {len(scenes)} scene(s).")

    if not scenes:
        logging.warning("No scenes detected or detection failed.")
        return

    # We will store known faces in a list (each item = {"name": "face_1", "embedding": ...})
    known_faces = []

    # Counters for final statistics
    scenes_with_0_faces = 0
    scenes_with_1_face  = 0
    scenes_with_many_faces = 0

    # 2) Iterate over each detected scene
    for idx, scene_data in enumerate(scenes):
        start_sec = scene_data["start"]
        end_sec   = scene_data["end"]
        scene_id  = scene_data["scene_id"]

        logging.info(f"=== Processing Scene {idx} (ID={scene_id}) Start={start_sec:.2f}s End={end_sec:.2f}s ===")

        # 2a) Extract frames for this clip
        frames_array = extract_video_segment_as_array(video_path, start_sec, end_sec)
        if len(frames_array) == 0:
            logging.info("No frames extracted. Skipping this scene.")
            scenes_with_0_faces += 1  # No frames => no faces
            continue

        # We'll focus on the first frame only
        first_frame = frames_array[1] # make it zero
        
        os.makedirs("./.cache/", exist_ok=True)
        cv2.imwrite(f"./.cache/scene_{idx}_frame.jpg", first_frame)

        # 2b) Use DeepFace.extract on the first frame
        # DeepFace expects either a file path or an array. We'll pass the array:
        # The 'extract' method returns a list of dicts, each containing "facial_area" and "embedding" (by default).
        try:
            extracted_faces = DeepFace.extract_faces(
                img_path = first_frame,        # can be a NumPy array
                detector_backend = 'retinaface',   # or 'mtcnn', 'retinaface', etc.
                enforce_detection = False,     # set to True if you want to raise errors on no-face
                align = True                  # whether to align face or not
            )
        except Exception as e:
            logging.error(f"DeepFace extract failed: {e}")
            extracted_faces = []

        extracted_faces = [f for f in extracted_faces if f.get("confidence", 0) > 0.8]
        num_faces = len(extracted_faces)
        logging.info(f"Detected {num_faces} face(s) in scene {idx}.")

        # Count for final stats
        if num_faces == 0:
            scenes_with_0_faces += 1
        elif num_faces == 1:
            scenes_with_1_face += 1
        else:
            scenes_with_many_faces += 1

        # 2c) For each detected face in the current scene's first frame,
        #     check if it already exists in known_faces
        for face_data in extracted_faces:

            # This dictionary typically has: 
            #   { "facial_area": ..., "embedding": np.array([...]) } 
            new_embedding = face_data["face"]

            # We’ll compare new_embedding with all known faces to see if it’s the same person.
            is_same = False
            for known_face in known_faces:
                # --- Option 1: Use DeepFace.verify to compare embeddings directly ---
                #   result = DeepFace.verify(
                #       img1_path = new_embedding,  # or pass as "embedding1" in new version
                #       img2_path = known_face["embedding"],
                #       enforce_detection=False,
                #       model_name="Facenet"
                #   )
                #   distance = result["distance"]
                #   threshold = result["threshold"]
                
                # --- Option 2: Manual distance check if you have the embedding arrays:
                # e.g. using Cosine distance or L2 distance
                # distance = deepface_distance.findCosineDistance(new_embedding, known_face["embedding"])
                # if distance < 0.3:  # Example threshold for "same" face
                #     is_same = True
                
                # For demonstration, we'll do a simple approach using verify with embeddings:
                try:
                    result = DeepFace.verify(
                        img1_path = new_embedding,
                        img2_path = known_face["embedding"],
                        model_name = "Facenet512",
                        detector_backend = "skip",
                        distance_metric = "cosine",
                        enforce_detection = False,
                        threshold = 5e-5
                    )
                    #print(result)
                    if result["verified"]:  
                        # If "verified" is True => same face 
                        is_same = True
                        break

                except Exception as e:
                    logging.warning(f"Verification error: {e}")
                    # We'll treat that as not matching if there's an error 
                    pass

            if not is_same:
                # This is a new face => Add to known_faces
                new_face_name = f"face_{len(known_faces) + 1}"
                known_faces.append({
                    "name": new_face_name,
                    "embedding": new_embedding
                })
                logging.info(f" --> New face found: {new_face_name}")
            else:
                logging.info(f" --> This face is already known; skipping append.")

    # 3) Print final statistics
    logging.info("=== Processing Complete ===")
    logging.info(f"Total Scenes Processed: {len(scenes)}")
    logging.info(f"Scenes with 0 faces: {scenes_with_0_faces}")
    logging.info(f"Scenes with 1 face: {scenes_with_1_face}")
    logging.info(f"Scenes with more than 1 face: {scenes_with_many_faces}")
    logging.info(f"Total Unique Faces Found: {len(known_faces)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video scenes and detect faces.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video.")
    parser.add_argument("--threshold", type=float, default=0.2, help="Scene detection threshold.")
    parser.add_argument("--similarity_threshold", type=float, default=5.0,
                        help="Color-similarity threshold for merging scene IDs.")
    args = parser.parse_args()

    main(
        video_path=args.video_path,
        threshold=args.threshold,
        similarity_threshold=args.similarity_threshold
    )


