import subprocess
import sys
import re
import logging
import math
from typing import List, Dict, Any

from PIL import Image
from io import BytesIO

import argparse



def compute_average_color(video_path: str, start: float) -> tuple:
    """
    Extracts a frame at `start` seconds and computes the average RGB color.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-ss", str(start), "-i", video_path,
        "-frames:v", "1", "-q:v", "2", "-f", "image2", "pipe:1"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        image = Image.open(BytesIO(result.stdout))
        image = image.convert("RGB")  # Ensure it's RGB
        
        # Compute average color
        pixels = list(image.getdata())
        avg_r = sum(p[0] for p in pixels) / len(pixels)
        avg_g = sum(p[1] for p in pixels) / len(pixels)
        avg_b = sum(p[2] for p in pixels) / len(pixels)

        return (avg_r, avg_g, avg_b)
    except Exception as e:
        logging.error(f"Error extracting frame: {e}")
        return (0, 0, 0)  # Fallback if extraction fails


def color_distance(c1: tuple, c2: tuple) -> float:
    """
    Computes the Euclidean distance between two RGB colors.
    """
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2)


def detect_scenes_and_assign_ids(
    input_video: str,
    threshold: float = 0.2,  # Less sensitive scene detection
    similarity_threshold: float = 5.0  # Lower means stricter similarity
) -> List[Dict[str, Any]]:
    """
    Detect scenes using ffmpeg and assign IDs based on visual similarity.
    
    Returns:
        List[Dict]:
            - "start": Start time of the scene
            - "end":   End time of the scene
            - "scene_id": Integer ID representing the scene type
    """

    cmd = [
        "ffmpeg", "-hide_banner", "-i", input_video,
        "-filter_complex", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr", "-f", "null", "-"
    ]

    logging.info(f"Running FFmpeg scene detection for {input_video}...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"Error in detect_scenes: {e.stderr}\n")
        return []

    # Extract timestamps from the FFmpeg output
    timestamps = []
    for line in (result.stdout + result.stderr).splitlines():
        match = re.search(r"time:([\d\.]+)", line)  # Extracts "time:XX.XX"
        if match:
            timestamps.append(float(match.group(1)))

    timestamps = sorted(timestamps)

    # Ensure the first scene starts at 0.0
    if timestamps and timestamps[0] > 0.0:
        timestamps.insert(0, 0.0)

    if not timestamps:
        logging.warning("No scene changes detected. Try reducing the threshold.")
        return []

    # Convert timestamps to scene segments with IDs
    scenes = []
    scene_signatures = []  # Stores (scene_id, avg_color)
    next_scene_id = 1

    for i in range(len(timestamps) - 1):
        start_time = timestamps[i]
        end_time = timestamps[i + 1]

        # Compute scene signature (average color)
        avg_color = compute_average_color(input_video, (start_time + end_time) / 2)

        # Compare with previous scenes
        assigned_id = None
        for scene_id, signature in scene_signatures:
            if color_distance(avg_color, signature) < similarity_threshold:
                assigned_id = scene_id
                break

        if assigned_id is None:
            assigned_id = next_scene_id
            scene_signatures.append((assigned_id, avg_color))
            next_scene_id += 1

        scenes.append({
            "start": start_time,
            "end": end_time,
            "scene_id": assigned_id
        })

    return scenes


# Run the function
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_path", type=str, required=True)
    args = parser.parse_args()

    result = detect_scenes_and_assign_ids(args.vid_path)
    print(result)











    
    
  ## v2 works fine  
# def detect_scenes_and_assign_ids(
#     input_video: str,
#     threshold: float = 0.2,  # Very sensitive scene detection
#     similarity_threshold: float = 20.0  # Lower means stricter similarity
# ) -> List[Dict[str, Any]]:
#     """
#     Detect scenes using ffmpeg and assign IDs based on visual similarity.
    
#     Returns:
#         List[Dict]:
#             - "start": Start time of the scene
#             - "end":   End time of the scene
#             - "scene_id": Integer ID representing the scene type
#     """

#     cmd = [
#         "ffmpeg", "-hide_banner", "-i", input_video,
#         "-filter_complex", f"select='gt(scene,{threshold})',showinfo",
#         "-vsync", "vfr", "-f", "null", "-"
#     ]

#     logging.info(f"Running FFmpeg scene detection for {input_video}...")

#     try:
#         result = subprocess.run(cmd, capture_output=True, text=True, check=True)
#     except subprocess.CalledProcessError as e:
#         sys.stderr.write(f"Error in detect_scenes: {e.stderr}\n")
#         return []

#     # Extract timestamps from the FFmpeg output
#     timestamps = []
#     for line in (result.stdout + result.stderr).splitlines():
#         match = re.search(r"time:([\d\.]+)", line)  # Extracts "time:XX.XX"
#         if match:
#             timestamps.append(float(match.group(1)))

#     timestamps = sorted(timestamps)

#     if not timestamps:
#         logging.warning("No scene changes detected. Try reducing the threshold.")
#         return []

#     # Convert timestamps to scene segments with IDs
#     scenes = []
#     scene_signatures = []  # Stores (scene_id, avg_color)
#     next_scene_id = 1

#     for i in range(len(timestamps) - 1):
#         start_time = timestamps[i]
#         end_time = timestamps[i + 1]

#         # Compute scene signature (average color)
#         avg_color = compute_average_color(input_video, (start_time + end_time) / 2)

#         # Compare with previous scenes
#         assigned_id = None
#         for scene_id, signature in scene_signatures:
#             if color_distance(avg_color, signature) < similarity_threshold:
#                 assigned_id = scene_id
#                 break

#         if assigned_id is None:
#             assigned_id = next_scene_id
#             scene_signatures.append((assigned_id, avg_color))
#             next_scene_id += 1

#         scenes.append({
#             "start": start_time,
#             "end": end_time,
#             "scene_id": assigned_id
#         })

#     return scenes
    
    
######IT WORKS FINE
# def detect_scenes_and_assign_ids(
#     input_video: str,
#     threshold: float = 0.002,  # Very sensitive scene detection
#     similarity_threshold: float = 20.0
# ) -> List[Dict[str, Any]]:
#     """
#     Detect scenes using ffmpeg and return scene timestamps.

#     Returns:
#         List[Dict]:
#             - "start": Start time of the scene
#             - "end":   End time of the scene
#             - "scene_id": Integer ID representing the scene type
#     """

#     cmd = [
#         "ffmpeg", "-hide_banner", "-i", input_video,
#         "-filter_complex", f"select='gt(scene,{threshold})',showinfo",
#         "-vsync", "vfr", "-f", "null", "-"
#     ]

#     logging.info(f"Running FFmpeg scene detection for {input_video}...")

#     try:
#         result = subprocess.run(cmd, capture_output=True, text=True, check=True)
#     except subprocess.CalledProcessError as e:
#         sys.stderr.write(f"Error in detect_scenes: {e.stderr}\n")
#         return []

#     # Extract timestamps from the FFmpeg output
#     timestamps = []
#     for line in (result.stdout + result.stderr).splitlines():
#         match = re.search(r"time:([\d\.]+)", line)  # Extracts "time:XX.XX"
#         if match:
#             timestamps.append(float(match.group(1)))

#     timestamps = sorted(timestamps)

#     if not timestamps:
#         logging.warning("No scene changes detected. Try reducing the threshold.")
#         return []

#     # Convert timestamps to scene segments with IDs
#     scenes = []
#     next_scene_id = 1

#     for i in range(len(timestamps) - 1):
#         scenes.append({
#             "start": timestamps[i],
#             "end": timestamps[i + 1],
#             "scene_id": next_scene_id
#         })
#         next_scene_id += 1

#     # Ensure the last timestamp is included
#     if timestamps:
#         scenes.append({
#             "start": timestamps[-1],
#             "end": timestamps[-1],  # End is the same as the last timestamp
#             "scene_id": next_scene_id
#         })

#     return scenes






#IDK IF THEY WORKS

# def get_video_duration(video_path: str) -> float:
#     """
#     Get duration of the video in seconds.
#     """
#     cmd = [
#         "ffprobe", "-v", "error", "-show_entries",
#         "format=duration", "-of",
#         "default=noprint_wrappers=1:nokey=1", video_path
#     ]
#     try:
#         result = subprocess.run(cmd, capture_output=True, text=True, check=True)
#         return float(result.stdout.strip())
#     except subprocess.CalledProcessError as e:
#         logging.error(f"Could not get video duration: {e.stderr}")
#         return 0.0


# def compute_average_color(video_path: str, start: float, end: float) -> tuple:
#     """
#     Extract a frame around the midpoint of [start, end] and compute average RGB color.
#     A simple approach for demonstration only.
#     """
#     midpoint = start + (end - start) / 2
#     # Extract one frame at the midpoint.
#     # -ss must come before -i for ffmpeg to do a fast seek in many cases
#     cmd = [
#         "ffmpeg",
#         "-hide_banner",
#         "-ss", str(midpoint),
#         "-i", video_path,
#         "-frames:v", "1",
#         "-q:v", "2",            # output frame quality
#         "-f", "image2",
#         "pipe:1"                # send raw image data to stdout
#     ]
#     try:
#         # Capture raw image data in memory (PNG/JPEG).
#         result = subprocess.run(cmd, capture_output=True, check=True)
#     except subprocess.CalledProcessError as e:
#         logging.error(f"Error extracting frame: {e.stderr}")
#         return (0, 0, 0)  # fallback

#     # Decode the raw bytes into something like Pillow (PIL) image
#     try:
#         from PIL import Image
#         from io import BytesIO

#         image = Image.open(BytesIO(result.stdout))
#         # Convert to RGB if not already
#         image = image.convert("RGB")

#         # Compute average color
#         # sum of pixel values / total pixels
#         pixels = list(image.getdata())
#         r_sum, g_sum, b_sum = 0, 0, 0
#         for (r, g, b) in pixels:
#             r_sum += r
#             g_sum += g
#             b_sum += b
#         num_pixels = len(pixels)
#         avg_r = r_sum / num_pixels
#         avg_g = g_sum / num_pixels
#         avg_b = b_sum / num_pixels

#         return (avg_r, avg_g, avg_b)
#     except ImportError:
#         # If PIL is not installed, just return placeholder
#         logging.warning("PIL not installed. Returning (0,0,0) for average color.")
#         return (0, 0, 0)
#     except Exception as e:
#         logging.error(f"Unexpected error reading frame data: {e}")
#         return (0, 0, 0)


# def color_distance(c1: tuple, c2: tuple) -> float:
#     """
#     Euclidean distance between two RGB colors.
#     """
#     return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2)
