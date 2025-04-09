# video_utils.py

import os
import cv2
import numpy as np

################################################################################
# Video Trimming
################################################################################

def get_video_length_seconds(video_path):
    """
    Return the duration (in seconds) of a video using OpenCV.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        raise ValueError(f"Cannot read valid FPS from {video_path}")
    return frame_count / fps

def extract_subvideo(input_path, output_path, start_sec, end_sec):
    """
    Extract portion [start_sec, end_sec] from input video using OpenCV 
    and save to output_path in MP4 format.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    cap.release()

def split_video_by_seconds(video_path, segment_length, output_dir):
    """
    Split a video into multiple segments of fixed `segment_length` in seconds.
    The last segment may be shorter if the video duration is not a multiple of segment_length.

    Args:
        video_path (str): Input video file path.
        segment_length (float): Length of each segment (seconds).
        output_dir (str): Directory to store the resulting segments.
    """
    os.makedirs(output_dir, exist_ok=True)
    duration = get_video_length_seconds(video_path)

    start = 0
    segment_index = 0

    while start < duration:
        end = start + segment_length
        if end > duration:
            end = duration

        out_path = os.path.join(output_dir, f"segment_{segment_index}.mp4")
        extract_subvideo(video_path, out_path, start, end)

        start = end
        segment_index += 1

    print(f"Split {video_path} into {segment_index} segments in {output_dir}.")

def merge_segments(segments_list, merged_path):
    """
    Merge a list of .mp4 segment files (in chronological order) into a single .mp4 video.
    Uses OpenCV for concatenation.

    Args:
        segments_list (list of str): Sorted list of segment paths to merge.
        merged_path (str): File path for the merged output video.
    """
    if not segments_list:
        print("No segments found to merge.")
        return

    # Read info from the first segment
    cap0 = cv2.VideoCapture(segments_list[0])
    fps = cap0.get(cv2.CAP_PROP_FPS)
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(merged_path, fourcc, fps, (width, height))

    for seg_path in segments_list:
        cap = cv2.VideoCapture(seg_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()
    print(f"Merged segments saved as {merged_path}.")

def split_video_in_chunks(video_path, chunk_length_sec, output_dir):
    """
    Split a video into multiple "big chunks" (e.g., 1-minute chunks). Each chunk is stored as
    chunk_0.mp4, chunk_1.mp4, etc. This is helpful if you have a very long video and want to
    process it in larger slices.

    Args:
        video_path (str): Input video file path.
        chunk_length_sec (float): Duration of each chunk in seconds.
        output_dir (str): Directory to store chunked outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    duration = get_video_length_seconds(video_path)

    start = 0.0
    chunk_index = 0

    while start < duration:
        end = min(start + chunk_length_sec, duration)
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_index}.mp4")
        extract_subvideo(video_path, chunk_path, start, end)
        chunk_index += 1
        start = end

    print(f"Split {video_path} into {chunk_index} chunks (length ~{chunk_length_sec}s) in {output_dir}.")
    
    
def extract_first_frame(video_path, output_image_path):
    """
    Extract the first frame from video_path and save it as an image to output_image_path.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_image_path, frame)
    cap.release()
    
    
def extract_first_frame_as_array(video_path):
    """
    Extract the first frame from a video file and return it as a NumPy ndarray (BGR format).
    
    :param video_path: Path to the video file.
    :return: NumPy ndarray (BGR format) of the first frame, or None if extraction fails.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame  # Returns the first frame as a NumPy array (BGR format)
    else:
        return None  # Return None if no frame is read
    

def extract_video_segment_as_array(input_video, start_time, end_time):
    """
    Extracts a specific segment from a video and returns it as a list of NumPy arrays.

    :param input_video: Path to the input video file.
    :param start_time: Start time in seconds.
    :param end_time: End time in seconds.
    :return: List of NumPy arrays (frames in BGR format).
    """
    cap = cv2.VideoCapture(input_video)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Move to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []  # List to store frames

    while cap.isOpened():
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if current_frame >= end_frame:
            break  # Stop when we reach the end time

        ret, frame = cap.read()
        if not ret:
            break  # Stop if we can't read a frame

        frames.append(frame)  # Store the frame as a NumPy array

    cap.release()
    return np.array(frames)  # Return all extracted frames as an array
    
    

    
    
################################################################################
# Scene Detection
################################################################################

