import cv2

# def trim_video(input_path, output_path, start_sec, end_sec):
#     # Open the video file
#     cap = cv2.VideoCapture(input_path)
    
#     if not cap.isOpened():
#         print("Error: Could not open video file.")
#         return
    
#     # Get video properties
#     fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     # Calculate start and end frame numbers
#     start_frame = int(start_sec * fps)
#     end_frame = int(end_sec * fps)

#     if start_frame >= total_frames or end_frame > total_frames:
#         print("Error: Specified time range exceeds video duration.")
#         cap.release()
#         return

#     # Set up video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
#     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#     # Set the starting position
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
#     # Process and save the trimmed video
#     frame_count = start_frame
#     while frame_count < end_frame:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         out.write(frame)
#         frame_count += 1

#     # Release resources
#     cap.release()
#     out.release()
#     print(f"Trimmed video saved to {output_path}")

# if __name__ == "__main__":
    
    
#     trim_video("/home/aegin/projects/anonymization/AnonNET/Generation/vids/videoplayback.mp4", "combined_trim.mp4", 60, 120)



import cv2
import os
import subprocess


import cv2

def trim_first_20_seconds(
    input_path,
    output_temp_path,
    duration_sec=20,
    ref_fps=24,
    ref_width=640,
    ref_height=480
):
    """
    Reads the first `duration_sec` from `input_path` and writes them to
    `output_temp_path` at the given `ref_fps` and resolution (`ref_width`, `ref_height`).
    """
    cap = cv2.VideoCapture(input_path)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0:
        print(f"Warning: Could not determine FPS for {input_path}, defaulting to ref_fps={ref_fps}")
        input_fps = ref_fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_temp_path,
        fourcc,
        ref_fps,  # force the new FPS
        (ref_width, ref_height)
    )

    # We'll read up to `duration_sec` from the input
    max_frames_input = int(duration_sec * input_fps)
    frame_count = 0
    
    while frame_count < max_frames_input:
        ret, frame = cap.read()
        if not ret:
            break  # no more frames
        # Resize to reference resolution if needed
        if (frame.shape[1], frame.shape[0]) != (ref_width, ref_height):
            frame = cv2.resize(frame, (ref_width, ref_height))
        # Write the frame to temp file (re-encoded at ref_fps)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()



def merge_two_videos(video1_temp_path, video2_temp_path, output_path):
    """
    Merges two video files (that have the same resolution & FPS) sequentially
    into a single video using OpenCV.
    """
    cap1 = cv2.VideoCapture(video1_temp_path)
    cap2 = cv2.VideoCapture(video2_temp_path)

    # We assume both have the same fps/resolution because we enforced it
    fps_out = cap1.get(cv2.CAP_PROP_FPS)
    width  = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_out, (width, height))

    # Write all frames from the first trimmed file
    while True:
        ret, frame = cap1.read()
        if not ret:
            break
        out.write(frame)

    # Write all frames from the second trimmed file
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        out.write(frame)

    cap1.release()
    cap2.release()
    out.release()





if __name__ == "__main__":
    # append_trimmed_videos("/data/stars/share/HDTF-dataset/HDTF-RGB/WDA_MarkWarner_000.mp4",
    #                       "/data/stars/share/HDTF-dataset/HDTF-RGB/WRA_SteveDaines0_000.mp4",
    #                       "/home/aegin/projects/anonymization/AnonNET/Generation/vids/output.mp4",
    #                       duration=20, multiplier=2)


    video1_path = "/data/stars/share/HDTF-dataset/HDTF-RGB/WDA_MarkWarner_000.mp4"
    video2_path = "/data/stars/share/HDTF-dataset/HDTF-RGB/WRA_SteveDaines0_000.mp4"
    
    # Trim outputs
    video1_temp_path = "/home/aegin/projects/anonymization/AnonNET/Generation/vids/output_temp1.mp4"
    video2_temp_path = "/home/aegin/projects/anonymization/AnonNET/Generation/vids/output_temp2.mp4"
    
    # Decide on a "reference" size & fps
    # Option 1: Hardcode something like (640x480) @ 24 fps
    # Option 2: Actually read properties from the first video
    cap_ref = cv2.VideoCapture(video1_path)
    ref_fps   = cap_ref.get(cv2.CAP_PROP_FPS)
    ref_width  = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    ref_height = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_ref.release()

    # Also check second video for min fps if you want
    cap2 = cv2.VideoCapture(video2_path)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    cap2.release()
    if fps2 > 0 and ref_fps > 0:
        ref_fps = min(ref_fps, fps2)  # unify to min FPS
    else:
        ref_fps = 24.0  # fallback

    # 1) Trim each video's first 20 seconds into temp files
    trim_first_20_seconds(
        input_path=video1_path,
        output_temp_path=video1_temp_path,
        duration_sec=20,
        ref_fps=ref_fps,
        ref_width=ref_width,
        ref_height=ref_height
    )
    trim_first_20_seconds(
        input_path=video2_path,
        output_temp_path=video2_temp_path,
        duration_sec=20,
        ref_fps=ref_fps,
        ref_width=ref_width,
        ref_height=ref_height
    )

    # 2) Merge them into a single file
    output_path = "/home/aegin/projects/anonymization/AnonNET/Generation/vids/output.mp4"
    merge_two_videos(video1_temp_path, video2_temp_path, output_path)

    print("Merging completed. Final output:", output_path)
    print("Merging completed. Final output:", output_path)
