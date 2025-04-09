import cv2

def side_by_side_19sec(a,b):
    # Ask user for input video paths
    left_video_path = a
    right_video_path = b
    
    # Open captures for both videos
    cap_left = cv2.VideoCapture(left_video_path)
    cap_right = cv2.VideoCapture(right_video_path)
    
    # Get frames per second for each video
    fps_left = cap_left.get(cv2.CAP_PROP_FPS)
    fps_right = cap_right.get(cv2.CAP_PROP_FPS)
    # Choose an FPS to use for the output (here we pick the lower to avoid reading extra frames)
    fps = min(fps_left, fps_right)
    
    # Get the frame sizes of each video
    width_left = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_left = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    width_right = int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_right = int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # The output video width is sum of both widths, and the height is the max of both
    out_width = width_left + width_right
    out_height = max(height_left, height_right)
    
    # Create a VideoWriter for the output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = "/home/aegin/projects/anonymization/AnonNET/Generation/mergingvids/output_real_anon.mp4"
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    # We'll combine up to 19 seconds: total frames = fps * 19
    max_frames = int(fps * 19)
    frame_count = 0
    
    while frame_count < max_frames:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        # If either video ends early, stop writing
        if not ret_left or not ret_right:
            break
        
        # If heights differ, resize to match the output height (for a clean horizontal concat)
        if height_left != out_height:
            frame_left = cv2.resize(frame_left, (width_left, out_height))
        if height_right != out_height:
            frame_right = cv2.resize(frame_right, (width_right, out_height))
        
        # Combine frames horizontally
        combined_frame = cv2.hconcat([frame_left, frame_right])
        
        # Write to the output video
        out_writer.write(combined_frame)
        
        frame_count += 1
    
    # Release everything
    cap_left.release()
    cap_right.release()
    out_writer.release()
    print(f"Done! Output saved to {output_path}")


# If you want to run this script directly:
if __name__ == "__main__":
    side_by_side_19sec(
        a = '/home/aegin/projects/anonymization/AnonNET/Generation/mergingvids/trimmed-0-300-tom.mp4',
        b = "/home/aegin/projects/anonymization/AnonNET/Generation/mergingvids/tom_cruise_256.mp4_output.mp4"
        
        
    )
