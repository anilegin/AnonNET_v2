import os
import time
from utils import video_utils
from tqdm import tqdm


def create_clip_tasks(scene_video_paths, clip_length):
    """
    Create tasks to split each scene into clips of a specified length.
    """
    tasks = []
    for i, (scene_id, scene_video_path) in enumerate(scene_video_paths):
        clip_dir = os.path.join(
            os.path.dirname(scene_video_path),
            f"clips_sceneIndex_{i}"  # use scene INDEX instead of scene_id
        )
        os.makedirs(clip_dir, exist_ok=True)
        
        print(f"[DEBUG] Splitting scene_index={i}, scene_id={scene_id} -> {scene_video_path}")
        print(f"[DEBUG] Output clips go in: {clip_dir}")
        
        video_utils.split_video_by_seconds(scene_video_path, clip_length, clip_dir)
        
        # Collect newly created clips
        clips = sorted([
            os.path.join(clip_dir, f) for f in os.listdir(clip_dir) if f.endswith('.mp4')
        ])
        
        # Print the discovered clips
        print(f"[DEBUG] Found {len(clips)} clipped files for scene_id={scene_id}:")
        for c in clips:
            print(f"   -> {c}")

        # Build tasks
        for i, clip in enumerate(clips):
            out_clip = os.path.join(clip_dir, f"processed_clip_{i}.mp4")
            tasks.append((clip, scene_id, out_clip))
            print(f"[DEBUG] +Task: input={clip}, output={out_clip}")

    print(f"[DEBUG] Total tasks created: {len(tasks)}")
    return tasks


def extract_subvideos_for_scenes(scenes, driving_path, temp_driving_dir):
    """
    Extract subvideos for each scene and return the paths.
    """
    scene_video_paths = []  # list[(scene_id, scene_video_path), ...]
    used_ids = set()
    for i, scene_info in enumerate(scenes):
        sid = scene_info["scene_id"]
        start, end = scene_info["start"], scene_info["end"]
        scene_vid_path = os.path.join(temp_driving_dir, f"scene_{i}.mp4")

        # Extract subvideo for [start, end]
        video_utils.extract_subvideo(driving_path, scene_vid_path, start, end)
        scene_video_paths.append((sid, scene_vid_path))

        # If first occurrence of scene_id, optionally extract first frame
        if sid not in used_ids:
            used_ids.add(sid)
            # Optionally extract first frame (if needed)
            source_png = os.path.join(temp_source_dir, f"source_{sid}.png")
            video_utils.extract_first_frame(scene_vid_path, source_png)
            # (Uncomment if required)

    return scene_video_paths


def main(args):
    os.makedirs(args.save_folder, exist_ok=True)
    temp_driving_dir = os.path.join(args.save_folder, "temp_driving")
    os.makedirs(temp_driving_dir, exist_ok=True)

    print("Detecting scenes in driving video...")
    scenes = detect_scenes_and_assign_ids(
        args.driving_path,
        threshold=args.scene_threshold,
        similarity_threshold=args.scene_similarity_threshold
    )
    
    # ▼▼ ADDING A DEBUG PRINT FOR #unique scene IDs ▼▼
    unique_ids = set(s["scene_id"] for s in scenes)
    print(f"Detected {len(scenes)} scene segments in total.")
    print(f"[DEBUG] UNIQUE SCENE IDs = {sorted(unique_ids)} (TOTAL: {len(unique_ids)})")

    # Also, show each segment's ID/start/end
    for i, scene_info in enumerate(scenes):
        sid = scene_info["scene_id"]
        start, end = scene_info["start"], scene_info["end"]
        print(f"[DEBUG] scene_index={i} => scene_id={sid}, start={start}, end={end}")

    # Extract sub-videos for each scene
    scene_video_paths = extract_subvideos_for_scenes(scenes, args.driving_path, temp_driving_dir)

    # Create clip tasks
    tasks = create_clip_tasks(scene_video_paths, args.clip_length)
    print(f"Created {len(tasks)} clip tasks.")

    # Count total frames for all tasks -> single tqdm bar
    total_frames = 0
    for (clip_path, sid, out_clip_path) in tasks:
        info = torchvision.io.read_video(clip_path, pts_unit='sec')
        frame_count = info[0].shape[0]
        total_frames += frame_count

    print(f"Total frames across all clips: {total_frames}")

    # Parallel processing w/ progress
    # This part is kept as is to process the clips in parallel, can adjust as needed.
    # pool = ...

if __name__ == "__main__":
    # Argument parsing remains as is
    parser = argparse.ArgumentParser()
    parser.add_argument("--driving_path", type=str, default="driving.mp4")
    parser.add_argument("--save_folder", type=str, default="./res_presentation")
    parser.add_argument("--clip_length", type=int, default=20, help="Split scenes into 20s sub-videos.")
    parser.add_argument("--scene_threshold", type=float, default=0.2)
    parser.add_argument("--scene_similarity_threshold", type=float, default=5.0)
    parser.add_argument("--num_workers", type=int, default=4, help="Multiprocessing worker count.")
    args = parser.parse_args()

    main(args)
