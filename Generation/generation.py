#!/usr/bin/env python3
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, Manager, Queue


from utils import video_utils
from utils.scene_detection import detect_scenes_and_assign_ids
from networks.generator import Generator

# Ensure "spawn" start method for multiprocessing
mp.set_start_method('spawn', force=True)
os.environ["PYTHONWARNINGS"] = "ignore"  # Suppress warnings

################################################################################
# Preprocessing / Video I/O Helpers
################################################################################

def load_image(image, size):
    """
    Convert 'image' (path, PIL.Image, or np.array) -> (3,size,size) in [0,1].
    """
    if isinstance(image, str):
        img = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        img = image.convert('RGB')
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert('RGB')
    else:
        raise ValueError("Unsupported image type in load_image().")

    img = img.resize((size, size))
    arr = np.asarray(img)
    # (H,W,C) -> (C,H,W)
    arr = np.transpose(arr, (2, 0, 1))
    return arr / 255.0

def img_preprocessing(img_path, size):
    """
    Load and preprocess an image to shape (1,3,size,size) in [-1,1].
    """
    arr = load_image(img_path, size)  # (3,H,W) in [0,1]
    tensor = torch.from_numpy(arr).unsqueeze(0).float()  # => (1,3,H,W)
    tensor = (tensor - 0.5) * 2.0  # => [-1,1]
    return tensor

def vid_preprocessing(vid_path, size=256):
    """
    Read entire video -> shape: (1,T,3,H,W) in [-1,1], returns (vid_tensor, fps).
    """
    vid_data = torchvision.io.read_video(vid_path, pts_unit='sec')
    vid = vid_data[0]  # (T,H,W,C)
    fps = vid_data[2]['video_fps']

    vid_torch = vid.permute(0, 3, 1, 2).float()  # => (T,C,H,W)
    vid_resized = F.interpolate(vid_torch, size=(size, size), mode='bilinear', align_corners=False)
    vid_resized = vid_resized.unsqueeze(0) / 255.0  # => (1,T,C,H,W) in [0,1]
    vid_norm = (vid_resized - 0.5) * 2.0  # => [-1,1]
    return vid_norm, fps

def save_video(vid_tensor, save_path, fps):
    """
    Save (1,T,3,H,W) in [-1,1] to MP4 at 'save_path'.
    """
    # => (1,T,H,W,C)
    vid = vid_tensor.permute(0, 2, 3, 4, 1).clamp(-1,1).cpu()
    # scale [vid.min(), vid.max()] -> [0,255]
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).byte()
    # Cast fps to int if needed
    fps_int = int(round(float(fps)))
    torchvision.io.write_video(save_path, vid[0], fps=fps_int)

def load_source_tensor_from_temp(scene_id, size, default_source=None):
    """
    Load 'temp_source_images/source_<scene_id>.jpg' -> (1,3,size,size) in [-1,1], on cuda.
    If not found, optionally fallback to default_source path.
    """
    path = os.path.join("temp_source_images", f"source_{scene_id}.jpg")
    if os.path.exists(path):
        return img_preprocessing(path, size).cuda()
    elif default_source and os.path.exists(default_source):
        return img_preprocessing(default_source, size).cuda()
    else:
        raise FileNotFoundError(f"No source image found for scene {scene_id} and no fallback given.")

################################################################################
# Model Initialization & Core Inference
################################################################################

def init_generator(args):
    """
    Load your pretrained Generator model from the appropriate checkpoint.
    """
    if args.model == 'vox':
        ckpt = 'checkpoints/vox.pt'
    elif args.model == 'taichi':
        ckpt = 'checkpoints/taichi.pt'
    elif args.model == 'ted':
        ckpt = 'checkpoints/ted.pt'
    else:
        raise NotImplementedError(f"Unknown model: {args.model}")

    print(f"==> Loading generator from: {ckpt}")
    gen = Generator(
        args.size,
        args.latent_dim_style,
        args.latent_dim_motion,
        args.channel_multiplier
    ).cuda()
    weight = torch.load(ckpt, map_location=lambda storage, loc: storage)['gen']
    gen.load_state_dict(weight)
    gen.eval()
    return gen

def process_segment(gen, args, source_tensor, segment_path, save_path, progress_queue=None):
    """
    Process a single clip. 
    - Reads the clip (vid_preprocessing),
    - runs inference frame-by-frame,
    - saves output to 'save_path'.
    - If progress_queue is not None, put(1) after each frame for progress tracking.
    """
    vid_target, fps = vid_preprocessing(segment_path, size=args.size)
    vid_target = vid_target.cuda()  # (1,T,3,H,W)
    T = vid_target.size(1)

    if args.model == 'ted':
        h_start = None
    else:
        h_start = gen.enc.enc_motion(vid_target[:, 0])

    frames = []
    with torch.no_grad():
        for i in range(T):
            driving_frame = vid_target[:, i]  # (1,3,H,W)
            out_frame = gen(source_tensor, driving_frame, h_start)  # e.g. (1,3,H,W)
            frames.append(out_frame.unsqueeze(2))  # => (1,3,1,H,W)
            if progress_queue:
                progress_queue.put(1)  # one frame processed

    vid_out = torch.cat(frames, dim=2)  # => (1,3,T,H,W)
    save_video(vid_out, save_path, fps)
    return save_path

################################################################################
# Tasks & Multiprocessing
################################################################################

def create_clip_tasks(scene_video_paths, clip_length):
    print(f"[DEBUG] create_clip_tasks called with clip_length={clip_length}")
    tasks = []
    for scene_id, scene_video_path in scene_video_paths:
        clip_dir = os.path.join(os.path.dirname(scene_video_path), f"clips_scene_{scene_id}")
        os.makedirs(clip_dir, exist_ok=True)

        print(f"[DEBUG] Splitting scene_id={scene_id} -> {scene_video_path}")
        print(f"[DEBUG] Output clips go in: {clip_dir}")

        # Call your splitting function
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


def process_clip_worker(task, args, gen, progress_queue):
    """
    Worker entry point. Processes a single clip task:
      task = (clip_path, scene_id, output_clip_path)
    """
    clip_path, scene_id, output_clip_path = task
    # Load scene-specific source
    source_tensor = load_source_tensor_from_temp(scene_id, args.size)
    # Process the segment (each frame => progress_queue.put(1))
    process_segment(gen, args, source_tensor, clip_path, output_clip_path, progress_queue)
    return output_clip_path

################################################################################
# Main
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--model", type=str, choices=['vox', 'taichi', 'ted'], default='vox')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    # No explicit source_path arg because we generate them from first frames
    parser.add_argument("--driving_path", type=str, default="driving.mp4")
    parser.add_argument("--save_folder", type=str, default="./res_presentation")
    parser.add_argument("--clip_length", type=int, default=20, help="Split scenes into 20s sub-videos.")
    parser.add_argument("--scene_threshold", type=float, default=0.2)
    parser.add_argument("--scene_similarity_threshold", type=float, default=5.0)
    parser.add_argument("--num_workers", type=int, default=4, help="Multiprocessing worker count.")
    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)
    base_filename = os.path.basename(args.driving_path)

    gen = init_generator(args)

    # Scene detection
    print("Detecting scenes in driving video...")
    scenes = detect_scenes_and_assign_ids(
        args.driving_path,
        threshold=args.scene_threshold,
        similarity_threshold=args.scene_similarity_threshold
    )
    print(f"Detected {len(scenes)} scenes.")

    # Extract sub-videos for each scene + first-frame source images
    temp_driving_dir = os.path.join(args.save_folder, "temp_driving")
    os.makedirs(temp_driving_dir, exist_ok=True)
    temp_source_dir = "temp_source_images"
    os.makedirs(temp_source_dir, exist_ok=True)

    scene_video_paths = []  # list[(scene_id, scene_video_path), ...]
    used_ids = set()
    for i, scene_info in enumerate(scenes):
        sid = scene_info["scene_id"]
        start, end = scene_info["start"], scene_info["end"]
        scene_vid_path = os.path.join(temp_driving_dir, f"scene_{i}.mp4")
        # Extract subvideo for [start, end]
        video_utils.extract_subvideo(args.driving_path, scene_vid_path, start, end)
        scene_video_paths.append((sid, scene_vid_path))
        # If first occurrence of scene_id, extract first frame as source
        if sid not in used_ids:
            used_ids.add(sid)
            source_jpg = os.path.join(temp_source_dir, f"source_{sid}.jpg")
            video_utils.extract_first_frame(scene_vid_path, source_jpg)

    # Create clip tasks
    tasks = create_clip_tasks(scene_video_paths, args.clip_length)
    print(f"Created {len(tasks)} clip tasks.")

    # Count total frames for all tasks -> for single tqdm bar
    total_frames = 0
    for (clip_path, sid, out_clip_path) in tasks:
        info = torchvision.io.read_video(clip_path, pts_unit='sec')
        frame_count = info[0].shape[0]
        total_frames += frame_count

    print(f"Total frames across all clips: {total_frames}")

    # Run tasks in parallel, capturing progress
    manager = mp.Manager()
    progress_queue = manager.Queue()

    pool = mp.Pool(processes=args.num_workers)
    async_results = []
    for task in tasks:
        # apply_async -> returns AsyncResult
        r = pool.apply_async(process_clip_worker, (task, args, gen, progress_queue))
        async_results.append(r)

    finished_tasks = 0

    # Single tqdm bar for all frames
    with tqdm(total=total_frames, desc="Overall Progress") as pbar:
        while finished_tasks < len(async_results):
            # Drain progress queue
            while not progress_queue.empty():
                n = progress_queue.get()  # each .put(1) means 1 frame processed
                pbar.update(n)

            # Check how many tasks have finished
            finished_tasks = sum(r.ready() for r in async_results)
            time.sleep(0.05)  # small sleep to reduce CPU usage

    processed_clip_paths = [r.get() for r in async_results]

    pool.close()
    pool.join()

    print("All clips processed. Now merging them into final video...")

    # Merge the processed clips in the same order as tasks
    #    Because tasks were appended in the order of scene_1's clips, then scene_2's clips, etc.
    final_out_path = os.path.join(args.save_folder, f"{base_filename}_output.mp4")
    video_utils.merge_segments(processed_clip_paths, final_out_path)

    print(f"Done! Final result is at: {final_out_path}")
