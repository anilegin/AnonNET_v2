#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, Queue

# --------------------------------------------------------------------------
#  Import your custom modules.
#  - video_utils: Contains splitting, merging, first-frame extraction, etc.
#  - networks.generator: Your Generator model definition
#  - detect_scenes_and_assign_ids: The scene detection function
# --------------------------------------------------------------------------
from utils import video_utils
from networks.generator import Generator
from utils.scene_detection import detect_scenes_and_assign_ids

# Ensure "spawn" start method for multiprocessing on some OS's
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

os.environ["PYTHONWARNINGS"] = "ignore"  # Suppress all warnings globally

################################################################################
# Preprocessing / Postprocessing Helpers
################################################################################

def load_image(image, size):
    """
    Load an image (file path, PIL Image, or np.ndarray),
    convert to (3, size, size) in [0,1].
    """
    if isinstance(image, str):
        img = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        img = image.convert('RGB')
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert('RGB')
    else:
        raise ValueError("Unsupported input type for load_image().")
    
    img = img.resize((size, size))
    img = np.asarray(img)
    # (H,W,C) => (C,H,W)
    img = np.transpose(img, (2, 0, 1))
    return img / 255.0

def img_preprocessing(img_path, size):
    """
    Load an image from img_path, resize to (size, size), normalize to [-1,1].
    Returns a torch tensor of shape (1, 3, size, size).
    """
    img = load_image(img_path, size)  # => (3, size, size) in [0,1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # => (1,3,size,size)
    img = (img - 0.5) * 2.0  # => [-1,1]
    return img

def vid_preprocessing(vid_path, size=256):
    """
    Read the entire video at `vid_path`, resize to (size,size), normalize to [-1,1].
    Returns (video_tensor, fps).
       video_tensor: (1, T, 3, H, W) in [-1,1]
       fps: frames per second as a float
    """
    vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
    vid = vid_dict[0]   # shape: (T, H, W, C)
    fps = vid_dict[2]['video_fps']

    # Resize frames
    # (T,H,W,C) => (T,C,H,W)
    vid_torch = vid.permute(0, 3, 1, 2).float()
    vid_resized = torch.nn.functional.interpolate(
        vid_torch, size=(size, size), mode='bilinear', align_corners=False
    )
    # Normalize to [-1,1], and add batch dimension => (1,T,C,H,W)
    vid_resized = vid_resized.unsqueeze(0) / 255.0
    vid_norm = (vid_resized - 0.5) * 2.0
    return vid_norm, fps

def save_video(video_tensor, save_path, fps):
    """
    Save a video tensor of shape (1, T, 3, H, W) in [-1,1] to MP4 at `save_path`.
    """
    # (1,T,3,H,W) => (1,T,H,W,C)
    vid = video_tensor.permute(0, 2, 3, 4, 1).clamp(-1, 1).cpu()
    # Scale to [0,255] and convert to byte
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).byte()
    # Cast fps to int if needed
    fps_int = int(round(float(fps)))
    # Write
    torchvision.io.write_video(save_path, vid[0], fps=fps_int)

def load_source_tensor_from_temp(scene_id, size, default_source=None):
    """
    Load the source image for a given scene_id from ./temp_source_images/source_<scene_id>.jpg
    If not found, fall back to `default_source` if given.
    Returns a torch tensor on CUDA, shape: (1,3,size,size).
    """
    temp_source_path = os.path.join("temp_source_images", f"source_{scene_id}.jpg")
    if os.path.exists(temp_source_path):
        tensor = img_preprocessing(temp_source_path, size).cuda()
        return tensor
    elif default_source is not None and os.path.exists(default_source):
        # fallback
        return img_preprocessing(default_source, size).cuda()
    else:
        raise FileNotFoundError(
            f"No source image for scene_id={scene_id} in temp_source_images, "
            "and no valid fallback source provided."
        )

################################################################################
# Model initialization & Processing
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
    Run inference on the entire clip at `segment_path`. 
    - segment_path: path to e.g. a 20s sub-video
    - save_path: output path for the processed sub-video
    - source_tensor: (1,3,H,W) in [-1,1]
    - progress_queue: optional to track progress
    """
    vid_target, fps = vid_preprocessing(segment_path, size=args.size)
    vid_target = vid_target.cuda()  # shape: (1,T,3,H,W)
    T = vid_target.size(1)

    # If using a specific model, possibly compute initial hidden states:
    if args.model == 'ted':
        h_start = None
    else:
        # e.g. motion encoder
        h_start = gen.enc.enc_motion(vid_target[:, 0])  # shape: ?

    frames = []
    with torch.no_grad():
        for i in range(T):
            driving_frame = vid_target[:, i]  # shape: (1,3,H,W)
            out_frame = gen(source_tensor, driving_frame, h_start)  # shape: (1,3,H,W)
            frames.append(out_frame.unsqueeze(2))  # => (1,3,1,H,W)
            if progress_queue is not None:
                progress_queue.put(1)

    # Concatenate over time => (1,3,T,H,W)
    vid_out = torch.cat(frames, dim=2)
    # Save
    save_video(vid_out, save_path, fps)

################################################################################
# New "Clips in a Queue" Approach
################################################################################

def create_clip_tasks(scene_video_paths, clip_length):
    """
    For each (scene_id, scene_video_path), split into clip_length-second sub-videos.
    Return a list of tasks (clip_path, scene_id, output_clip_path),
    preserving the order: all clips from scene_1, then scene_2, etc.
    """
    tasks = []
    for scene_id, scene_video_path in scene_video_paths:
        # A subfolder for clips of this scene
        clip_dir = os.path.join(os.path.dirname(scene_video_path), f"clips_scene_{scene_id}")
        os.makedirs(clip_dir, exist_ok=True)
        # Split
        video_utils.split_video_by_seconds(scene_video_path, clip_length, clip_dir)
        clips = sorted([os.path.join(clip_dir, f) for f in os.listdir(clip_dir) if f.endswith('.mp4')])
        for i, clip in enumerate(clips):
            out_path = os.path.join(clip_dir, f"processed_clip_{i}.mp4")
            tasks.append((clip, scene_id, out_path))
    return tasks

def process_clip_worker(task, args, gen):
    """
    A worker function for multiprocessing:
      task = (clip_path, scene_id, output_clip_path)
    """
    clip_path, scene_id, output_clip_path = task
    # Load the scene-specific source
    source_tensor = load_source_tensor_from_temp(
        scene_id,
        args.size,
        default_source=None  # or fallback if you have a default
    )
    # No real progress queue here, so pass None
    process_segment(gen, args, source_tensor, clip_path, output_clip_path, progress_queue=None)
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
    # No --source_path argument here, since we create source images dynamically
    parser.add_argument("--driving_path", type=str, default="driving.mp4")
    parser.add_argument("--save_folder", type=str, default="./res_presentation")
    parser.add_argument("--clip_length", type=int, default=20, help="Each scene is split into sub-videos of this length.")
    parser.add_argument("--scene_threshold", type=float, default=0.2)
    parser.add_argument("--scene_similarity_threshold", type=float, default=5.0)
    # You can choose how many parallel workers to use
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel processes")
    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)

    # 1. Load the model (Generator)
    gen = init_generator(args)

    # 2. Detect Scenes
    print("Detecting scenes in driving video...")
    scenes = detect_scenes_and_assign_ids(
        args.driving_path,
        threshold=args.scene_threshold,
        similarity_threshold=args.scene_similarity_threshold
    )
    print(f"Detected {len(scenes)} scenes.")

    # 3. For each scene, extract subvideo + first-frame source image
    temp_driving_dir = os.path.join(args.save_folder, "temp_driving")
    os.makedirs(temp_driving_dir, exist_ok=True)
    temp_source_dir = "temp_source_images"
    os.makedirs(temp_source_dir, exist_ok=True)

    scene_video_paths = []  # list of (scene_id, path_to_scene_video)
    processed_ids = set()
    for idx, scene_info in enumerate(scenes):
        scene_id = scene_info["scene_id"]
        start = scene_info["start"]
        end = scene_info["end"]

        # Extract subvideo
        scene_vid_path = os.path.join(temp_driving_dir, f"scene_{idx}.mp4")
        video_utils.extract_subvideo(args.driving_path, scene_vid_path, start, end)
        scene_video_paths.append((scene_id, scene_vid_path))

        # Extract first frame if this is the first occurrence of that scene_id
        if scene_id not in processed_ids:
            processed_ids.add(scene_id)
            source_jpg = os.path.join(temp_source_dir, f"source_{scene_id}.jpg")
            video_utils.extract_first_frame(scene_vid_path, source_jpg)

    # 4. Create tasks for each 20s clip in each scene
    tasks = create_clip_tasks(scene_video_paths, args.clip_length)
    print(f"Created {len(tasks)} clip tasks (scenes -> 20s clips).")

    # 5. Multiprocessing: process each clip
    from multiprocessing import Pool
    pool = Pool(processes=args.num_workers)
    results = pool.map(partial(process_clip_worker, args=args, gen=gen), tasks)
    pool.close()
    pool.join()

    # 6. Merge all processed clips in order => final output
    #    results is a list of processed clip paths, in the same order as tasks
    final_out_path = os.path.join(args.save_folder, "final_merged_output.mp4")
    video_utils.merge_segments(results, final_out_path)

    print(f"Done! Final result is at: {final_out_path}")
