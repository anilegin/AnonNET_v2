import os
import argparse
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision
from multiprocessing import Process, Queue
from tqdm import tqdm
from PIL import Image
import numpy as np


import video_utils
from networks.generator import Generator

mp.set_start_method('spawn', force=True)
os.environ["PYTHONWARNINGS"] = "ignore"  # Suppress all warnings globally

################################################################################
# Preprocessing / Postprocessing helpers
################################################################################

def load_image(image, size):
    if isinstance(image, str): 
        img = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):  
        img = image.convert('RGB')
    elif isinstance(image, np.ndarray):  
        img = Image.fromarray(image).convert('RGB')
    else:
        raise ValueError("Unsupported input type. Provide a file path, PIL Image, or NumPy array.")
    
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # shape: (3, size, size)
    return img / 255.0

def img_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0,1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # shape: [1, 3, size, size]
    img = (img - 0.5) * 2.0  # [-1,1]
    return img

def vid_preprocessing(vid_path, size=256, max_duration=None):
    """
    Read the video, up to `max_duration` seconds if not None, resize to `size`, normalize to [-1,1].
    Returns (vid_norm, fps).
    """
    vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
    vid = vid_dict[0]  # shape: (T, H, W, C)
    fps = vid_dict[2]['video_fps']

    if max_duration is not None:
        max_frames = int(fps * max_duration)
        vid = vid[:max_frames]

    vid_resized = torch.nn.functional.interpolate(
        vid.permute(0, 3, 1, 2).float(),  # (T,C,H,W)
        size=(size, size),
        mode='bilinear',
        align_corners=False
    )

    # Normalize to [-1,1], and add batch dimension => (1, T, C, H, W)
    vid_resized = vid_resized.unsqueeze(0)
    vid_norm = (vid_resized / 255.0 - 0.5) * 2.0
    return vid_norm, fps

def save_video(vid_target_recon, save_path, fps):
    """
    Take a video tensor in shape (1, T, 3, H, W) in [-1,1], and save it as MP4.
    """
    vid = vid_target_recon.permute(0, 2, 3, 4, 1)  # => (1, T, H, W, C)
    vid = vid.clamp(-1, 1).cpu()
    # Scale to [0,255]
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).byte()
    torchvision.io.write_video(save_path, vid[0], fps=fps)

################################################################################
# Model initialization & Parallel Processing
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
    gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
    weight = torch.load(ckpt, map_location=lambda storage, loc: storage)['gen']
    gen.load_state_dict(weight)
    gen.eval()
    return gen

def process_segment(gen, args, source_tensor, segment_path, save_path, progress_queue):
    """
    Run inference on a single 15-second (or shorter) segment in a child process.
    """
    vid_target, fps = vid_preprocessing(segment_path, size=args.size, max_duration=None)
    vid_target = vid_target.cuda()  # shape: (1,T,C,H,W)
    T = vid_target.size(1)

    if args.model == 'ted':
        h_start = None
    else:
        h_start = gen.enc.enc_motion(vid_target[:, 0])

    frames = []
    with torch.no_grad():
        for i in range(T):
            driving_frame = vid_target[:, i]  # shape: (1,3,H,W)
            out_frame = gen(source_tensor, driving_frame, h_start)
            frames.append(out_frame.unsqueeze(2))  # shape => (1, 3, 1, H, W)
            progress_queue.put(1)  # update 1 frame processed

    # Concatenate over time dimension => (1,3,T,H,W)
    vid_out = torch.cat(frames, dim=2)
    save_video(vid_out, save_path, fps)

def process_subsegments_in_parallel(gen, args, source_tensor, chunk_path, output_path):
    """
    For a given chunk (e.g., 1-minute chunk):
      1) Split into 15-second sub-segments.
      2) Parallel-process each sub-segment -> produce processed_X.mp4
      3) Merge processed sub-segments -> output_path
      4) Clean up (remove sub-segments and their processed results).
    """
    temp_dir = os.path.join(args.save_folder, "temp_subsegments")
    os.makedirs(temp_dir, exist_ok=True)

    # Split chunk into 15-sec subsegments
    video_utils.split_video_by_seconds(chunk_path, args.segment_length, temp_dir)
    
    segments = sorted([
        os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".mp4")
    ])
    if not segments:
        print(f"No sub-segments found in {temp_dir}, skipping.")
        return

    total_frames = 0
    for seg in segments:
        info = torchvision.io.read_video(seg, pts_unit='sec')
        total_frames += info[0].shape[0]

    # Spawn processes for each sub-segment
    progress_queue = Queue()
    processes = []
    processed_segment_paths = []
    for idx, seg in enumerate(segments):
        processed_seg_path = os.path.join(temp_dir, f"processed_{idx}.mp4")
        p = Process(
            target=process_segment,
            args=(gen, args, source_tensor, seg, processed_seg_path, progress_queue)
        )
        p.start()
        processes.append(p)
        processed_segment_paths.append(processed_seg_path)

    # Total progress
    with tqdm(total=total_frames, desc=f"Processing {os.path.basename(chunk_path)}") as pbar:
        while any(p.is_alive() for p in processes):
            while not progress_queue.empty():
                pbar.update(progress_queue.get())

    for p in processes:
        p.join()
        # Drain any leftover progress signals
        while not progress_queue.empty():
            pbar.update(progress_queue.get())

    # Merge processed sub-segments into final `output_path`
    video_utils.merge_segments(processed_segment_paths, output_path)

    # Remove the sub-segments + processed sub-segments
    for f in segments + processed_segment_paths:
        if os.path.exists(f):
            os.remove(f)

    # Also remove temp_dir itself if you like (be cautious if multiple chunks share it)
    # os.rmdir(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--model", type=str, choices=['vox', 'taichi', 'ted'], default='vox')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--source_path", type=str, default='source.jpg')
    parser.add_argument("--driving_path", type=str, default='driving.mp4')
    parser.add_argument("--save_folder", type=str, default='./res_presentation')
    parser.add_argument("--segment_length", type=int, default=15, help="Sub-segment length in seconds")
    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)

    gen = init_generator(args)
    source_tensor = img_preprocessing(args.source_path, args.size).cuda()

    # Split the ENTIRE video into ~1-minute chunks (or last chunk is leftover)
    chunk_dir = os.path.join(args.save_folder, "minute_chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    # "one chunk" = 60 seconds
    chunk_length = 60.0  
    video_utils.split_video_by_seconds(args.driving_path, chunk_length, chunk_dir)

    chunks = sorted([
        os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith('.mp4')
    ])

    # Process each 1-minute chunk:
    processed_chunks = []
    for i, chunk_path in enumerate(chunks):
        chunk_out_path = os.path.join(args.save_folder, f"processed_chunk_{i}.mp4")

        # Process chunk by splitting into 15s sub-segments & merging
        process_subsegments_in_parallel(gen, args, source_tensor, chunk_path, chunk_out_path)

        processed_chunks.append(chunk_out_path)
        # Remove the original chunk
        if os.path.exists(chunk_path):
            os.remove(chunk_path)

    # Merge all processed chunks into one final video
    final_out_path = os.path.join(args.save_folder, "final_merged_output.mp4")
    video_utils.merge_segments(processed_chunks, final_out_path)

    # Remove each processed chunk
    for pc in processed_chunks:
        if os.path.exists(pc):
            os.remove(pc)

    print(f"Done! Final result is at: {final_out_path}")
