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


from utils import video_utils
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
    # Convert fps to a native Python integer (most videos have integer fps)
    fps_int = int(round(float(fps)))
    torchvision.io.write_video(save_path, vid[0], fps=fps_int)

    
def load_source_tensor_from_temp(scene_id, args):
    """
    Load the source image for the given scene_id from the temporary source images folder,
    process it using img_preprocessing, and return the source tensor on GPU.
    """
    import os
    source_image_path = os.path.join("temp_source_images", f"source_{scene_id}.jpg")
    if not os.path.exists(source_image_path):
        print(f"Source image for scene {scene_id} not found, using default {args.source_path}")
        source_image_path = args.source_path
    tensor = img_preprocessing(source_image_path, args.size).cuda()
    return tensor

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
    # for f in segments + processed_segment_paths:
    #     if os.path.exists(f):
    #         os.remove(f)

    # Also remove temp_dir itself if you like (be cautious if multiple chunks share it)
    # os.rmdir(temp_dir)
    
    
def process_scene_video(gen, args, scene_video_path, source_tensor, scene_id):
    """
    Process one scene video (previously extracted from the driving video).
    This function splits the scene video into 60‑second chunks (or uses the entire
    scene if it is shorter than the chunk length), processes each chunk using the 
    existing pipeline (splitting further into 15‑sec subsegments), and then merges the
    processed chunks into one processed scene video.
    
    Returns the file path of the processed scene video.
    """
    import shutil  # for copying short videos
    # Create a temporary directory for chunks for this scene
    scene_chunk_dir = os.path.join(args.save_folder, f"temp_chunks_scene_{scene_id}")
    os.makedirs(scene_chunk_dir, exist_ok=True)
    
    # Get the duration of the scene video.
    duration = video_utils.get_video_length_seconds(scene_video_path)
    
    if duration < args.chunk_length:
        # If the scene video is shorter than the chunk length, simply copy it as a single chunk.
        segment_path = os.path.join(scene_chunk_dir, "segment_0.mp4")
        shutil.copy(scene_video_path, segment_path)
        chunks = [segment_path]
    else:
        # Otherwise, split the scene video into chunks of args.chunk_length seconds.
        video_utils.split_video_by_seconds(scene_video_path, args.chunk_length, scene_chunk_dir)
        chunks = sorted([os.path.join(scene_chunk_dir, f) for f in os.listdir(scene_chunk_dir) if f.endswith('.mp4')])
    
    processed_chunks = []
    for i, chunk_path in enumerate(chunks):
        chunk_out_path = os.path.join(args.save_folder, f"processed_scene_{scene_id}_chunk_{i}.mp4")
        process_subsegments_in_parallel(gen, args, source_tensor, chunk_path, chunk_out_path)
        processed_chunks.append(chunk_out_path)
        # Remove the original chunk after processing
        # if os.path.exists(chunk_path):
        #     os.remove(chunk_path)
    
    # Merge the processed chunks into one video for the scene.
    processed_scene_path = os.path.join(args.save_folder, f"processed_scene_{scene_id}.mp4")
    video_utils.merge_segments(processed_chunks, processed_scene_path)
    
    # Clean up processed chunk files.
    # for pc in processed_chunks:
    #     if os.path.exists(pc):
    #         os.remove(pc)
    
    return processed_scene_path


if __name__ == "__main__":
    
    from utils.scene_detection import detect_scenes_and_assign_ids

    # (Assume your core functions like load_image, img_preprocessing, vid_preprocessing, 
    # save_video, init_generator, process_segment, process_subsegments_in_parallel are defined above.)

    # Also assume that detect_scenes_and_assign_ids is already defined and imported.

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
    parser.add_argument("--chunk_length", type=int, default=60, help="Chunk length in seconds")
    # Scene detection parameters (if needed)
    parser.add_argument("--scene_threshold", type=float, default=0.2)
    parser.add_argument("--scene_similarity_threshold", type=float, default=5.0)
    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)

    gen = init_generator(args)

    # 1. Run scene detection on the entire driving video.
    print("Detecting scenes in driving video...")
    scenes = detect_scenes_and_assign_ids(args.driving_path, threshold=args.scene_threshold,
                                          similarity_threshold=args.scene_similarity_threshold)
    print(f"Detected {len(scenes)} scenes.")

    # 2. Create temporary directories for scene videos and source images.
    temp_driving_dir = os.path.join(args.save_folder, "temp_driving")
    os.makedirs(temp_driving_dir, exist_ok=True)
    temp_source_dir = os.path.join("temp_source_images")
    os.makedirs(temp_source_dir, exist_ok=True)

    # 3. For each scene, extract its subvideo and, for the first occurrence of each scene_id,
    #    extract the first frame and save as a source image.
    scene_video_paths = []  # list of tuples (scene_id, scene_video_path)
    processed_scene_ids = set()
    for idx, scene in enumerate(scenes):
        scene_id = scene["scene_id"]
        scene_video_path = os.path.join(temp_driving_dir, f"scene_{idx}.mp4")
        # Extract subvideo for the scene from the driving video.
        video_utils.extract_subvideo(args.driving_path, scene_video_path, scene["start"], scene["end"])
        scene_video_paths.append((scene_id, scene_video_path))
        # For the first occurrence of this scene_id, extract the first frame and save it.
        if scene_id not in processed_scene_ids:
            processed_scene_ids.add(scene_id)
            source_image_path = os.path.join(temp_source_dir, f"source_{scene_id}.jpg")
            video_utils.extract_first_frame(scene_video_path, source_image_path)

    # 4. Process each scene video using its corresponding source image.
    processed_scene_videos = []
    for scene_id, scene_video_path in scene_video_paths:
        print(f"Processing scene {scene_id} from video {scene_video_path}...")
        # Load the source tensor from the temp source image folder.
        source_tensor = load_source_tensor_from_temp(scene_id, args)
        # Process this scene video (split into chunks, process subsegments, merge chunks).
        processed_scene_video = process_scene_video(gen, args, scene_video_path, source_tensor, scene_id)
        processed_scene_videos.append(processed_scene_video)
        # Optionally, remove the temporary scene video.
        # if os.path.exists(scene_video_path):
        #     os.remove(scene_video_path)

    # 5. Merge all processed scene videos (in order) into the final output video.
    final_out_path = os.path.join(args.save_folder, "final_merged_output.mp4")
    video_utils.merge_segments(processed_scene_videos, final_out_path)

    # Optionally, remove the temporary scene videos.
    # for ps in processed_scene_videos:
    #     if os.path.exists(ps):
    #         os.remove(ps)

    print(f"Done! Final result is at: {final_out_path}")

