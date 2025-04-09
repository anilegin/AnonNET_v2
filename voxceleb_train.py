import os
import torch
import numpy as np
from PIL import Image
import argparse
import cv2
import shutil
from tqdm import tqdm 

import time
# from anonymize import main as anonymize
from AnonHead.segment_multiple import Segment
from AnonHead.predict_multiple_voxceleb import Predictor
from Generation.utils.head_pose import get_frontal_frame

# from LivePortrait.inference import main as motion_lp
# from AnonHead.predict_multiple_voxceleb import Predictor
# from LivePortrait.src.config.argument_config import ArgumentConfig
# from LivePortrait.src.config.inference_config import InferenceConfig
# from LivePortrait.src.config.crop_config import CropConfig
# from LivePortrait.src.live_portrait_pipeline import LivePortraitPipeline
# from LivePortrait.inference import main as motion_lp

import glob

vox_files = '/data/stars/share/vox-celeb-dataset/vox-celeb-50000'
vox_route = '/data/stars/share/vox-celeb-anon/anon_im'
vox_source = '/data/stars/share/vox-celeb-anon/source_im'
vox_video = "/data/stars/share/vox-celeb-anon/anon_videos"

def get_valid_paths(txt_file):
    """
    Reads a text file containing one path per line, strips whitespace,
    ignores empty lines, and checks if each path exists.

    Args:
        txt_file (str): Path to the text file containing paths.
    Returns:
        List[str]: A list of valid, existing paths.
    """
    valid_paths = []
    
    with open(txt_file, 'r') as f:
        for line in f:
            # Strip leading/trailing whitespace
            path = line.strip()
            
            full_path = os.path.join(vox_files, path + ".mp4")
            # Skip if empty
            if not path:
                continue
            
            # Check if the path is valid
            if os.path.exists(full_path):
                valid_paths.append(full_path)
            else:
                print(f"Skipping invalid path: {full_path}")
    
    return valid_paths

def filter_processed_videos(video_paths, output_root, file_type='.png'):
    """
    Filters out videos that already have an anonymized image saved.
    Args:
        video_paths (List[str]): List of video file paths to process.
        output_root (str): Root directory where anonymized images are saved.
        file_type (str): File extension to search for.
    Returns:
        List[str]: Filtered list of video paths that haven't been processed.
    """
    valid_videos = []

    for vid_path in video_paths:
        base_name = os.path.basename(vid_path)       # e.g., "abc123.mp4"
        video_name = os.path.splitext(base_name)[0]  # e.g., "abc123"

        # Extract relative path (vid_id) from original path
        parts = vid_path.split(os.sep)
        try:
            base_index = parts.index("vox-celeb-50000") + 1
            vid_id = os.path.join(*parts[base_index:-1])  # skip "vox-celeb-50000" and filename
        except ValueError:
            print(f"Warning: 'vox-celeb-50000' not in path: {vid_path}")
            valid_videos.append(vid_path)
            continue

        vid_folder = os.path.join(output_root, vid_id)
        pattern = os.path.join(vid_folder, f"*{video_name}*_anon_*{file_type}")
        matching_files = glob.glob(pattern)

        if not matching_files:
            valid_videos.append(vid_path)
        # else:
        #     print(f"Skipping {vid_path}, already processed.")

    return valid_videos


@torch.no_grad()
def main(args):
    
    txt_file = "/data/stars/share/vox-celeb-dataset/vox-celeb-50000/file.list"  # replace with your actual text file
    all_videos = get_valid_paths(txt_file)
    

    # if args.even == 1:
    #     all_videos = all_videos[0::5]
    # elif args.even == 2:
    #     all_videos = all_videos[1::5]
    # elif args.even == 3:
    #     all_videos = all_videos[2::5]
    # elif args.even == 4:
    #     all_videos = all_videos[3::5]
    # elif args.even == 5:
    #     all_videos = all_videos[4::5]
    
    
    
    valid_paths_list = filter_processed_videos(all_videos,vox_route)
    
    print(f"Found {len(valid_paths_list)} videos.")
    
    # Start timer
    start_time = time.time()
    
    # ---------------------------------------------------
    # 0) Initialize logging
    # ---------------------------------------------------
    log_file = os.path.join('/home/aegin/projects/anonymization/AnonNET/stats_and_jobs_voxceleb', f"processing_summary_voxceleb_{args.even}.txt")
    
    # Get GPU info
    gpu_info = {
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else "N/A",
        'total_memory': f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB" 
                       if torch.cuda.is_available() else "N/A"
    }
    
    # Write initial info to log file
    with open(log_file, "w") as f:
        f.write(f"Anonymization Processing Log\n{'='*50}\n")
        f.write(f"Started at: {time.ctime()}\n\n")
        f.write("System Configuration:\n")
        f.write(f"- GPU: {gpu_info['device']}\n")
        f.write(f"- CUDA: {gpu_info['cuda_version']}\n")
        f.write(f"- Total VRAM: {gpu_info['total_memory']}\n\n")
        f.write(f"Processing {len(valid_paths_list)} videos\n")
        f.write("="*50 + "\n")

    
    segment = Segment(load_det=False, load_seg=False)
    predictor = Predictor()

    # ---------------------------------------------------
    # 4) Process each video
    # ---------------------------------------------------
    processed_count = 0
    for vid_path in tqdm(valid_paths_list, desc="Anon_video"):
        try:
            base_name = os.path.basename(vid_path)
            video_name = os.path.splitext(base_name)[0]
            # vid_folder = vox_route
            
            parts = vid_path.split(os.sep)  # Use os.sep for cross-platform compatibility
            try:
                base_index = parts.index("vox-celeb-50000") + 1
                vid_id = os.path.join(*parts[base_index:-1])  # Relative path without filename
            except ValueError:
                print(f"Error: 'vox-celeb-50000' not found in path: {vid_path}")
                continue  # Skip this file if structure is wrong
            
            
            # # Create directories if they don't exist
            vid_folder = os.path.join(vox_route, vid_id)
            os.makedirs(vid_folder, exist_ok=True)
            
            src_folder = os.path.join(vox_source, vid_id)
            os.makedirs(src_folder, exist_ok=True)
            
            # print(f"{vid_path} is processing. frontal is started")
            # src_path = os.path.join(src_folder, f"{video_name}_source.png")
        
            # Determine output path and check if it's already processed
            expected_output_pattern = os.path.join(vid_folder, f"{video_name}_anon_*.png")
            if glob.glob(expected_output_pattern):
                print(f"Skipping {vid_path}: already processed (existing file in {vid_folder})")
                continue

            print(f"{vid_path} is processing. Frontal extraction started.")
            src_path = os.path.join(src_folder, f"{video_name}_source.png")

            if not os.path.exists(src_path):
                frame = get_frontal_frame(vid_path, 0, 2, best=True)     
                print("Frontal frame found.")
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil.save(src_path)
            else:
                frame_pil = Image.open(src_path)

            org_size = frame_pil.size
            
            # Face detection and segmentation
            output = segment.annotate_face(frame_pil, fill=True)  

            torch.cuda.empty_cache()

            # Anonymization
            conf = {
                "image": frame_pil,
                "prompt": "",
                "mask": None,
                "negative_prompt": args.negative_prompt,
                "strength": [0.9, 0.4, 0.3],
                "max_height": args.max_height,
                "max_width": args.max_width,
                "steps": 35,
                "seed": args.seed,
                "guidance_scale": 8.0,
                "im_path": src_path
            }
            
            # conf = {
            #     "image": frame_pil,
            #     "prompt": "",
            #     "mask": None,
            #     "negative_prompt": args.negative_prompt,
            #     "strength": args.strength,
            #     "max_height": args.max_height,
            #     "max_width": args.max_width,
            #     "steps": args.steps,
            #     "seed": args.seed,
            #     "guidance_scale": args.guidance_scale,
            #     "im_path": src_path
            # }

            generated = []
            dist = 0.0
            attrs = []

            conf["mask"] = output
            crop, dist, attrs = predictor.anonymize(**conf)
            if crop is None:
                continue

            out_image = crop.resize(org_size, Image.LANCZOS)
            dist_str = f"{dist:.3f}"
            attrs_str = "_".join(a.replace(" ", "-") for a in attrs)
            filename_info = f"{dist_str}_{attrs_str}" if attrs_str else dist_str
            save_p = os.path.join(vid_folder, f"{video_name}_anon_{filename_info}.png")
            out_image.save(save_p)
            
            processed_count += 1
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {vid_path}: {e}")
            continue

    # ---------------------------------------------------
    # Final logging
    # ---------------------------------------------------
    total_time = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0
    
    with open(log_file, "a") as f:
        f.write("\nProcessing Results:\n")
        f.write(f"- Completed at: {time.ctime()}\n")
        f.write(f"- Total duration: {total_time/60:.2f} minutes\n")
        f.write(f"- Videos processed: {processed_count}/{len(valid_paths_list)}\n")
        if torch.cuda.is_available():
            f.write(f"- Peak GPU memory used: {peak_mem:.2f} GB\n")
        f.write("="*50 + "\n")

    print(f"\nProcessing complete. Results saved to {log_file}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Videos processed: {processed_count}/{len(valid_paths_list)}")
    if torch.cuda.is_available():
        print(f"Peak GPU memory used: {peak_mem:.2f} GB")
        
def get_anon_filepath(video_name, anon_output_folder):
    # This pattern excludes files containing _nodet or _detection
    pattern = os.path.join(
        anon_output_folder, 
        f"*{video_name}*_anon_*[!nodet!detection]*.png"
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None        
        
@torch.no_grad()        
def animate(args):
    from LivePortrait.inference import main as motion_lp
    from LivePortrait.src.config.argument_config import ArgumentConfig
    from LivePortrait.src.config.inference_config import InferenceConfig
    from LivePortrait.src.config.crop_config import CropConfig
    from LivePortrait.src.live_portrait_pipeline import LivePortraitPipeline
    
    txt_file = "/data/stars/share/vox-celeb-dataset/vox-celeb-50000/file.list"  # replace with your actual text file
    all_videos = get_valid_paths(txt_file)
    
    if args.even == 1:
        all_videos = all_videos[0::10]
    elif args.even == 2:
        all_videos = all_videos[1::10]
    elif args.even == 3:
        all_videos = all_videos[2::10]
    elif args.even == 4:
        all_videos = all_videos[3::10]
    elif args.even == 5:
        all_videos = all_videos[4::10]
    elif args.even == 6:
        all_videos = all_videos[5::10]
    elif args.even == 7:
        all_videos = all_videos[6::10]
    elif args.even == 8:
        all_videos = all_videos[7::10]
    elif args.even == 9:
        all_videos = all_videos[8::10]
    elif args.even == 10:
        all_videos = all_videos[9::10]    
    
    valid_paths_list = filter_processed_videos(all_videos,vox_video)
    
    print(f"Found {len(valid_paths_list)} videos.")
    
    # Start timer
    start_time = time.time()
    
    # ---------------------------------------------------
    # 0) Initialize logging
    # ---------------------------------------------------
    log_file = os.path.join('/home/aegin/projects/anonymization/AnonNET/stats_and_jobs_voxceleb_video', f"processing_summary_voxceleb_{args.even}.txt")
    
    # Get GPU info
    gpu_info = {
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else "N/A",
        'total_memory': f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB" 
                       if torch.cuda.is_available() else "N/A"
    }
    
    # Write initial info to log file
    with open(log_file, "w") as f:
        f.write(f"Anonymization Processing Log\n{'='*50}\n")
        f.write(f"Started at: {time.ctime()}\n\n")
        f.write("System Configuration:\n")
        f.write(f"- GPU: {gpu_info['device']}\n")
        f.write(f"- CUDA: {gpu_info['cuda_version']}\n")
        f.write(f"- Total VRAM: {gpu_info['total_memory']}\n\n")
        f.write(f"Processing {len(valid_paths_list)} videos\n")
        f.write("="*50 + "\n")
    
    processed_count = 0
    last_time_checkpoint = start_time

    for idx, vid_path in enumerate(tqdm(valid_paths_list, desc="Anon_video")):
        try:
            base_name = os.path.basename(vid_path)
            video_name = os.path.splitext(base_name)[0]
            # vid_folder = vox_route
            
            parts = vid_path.split(os.sep)  # Use os.sep for cross-platform compatibility
            try:
                base_index = parts.index("vox-celeb-50000") + 1
                vid_id = os.path.join(*parts[base_index:-1])  # Relative path without filename
            except ValueError:
                print(f"Error: 'vox-celeb-50000' not found in path: {vid_path}")
                continue  # Skip this file if structure is wrong
            
            
            # # Create directories if they don't exist
            vid_folder = os.path.join(vox_route, vid_id)
            os.makedirs(vid_folder, exist_ok=True)
            
            anon_path = os.path.join(vox_route,vid_id)
            
            anon_im = get_anon_filepath(video_name, anon_path)
            if anon_im is None:  # Check if anonymized image exists
                print(f"No anonymized image found for {video_name}")
                continue
            
            output_folder = os.path.join(vox_video,vid_id)
            
            args_lp = ArgumentConfig(
                source=anon_im,
                driving=vid_path,
                flag_stitching=True,
                output_dir=output_folder
            )
            motion_lp(args_lp)
            torch.cuda.empty_cache()
            
            processed_count += 1  # <-- THIS WAS MISSING
            
            # Logging every N files or hours
            current_time = time.time()
            elapsed_since_last_log = current_time - last_time_checkpoint
            hours_passed = elapsed_since_last_log / 3600

            if hours_passed >= 5:
                with open(log_file, "a") as f:
                    f.write(f"[{time.ctime()}] ~{processed_count} videos processed so far (every 5-hour checkpoint).\n")
                last_time_checkpoint = current_time

            if processed_count > 0 and processed_count % 1000 == 0:
                with open(log_file, "a") as f:
                    f.write(f"[{time.ctime()}] {processed_count//1000}k batch done.\n")
                    
        except Exception as e:
            print(f"Error processing {vid_path}: {e}")
            continue
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #anonymize
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--max_height", type=int, default=612)
    parser.add_argument("--max_width", type=int, default=612)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_force", action='store_false', default=True, help='Try RetinaFace in case YOLO fails')
    parser.add_argument("--no_fill", action='store_false', default=True, help='mask every part of the face no black region on face')
    parser.add_argument("--no_cache", action='store_false', default=True, help='no save of masks and anon images')
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--out_path", type=str, default="./anon")
    parser.add_argument("--out_mask", type=str, default='./anon/masks')
    parser.add_argument("--segment", type=str, choices=['head', 'face'], default='face')
    parser.add_argument("--load_det", action='store_true', default=False, help='Load detection again.')
    #generate
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--model", type=str, choices=['vox', 'taichi', 'ted'], default='vox')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--driving_path", type=str, default="driving.mp4")
    parser.add_argument("--source_path", type=str, default="./source_images")
    parser.add_argument("--save_folder", type=str, default="./results")
    parser.add_argument("--no_stitch", action='store_false', default=True, help='no stitching, use if head movement is a lot')
    parser.add_argument("--clip_length", type=int, default=20, help="Split scenes into 20s sub-videos.")
    parser.add_argument("--max_len", type=int, default=None, help="Max Duration of the video")
    parser.add_argument("--scene_threshold", type=float, default=0.2) #old one 0.2
    parser.add_argument("--scene_similarity_threshold", type=float, default=2.0)
    parser.add_argument("--num_workers", type=int, default=4, help="Multiprocessing worker count.")
    #motion model 
    parser.add_argument("--motion", type=str, choices=['lia', 'lp'], default='lp')
    parser.add_argument("--batch", type=str, default='0-300')
    parser.add_argument("--even", type=int, choices=[1,2,3,4,5,6,7,8,9,10,11,12,31], default=None)
    args = parser.parse_args()
    
    #main(args)
    animate(args)


