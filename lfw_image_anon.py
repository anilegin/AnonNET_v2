import os
import glob
import argparse
import torch
import shutil
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import time

# ========== Project-Specific Imports ==========
# Make sure these are correct based on your project's structure
# from anonymize import main as anonymize
from AnonHead.segment_multiple import Segment
from AnonHead.predict_multiple_voxceleb import Predictor
from Generation.utils.head_pose import get_frontal_frame



# -------------------
# 1) INPUT / OUTPUT PATHS
# -------------------

source_folder = '/data/stars/share/LFW/lfw_funneled'
output_folder = '/data/stars/share/LFW/lfw_anon'


def get_lfw_photos(root_folder):
    jpg_files = []

    # Walk through all directories and files
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.jpg'):
                full_path = os.path.join(root, file)
                jpg_files.append(full_path)
    return jpg_files

def get_all_videos(folder):
    """
    Returns a list of FULL paths to every .mp4 file in `folder`.
    Does NOT recurse into subfolders. If you need recursion,
    replace `os.listdir(folder)` with `os.walk(...)`.
    """
    all_mp4 = []
    for fname in os.listdir(folder):
        if fname.lower().endswith('.mp4'):
            all_mp4.append(os.path.join(folder, fname))
    return all_mp4


def filter_processed_videos(video_paths, output_folder,file_type='.png'):
    """
    Pre-filter the video_paths list by removing any video that already
    has an anonymized image (with '_anon_' in the name) in the output_folder.
    """
    valid_videos = []
    for vid_path in video_paths:
        base_name = os.path.basename(vid_path)      # e.g. "example.mp4"
        video_name = os.path.splitext(base_name)[0]  # e.g. "example"
        
        # Check for any file that contains both the video_name and '_anon_'
        pattern = os.path.join(output_folder, f"*{video_name}*_anon_*{file_type}")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            valid_videos.append(vid_path)
        # else:
        #     #print(f"Skipping {vid_path}: already processed (found {len(matching_files)} matching files).")
        #     continue
    return valid_videos



def main(args):
    """
    Simplified version that only logs:
    - Start/end times and total duration
    - Basic GPU specifications
    """
    
    all_videos = get_lfw_photos(source_folder)
    # all_videos.extend(get_all_videos(celeba_male_val))
    
    if args.even == 1:
        all_videos = all_videos[0::3]
    elif args.even == 2:
        all_videos = all_videos[1::3]
    elif args.even == 3:
        all_videos = all_videos[2::3]
    # elif args.even == 4:
    #     all_videos = all_videos[3::8]
    # elif args.even == 5:
    #     all_videos = all_videos[4::8]
    # elif args.even == 6:
    #     all_videos = all_videos[5::8]
    # elif args.even == 7:
    #     all_videos = all_videos[6::8]
    # elif args.even == 8:
    #     all_videos = all_videos[7::8]
    
    valid_paths_list = filter_processed_videos(all_videos, output_folder)
    
    print(f"Found {len(valid_paths_list)} videos.")
    
    
    # Start timer
    start_time = time.time()
    
    # ---------------------------------------------------
    # 0) Initialize logging
    # ---------------------------------------------------
    log_file = os.path.join('/home/aegin/projects/anonymization/AnonNET/stats_and_jobs_lfw', f"processing_summary_lfw_{args.even}.txt")
    
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


    # ---------------------------------------------------
    # 3) Initialize Segmenter and Anonymization Predictor
    # ---------------------------------------------------
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
            
            frame_pil = Image.open(vid_path)
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
                "strength": args.strength,
                "max_height": args.max_height,
                "max_width": args.max_width,
                "steps": 30, #past 20
                "seed": args.seed,
                "guidance_scale": 8.0,
                "im_path": vid_path
            }

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
            save_p = os.path.join(output_folder, f"{video_name}_anon_{filename_info}.png")
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




# -------------------
# 5) Command-Line Interface
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ---- Anonymization Args ----
    parser.add_argument("--image", type=str, help="Path to input image", default=None)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--max_height", type=int, default=612)
    parser.add_argument("--max_width", type=int, default=612)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_force", action='store_false', default=True,
                        help='Try RetinaFace if YOLO fails (or vice versa).')
    parser.add_argument("--no_fill", action='store_false', default=True,
                        help='Mask entire face (instead of blacking out).')
    parser.add_argument("--no_cache", action='store_false', default=True,
                        help='Do not save masks/anon images to cache.')
    parser.add_argument("--guidance_scale", type=float, default=10.0)

    # ---- Generation Args ----
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--model", type=str, choices=['vox', 'taichi', 'ted'], default='vox')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--driving_path", type=str, default="driving.mp4")
    parser.add_argument("--source_path", type=str, default="./source_images")
    parser.add_argument("--save_folder", type=str, default="./results")
    parser.add_argument("--no_stitch", action='store_false', default=True,
                        help='Disable stitching (for large head movements).')
    parser.add_argument("--clip_length", type=int, default=20,
                        help="Split scenes into N-second sub-videos.")
    parser.add_argument("--max_len", type=int, default=None,
                        help="Max duration of the video.")
    parser.add_argument("--scene_threshold", type=float, default=0.2)
    parser.add_argument("--scene_similarity_threshold", type=float, default=2.0)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Multiprocessing workers.")
    parser.add_argument("--motion", type=str, choices=['lia', 'lp'], default='lp')
    parser.add_argument("--even", type=int, choices=[1,2,3,4,5,6,7,8,9,10,11,12], default=None)
    # ---- Batch Arg ----
    parser.add_argument("--batch", type=str, default='0-300',
                        help="Range of videos to process (e.g. '0-300').")

    args = parser.parse_args()
    main(args)