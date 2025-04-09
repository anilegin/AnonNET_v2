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
# from AnonHead.segment_multiple import Segment
# from AnonHead.predict_multiple_voxceleb import Predictor
# from Generation.utils.head_pose import get_frontal_frame



# -------------------
# 1) INPUT / OUTPUT PATHS
# -------------------

celebv_train = '/data/stars/share/CelebV-HQ/path_to_videos/train'
celebv_source = '/data/stars/share/CelebV_HQ_Anon/train_anon_source'
celebv_anon = '/data/stars/share/CelebV_HQ_Anon/train_anon_final_im_abl2'
output_folder = '/data/stars/share/CelebV_HQ_Anon/train_anon_videos'




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
    
    # from anonymize import main as anonymize
    from AnonHead.segment_multiple import Segment
    from AnonHead.predict_multiple_voxceleb import Predictor
    from Generation.utils.head_pose import get_frontal_frame
    
    all_videos = get_all_videos(celebv_train)
    # all_videos.extend(get_all_videos(celeba_male_val))
    
    # if args.even == 1:
    #     all_videos = all_videos[0::10]
    # elif args.even == 2:
    #     all_videos = all_videos[1::10]
    # elif args.even == 3:
    #     all_videos = all_videos[2::10]
    # elif args.even == 4:
    #     all_videos = all_videos[3::10]
    # elif args.even == 5:
    #     all_videos = all_videos[4::10]
    # elif args.even == 6:
    #     all_videos = all_videos[5::10]
    # elif args.even == 7:
    #     all_videos = all_videos[6::10]
    # elif args.even == 8:
    #     all_videos = all_videos[7::10]
    # elif args.even == 9:
    #     all_videos = all_videos[8::10]
    # elif args.even == 10:
    #     all_videos = all_videos[9::10]
    
    if args.even == 1:
        all_videos = all_videos[0::5]
    elif args.even == 2:
        all_videos = all_videos[1::5]
    elif args.even == 3:
        all_videos = all_videos[2::5]
    elif args.even == 4:
        all_videos = all_videos[3::5]
    elif args.even == 5:
        all_videos = all_videos[4::5]

    
    valid_paths_list = filter_processed_videos(all_videos, celebv_anon)
    
    print(f"Found {len(valid_paths_list)} videos.")
    
    
    # Start timer
    start_time = time.time()
    
    # ---------------------------------------------------
    # 0) Initialize logging
    # ---------------------------------------------------
    log_file = os.path.join('/home/aegin/projects/anonymization/AnonNET/stats_and_jobs_celebv_image2', f"processing_summary_celebv_{args.even}.txt")
    
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
    last_time_checkpoint = start_time

    for idx, vid_path in enumerate(tqdm(valid_paths_list, desc="Anon_video")):
        try:
            base_name = os.path.basename(vid_path)
            video_name = os.path.splitext(base_name)[0]
            src_path = os.path.join(celebv_source, f"{video_name}_source.png")
            
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
            # output = segment.annotate_face(frame_pil, fill=True)  
            
            outputs = segment.retinaface_detect_and_annotate(
                img=src_path,
                margin=0.8,
                method='face'
            )
            
            if len(outputs)==0:
                print(f"No faces found in {src_path} (or detection failed). Skipping.")
                out_image = Image.open(src_path)
                save_p = os.path.join(celebv_anon, f"{video_name}_anon_nodet.png")
                out_image.save(save_p)
                continue
            
            if len(outputs) > 1:
                # Calculate areas for each detection (width * height)
                areas = []
                for output in outputs:
                    mask_pil, cropped_img_pil, (x1, x2, y1, y2) = output
                    width = x2 - x1
                    height = y2 - y1
                    areas.append(width * height)
                
                # Get index of largest detection
                largest_idx = areas.index(max(areas))
                # Keep only the largest detection
                outputs = [outputs[largest_idx]]
                print(f"Selected largest face detection (kept 1 of {len(areas)} faces)")
            
            torch.cuda.empty_cache()

            # Anonymization
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

            generated = []
            dist = 0.0
            attrs = []

            # conf["mask"] = output
            # crop, dist, attrs = predictor.anonymize(**conf)
            # if crop is None:
            #     continue
                
            for i, (mask_img, seg_img, bbox) in enumerate(outputs):
                conf["mask"] = mask_img
                conf["image"] = seg_img

                crop, dist, attrs = predictor.anonymize(**conf)
                if crop is None:
                    continue

                # Merge the new face back in
                resized_mask = mask_img.resize(seg_img.size, Image.LANCZOS)
                resized_crop = crop.resize(seg_img.size, Image.LANCZOS)
                generated.append((resized_mask, resized_crop, bbox))

                # If you want only the first face, break here
                break

            # 4.4) Merge the face anonymization back into the original frame
            out_image = segment.merge_crops(frame_pil, generated)

            out_image = out_image.resize(org_size, Image.LANCZOS)
            dist_str = f"{dist:.3f}"
            attrs_str = "_".join(a.replace(" ", "-") for a in attrs)
            filename_info = f"{dist_str}_{attrs_str}" if attrs_str else dist_str
            save_p = os.path.join(celebv_anon, f"{video_name}_anon_{filename_info}.png")
            out_image.save(save_p)

            processed_count += 1
            torch.cuda.empty_cache()

            # ---------------------------------------------------
            # Additional Logging Every 5 Hours and Every 1k Videos
            # ---------------------------------------------------
            current_time = time.time()
            elapsed_since_last_log = current_time - last_time_checkpoint
            hours_passed = elapsed_since_last_log / 3600

            # Every 5 hours: log current index to the log file
            if hours_passed >= 5:
                with open(log_file, "a") as f:
                    f.write(f"[{time.ctime()}] ~{processed_count} videos processed so far (every 5-hour checkpoint).\n")
                last_time_checkpoint = current_time  # reset timer

            # Every 1000 videos: log milestone
            if processed_count > 0 and processed_count % 1000 == 0:
                with open(log_file, "a") as f:
                    f.write(f"[{time.ctime()}] {processed_count//1000}k batch done.\n")

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


# ---------------------------------------------------
# Animate
# ---------------------------------------------------

def get_anon_filepath(video_name, anon_output_folder):
    # This pattern excludes files containing _nodet or _detection
    pattern = os.path.join(
        anon_output_folder, 
        f"*{video_name}*_anon_*[!nodet!detection]*.png"
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def animate(args):
    from LivePortrait.inference import main as motion_lp
    from LivePortrait.src.config.argument_config import ArgumentConfig
    from LivePortrait.src.config.inference_config import InferenceConfig
    from LivePortrait.src.config.crop_config import CropConfig
    from LivePortrait.src.live_portrait_pipeline import LivePortraitPipeline
    
    all_videos = get_all_videos(celebv_train)
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
    
    valid_paths_list = filter_processed_videos(all_videos, output_folder, file_type='.mp4')
    
    print(f"Found {len(valid_paths_list)} videos.")
    
    # Start timer
    start_time = time.time()
    
    # ---------------------------------------------------
    # 0) Initialize logging
    # ---------------------------------------------------
    log_file = os.path.join('/home/aegin/projects/anonymization/AnonNET/stats_and_jobs_celebv_video', f"processing_video_celebv_{args.even}.txt")
    
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
            
            anon_im = get_anon_filepath(video_name, celebv_anon)
            if anon_im is None:  # Check if anonymized image exists
                print(f"No anonymized image found for {video_name}")
                continue
            
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
    #animate(args)