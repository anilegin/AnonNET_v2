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
import math

# ========== Path Configurations ==========
hdtf_train = '/data/stars/share/HDTF-dataset/HDTF-RGB'
hdtf_source = '/data/stars/share/HDTF_anon/hdtf_source'
hdtf_anon = '/data/stars/share/HDTF_anon/hdtf_anon'
output_folder = '/data/stars/share/HDTF_anon/hdtf_video'
cropped_original = '/data/stars/share/HDTF_anon/hdtf_cropped'

def get_all_videos(folder):
    """Returns list of full paths to all .mp4 files in folder."""
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.mp4')]

def filter_processed_videos(video_paths, output_folder, file_type='.png'):
    """Filters out videos that already have output files."""
    valid_videos = []
    for vid_path in video_paths:
        base_name = os.path.basename(vid_path)
        video_name = os.path.splitext(base_name)[0]
        pattern = os.path.join(output_folder, f"*{video_name}*_anon_*{file_type}")
        if not glob.glob(pattern):
            valid_videos.append(vid_path)
    return valid_videos

def split_batches(video_list, batch_num, total_batches=6):
    """Split video list into batches and return the requested batch."""
    if batch_num == 0:
        return video_list  # Process all
    
    total_videos = len(video_list)
    batch_size = math.ceil(total_videos / total_batches)
    start_idx = (batch_num - 1) * batch_size
    end_idx = min(batch_num * batch_size, total_videos)
    return video_list[start_idx:end_idx]

def main(args):
    from AnonHead.segment_multiple import Segment
    from AnonHead.predict_multiple_voxceleb import Predictor
    from Generation.utils.head_pose import get_frontal_frame
    
    # Get and filter videos
    all_videos = get_all_videos(hdtf_train)
    valid_paths_list = filter_processed_videos(all_videos, hdtf_anon)
    
    # Split into batches
    batch_videos = split_batches(valid_paths_list, args.batch_num, args.total_batches)
    
    print(f"Found {len(all_videos)} total videos")
    print(f"{len(valid_paths_list)} need processing")
    print(f"Processing batch {args.batch_num} of {args.total_batches} ({len(batch_videos)} videos)")
    
    # Initialize logging
    log_file = os.path.join('/home/aegin/projects/anonymization/AnonNET/stats_and_jobs_hdtf', 
                          f"processing_summary_hdtf_batch{args.batch_num}.txt")
    
    gpu_info = {
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        'cuda_available': torch.cuda.is_available(),
        'total_memory': f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB" 
                       if torch.cuda.is_available() else "N/A"
    }
    
    with open(log_file, "w") as f:
        f.write(f"Processing Batch {args.batch_num}/{args.total_batches}\n")
        f.write(f"Videos in batch: {len(batch_videos)}\n")
        f.write(f"GPU: {gpu_info['device']}\n")
        f.write(f"VRAM: {gpu_info['total_memory']}\n\n")
    
    # Initialize models
    segment = Segment(load_det=False, load_seg=False)
    predictor = Predictor()
    
    # Process batch
    start_time = time.time()
    processed_count = 0
    last_log_time = start_time
    
    with tqdm(batch_videos, desc=f"Batch {args.batch_num}", unit="video") as pbar:
        for vid_path in pbar:
            try:
                base_name = os.path.basename(vid_path)
                video_name = os.path.splitext(base_name)[0]

                src_path = os.path.join(hdtf_source, f"{video_name}_source.png")
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
                outputs = segment.retinaface_detect_and_annotate(
                    img=src_path,
                    margin=0.8,
                    method='face'
                )
                
                if not outputs:
                    out_image = Image.open(src_path)
                    save_p = os.path.join(hdtf_anon, f"{video_name}_anon_nodet.png")
                    out_image.save(save_p)
                    continue
                
                # Keep only largest face if multiple detected
                if len(outputs) > 1:
                    areas = [(x2-x1)*(y2-y1) for _, _, (x1, x2, y1, y2) in outputs]
                    outputs = [outputs[areas.index(max(areas))]]
                
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
                #     "im_path": vid_path
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
                for mask_img, seg_img, bbox in outputs[:1]:  # Process only first face
                    conf["mask"] = mask_img
                    conf["image"] = seg_img
                    crop, dist, attrs = predictor.anonymize(**conf)
                    if crop is None:
                        continue

                    resized_mask = mask_img.resize(seg_img.size, Image.LANCZOS)
                    resized_crop = crop.resize(seg_img.size, Image.LANCZOS)
                    generated.append((resized_mask, resized_crop, bbox))

                # Merge and save
                out_image = segment.merge_crops(frame_pil, generated)
                out_image = out_image.resize(org_size, Image.LANCZOS)
                dist_str = f"{dist:.3f}"
                attrs_str = "_".join(a.replace(" ", "-") for a in attrs)
                save_p = os.path.join(hdtf_anon, f"{video_name}_anon_{dist_str}_{attrs_str}.png")
                out_image.save(save_p)

                processed_count += 1
                torch.cuda.empty_cache()

                # Log every 5 hours
                if time.time() - last_log_time > 5*3600:
                    with open(log_file, "a") as f:
                        f.write(f"[{time.ctime()}] Processed {processed_count} videos\n")
                    last_log_time = time.time()

            except Exception as e:
                print(f"\nError processing {vid_path}: {e}")
                continue

    # Final logging
    total_time = (time.time() - start_time)/60
    peak_mem = torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0
    
    with open(log_file, "a") as f:
        f.write(f"\nCompleted in {total_time:.1f} minutes\n")
        f.write(f"Processed {processed_count}/{len(batch_videos)} videos\n")
        if torch.cuda.is_available():
            f.write(f"Peak GPU memory: {peak_mem:.2f} GB\n")
    
    print(f"\nBatch {args.batch_num} complete!")
    print(f"Processed {processed_count}/{len(batch_videos)} videos")
    print(f"Time: {total_time:.1f} minutes")

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
    
    all_videos = get_all_videos(hdtf_train)
    
    valid_paths_list = filter_processed_videos(all_videos, output_folder, file_type='.mp4')
    
    print(f"Found {len(valid_paths_list)} videos.")
    
    # Start timer
    start_time = time.time()
    
    # ---------------------------------------------------
    # 0) Initialize logging
    # ---------------------------------------------------
    log_file = os.path.join('/home/aegin/projects/anonymization/AnonNET/stats_and_jobs_hdtf', f"processing_video_hdtf.txt")
    
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
            
            cropped_path = os.path.join(cropped_original, f"{video_name}.mp4")
            
            if not os.path.exists(cropped_path):
                cap = cv2.VideoCapture(vid_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Calculate number of frames for 15 seconds
                frames_to_keep = min(int(fps * 15), total_frames)
                
                # Set up video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(cropped_path, fourcc, fps, (width, height))
                
                # Read and write frames
                frame_count = 0
                while frame_count < frames_to_keep:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                    frame_count += 1
                
                cap.release()
                out.release()
            
            anon_im = get_anon_filepath(video_name, hdtf_anon)
            if anon_im is None:  # Check if anonymized image exists
                print(f"No anonymized image found for {video_name}")
                continue
            
            args_lp = ArgumentConfig(
                source=anon_im,
                driving=cropped_path,
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
    
    # Anonymization args
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--max_height", type=int, default=612)
    parser.add_argument("--max_width", type=int, default=612)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    
    # Batch processing args
    parser.add_argument("--batch_num", type=int, default=0,
                       help="0=all, 1=first 1/6, 2=second 1/6, etc.")
    parser.add_argument("--total_batches", type=int, default=6,
                       help="Total number of batches to split into")
    
    args = parser.parse_args()
    #main(args)
    animate(args)