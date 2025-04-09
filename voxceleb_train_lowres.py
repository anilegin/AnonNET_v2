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
from AnonHead.predict_multiple_voxceleb_llm import Predictor
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
vox_route = '/data/stars/share/vox-celeb-anon/anon_lowres'
vox_source = '/data/stars/share/vox-celeb-anon/source_im'

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

from skimage.exposure import match_histograms
import numpy as np
from PIL import ImageFilter

# def post_process(out_image, original):
#     """SD v1.5-specific fixes"""
#     # 1. Color correction
#     out_np = np.array(out_image)
#     orig_np = np.array(original)
#     out_np = match_histograms(out_np, orig_np, channel_axis=-1)
    
#     # 2. Quality matching
#     out_image = Image.fromarray(out_np).filter(
#         ImageFilter.GaussianBlur(radius=0.5)
#     )
    
#     # 3. Noise matching (SD v1.5 tends to be too clean)
#     noise = np.random.normal(0, 5, out_np.shape).astype(np.uint8)
#     out_image = Image.fromarray(np.clip(out_np + noise, 0, 255))
    
#     return out_image


# def smooth_hist_match(generated, original, face_mask):
#     """
#     Match colors only in facial regions with smooth blending
#     Args:
#         generated: PIL Image of generated face
#         original: PIL Image of original frame
#         face_mask: PIL Image mask (white=face area)
#     """
#     # Convert to numpy
#     gen_arr = np.array(generated)
#     orig_arr = np.array(original)
#     mask_arr = np.array(face_mask) > 127  # Binary mask
    
#     # Only process face region
#     for c in range(3):  # For each RGB channel
#         gen_face = gen_arr[..., c][mask_arr]
#         orig_face = orig_arr[..., c][mask_arr]
        
#         # Match histograms only for face pixels
#         matched_vals = match_histograms(gen_face, orig_face)
#         gen_arr[..., c][mask_arr] = matched_vals
    
#     # Smooth transition at mask edges
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
#     smooth_mask = cv2.GaussianBlur(mask_arr.astype(float), (25,25), 0)
    
#     # Blend
#     return Image.fromarray(
#         (gen_arr * smooth_mask[...,None] + 
#          np.array(generated) * (1-smooth_mask[...,None])).astype(np.uint8))
    
# def lab_color_transfer(generated, original, mask):
#     """More natural color transfer in LAB space"""
#     gen = cv2.cvtColor(np.array(generated), cv2.COLOR_RGB2LAB)
#     orig = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2LAB)
    
#     # Only modify face region
#     mask = np.array(mask) > 127
#     for c in [1,2]:  # Only chroma channels (skip luminance)
#         gen_channel = gen[:,:,c]
#         gen_channel[mask] = orig[:,:,c][mask] * 0.7 + gen_channel[mask] * 0.3
    
#     return Image.fromarray(cv2.cvtColor(gen, cv2.COLOR_LAB2RGB))

from skimage.color import rgb2lab, deltaE_ciede2000

import numpy as np
import cv2
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.exposure import match_histograms

def calculate_ciede2000(img1, img2, mask):
    """
    Robust CIEDE2000 color difference calculation
    Args:
        img1: PIL Image or numpy array (RGB)
        img2: PIL Image or numpy array (RGB)
        mask: PIL Image or numpy array (binary)
    Returns:
        numpy array of differences
    """
    # Convert inputs to numpy arrays
    img1 = np.asarray(img1, dtype=np.uint8)
    img2 = np.asarray(img2, dtype=np.uint8)
    mask = np.asarray(mask, dtype=bool)
    
    # Handle grayscale images
    if img1.ndim == 2:
        img1 = np.stack([img1]*3, axis=-1)
    if img2.ndim == 2:
        img2 = np.stack([img2]*3, axis=-1)
    
    # Convert to LAB color space
    lab1 = rgb2lab(img1)
    lab2 = rgb2lab(img2)
    
    # Calculate differences in masked region
    diff = np.zeros_like(mask, dtype=float)
    diff[mask] = deltaE_ciede2000(
        lab1[mask], 
        lab2[mask],
        channel_axis=-1
    )
    return diff

def color_discrepancy(img1, img2, mask):
    """
    Calculate average color difference in masked region
    Returns:
        Normalized difference score (0-1 where 1 = max difference)
    """
    try:
        diff = calculate_ciede2000(img1, img2, mask)
        mask_arr = np.asarray(mask, dtype=bool)
        avg_diff = diff[mask_arr].mean()
        return min(avg_diff / 100.0, 1.0)  # Normalize to 0-1 range
    except Exception as e:
        print(f"Color discrepancy error: {e}")
        return float('inf')

def smooth_hist_match(generated, original, face_mask, blend_strength=0.8, kernel_size=51):
    """
    Improved histogram matching with:
    - Type safety
    - Noise preservation
    - Edge-aware blending
    """
    # Convert inputs with type checking
    gen_arr = np.asarray(generated, dtype=np.float32)
    orig_arr = np.asarray(original, dtype=np.float32)
    mask_arr = np.asarray(face_mask, dtype=np.uint8) > 127
    
    # Preserve original noise characteristics
    orig_noise = orig_arr - cv2.GaussianBlur(orig_arr, (0,0), 1.0)
    
    # Process each channel
    for c in range(3):
        # Get face regions
        gen_face = gen_arr[..., c][mask_arr]
        orig_face = orig_arr[..., c][mask_arr]
        
        # Skip empty masks
        if len(gen_face) == 0:
            continue
            
        # Histogram matching with clipping
        matched = np.clip(match_histograms(gen_face, orig_face), 0, 255)
        gen_arr[..., c][mask_arr] = (
            blend_strength * matched + 
            (1-blend_strength) * gen_face
        )
    
    # Edge blending
    smooth_mask = cv2.GaussianBlur(
        mask_arr.astype(np.float32), 
        (kernel_size, kernel_size), 
        0
    )
    smooth_mask = np.clip(smooth_mask * 1.5, 0, 1)
    
    # Composite result
    result = np.clip(
        gen_arr * smooth_mask[...,None] + 
        np.array(generated) * (1-smooth_mask[...,None]) + 
        orig_noise * 0.3,
        0, 255
    ).astype(np.uint8)
    
    return Image.fromarray(result)

def lab_color_transfer(generated, original, mask, luminance_preserve=0.9):
    """
    Natural color transfer in LAB space with:
    - Luminance preservation
    - Chroma normalization
    - Edge blending
    """
    # Convert to LAB
    gen = cv2.cvtColor(np.asarray(generated), cv2.COLOR_RGB2LAB)
    orig = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2LAB)
    mask = np.asarray(mask) > 127
    
    # Preserve luminance
    gen[:,:,0] = (
        luminance_preserve * gen[:,:,0] + 
        (1-luminance_preserve) * orig[:,:,0]
    )
    
    # Normalized chroma transfer
    for c in [1, 2]:  # a and b channels
        gen_chan = gen[:,:,c]
        orig_chan = orig[:,:,c]
        
        # Calculate statistics
        gen_mean, gen_std = gen_chan[mask].mean(), gen_chan[mask].std() + 1e-6
        orig_mean, orig_std = orig_chan[mask].mean(), orig_chan[mask].std() + 1e-6
        
        # Transfer normalized values
        transfer_vals = (gen_chan[mask] - gen_mean) * (orig_std/gen_std) + orig_mean
        gen[:,:,c][mask] = np.clip(transfer_vals, 0, 255)
    
    # Convert back to RGB
    result = cv2.cvtColor(gen, cv2.COLOR_LAB2RGB)
    
    # Edge blending
    smooth_mask = cv2.GaussianBlur(mask.astype(np.float32), (35,35), 0)
    blended = (
        result * smooth_mask[...,None] + 
        np.array(generated) * (1-smooth_mask[...,None])
    )
    
    return Image.fromarray(blended.astype(np.uint8))

def auto_color_transfer(generated, original, mask):
    """
    Smart color transfer that automatically selects the best method
    based on content analysis
    """
    # Convert inputs
    gen_arr = np.asarray(generated)
    orig_arr = np.asarray(original)
    mask_arr = np.asarray(mask) > 127
    
    # Skip if mask is empty
    if not np.any(mask_arr):
        return generated
    
    # Analyze face region
    face_pixels = orig_arr[mask_arr]
    skin_std = face_pixels.std(axis=0).mean()
    face_ratio = mask_arr.mean()
    
    # Method selection
    if skin_std < 25 or face_ratio < 0.2:
        # LAB transfer for uniform colors/small faces
        result = lab_color_transfer(
            generated, original, mask,
            luminance_preserve=0.85 + (skin_std * 0.002)
        )
    else:
        # Histogram matching for complex textures
        result = smooth_hist_match(
            generated, original, mask,
            blend_strength=0.7 + (face_ratio * 0.3),
            kernel_size=int(50 * (1 - face_ratio)) | 1
        )
    
    # Quality check
    if color_discrepancy(result, original, mask) > 0.1:
        # Fallback hybrid approach
        lab_result = lab_color_transfer(generated, original, mask)
        result = smooth_hist_match(lab_result, original, mask)
    
    return result


@torch.no_grad()
def main(args):
    
    txt_file = "/data/stars/share/vox-celeb-dataset/vox-celeb-50000/file.list"  # replace with your actual text file
    all_videos = get_valid_paths(txt_file)
    
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
    
    
    all_videos = all_videos[:15]
    
    # valid_paths_list = filter_processed_videos(all_videos,vox_route)
    valid_paths_list = all_videos
    
    print(f"Found {len(valid_paths_list)} videos.")
    
    # Start timer
    start_time = time.time()
    
    # ---------------------------------------------------
    # 0) Initialize logging
    # ---------------------------------------------------
    log_file = os.path.join('/home/aegin/projects/anonymization/AnonNET/voxceleb_lowres_ablation', f"processing_summary_voxceleb_{args.even}.txt")
    
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
            # expected_output_pattern = os.path.join(vid_folder, f"{video_name}_anon_*.png")
            # if glob.glob(expected_output_pattern):
            #     print(f"Skipping {vid_path}: already processed (existing file in {vid_folder})")
            #     continue

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
            # conf = {
            #     "image": frame_pil,
            #     "prompt": "",
            #     "mask": None,
            #     "negative_prompt": args.negative_prompt,
            #     "strength": [0.95, 0.25, 0.15],
            #     "max_height": args.max_height,
            #     "max_width": args.max_width,
            #     "steps": 35,
            #     "seed": args.seed,
            #     "guidance_scale": 8.0,
            #     "im_path": src_path
            # }
            
            conf = {
                "image": frame_pil,
                "prompt": "",
                "mask": None,
                "steps": 50,
                "seed": args.seed,
                "guidance_scale": 8.0,
                "im_path": src_path,
                "max_height": 320,  # Slightly upscale from 224
                "max_width": 320,
                "strength": [1, 0.5, 0.3],
                "negative_prompt": (
                    "8k, sharp focus, studio lighting, detailed skin texture, "
                    "perfect complexion, DSLR, professional photo"
                )
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
            
            # out_image = smooth_hist_match(out_image, frame_pil,output)
            # out_image2 = lab_color_transfer(out_image, frame_pil,output)
            # out_image3 = smooth_hist_match(out_image2,frame_pil,output)
            # out_image4 = auto_color_transfer(out_image,frame_pil,output)
            
            # save_p = os.path.join(vid_folder, f"{video_name}_anon5_smooth_{filename_info}.png")
            # save_p2 = os.path.join(vid_folder, f"{video_name}_anon5_lab_{filename_info}.png")
            # save_p3 = os.path.join(vid_folder, f"{video_name}_anon5_mix_{filename_info}.png")
            # save_p4 = os.path.join(vid_folder, f"{video_name}_anon5_auto_{filename_info}.png")
            # out_image.save(save_p)
            # out_image2.save(save_p2)
            # out_image3.save(save_p3)
            # out_image4.save(save_p4)
            from post_process_vox import auto_postprocess

            final_image = auto_postprocess(
                source=frame_pil,
                generated=out_image,
                mask=output,  # your face mask
                apply_blur=True,
                apply_histogram=True,
                apply_blend=True,
                apply_jpeg=False
            )
            save_p = os.path.join(vid_folder, f"{video_name}_anon6.png")
            final_image.save(save_p)            
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
    
    main(args)


