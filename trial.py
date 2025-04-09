# import cv2
import os
import os.path as osp
# import numpy as np
# import tyro
# import subprocess
# from LivePortrait.src.config.argument_config import ArgumentConfig
# from LivePortrait.src.config.inference_config import InferenceConfig
# from LivePortrait.src.config.crop_config import CropConfig
# from LivePortrait.src.live_portrait_pipeline import LivePortraitPipeline
# from PIL import Image
# from AnonHead.segment_multiple import Segment
# from LivePortrait.inference import main as motion_lp




# def paste_subvideos_back(
#     original_video_path = "/home/aegin/projects/anonymization/AnonNET/results/temp_driving/scene_0_id_0.mp4",
#     bboxes = [(92, 96, 257, 317),(324, 124, 460, 312)],
#     processed_video_paths = ['/home/aegin/projects/anonymization/AnonNET/results/temp_driving/processed_heads/head0.mp4/head0_crop--head0.mp4',
#                              '/home/aegin/projects/anonymization/AnonNET/results/temp_driving/processed_heads/head1.mp4/head1_crop--head1.mp4'],
#     output_path="./results/final_output.mp4"
# ):
#     """
#     Reads 'original_video_path' and 'processed_video_paths' together,
#     for each frame index, inserts the processed subvideo frames at
#     bounding boxes in the original frame. If there's a size mismatch
#     (like 1 pixel difference), it automatically resizes the processed
#     frame to match the bounding box.
#     """
#     import cv2
#     import os
    
#     for pv in processed_video_paths:
#         print(f"File: {pv} exists? {os.path.isfile(pv)} size={os.path.getsize(pv) if os.path.isfile(pv) else 'N/A'}")
#     cap_test = cv2.VideoCapture(pv)
#     ret, frame = cap_test.read()
#     print(f"File: {pv}, ret={ret}, frame_shape={frame.shape if ret else None}")

    

#     cap_orig = cv2.VideoCapture(original_video_path)
#     width  = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps    = cap_orig.get(cv2.CAP_PROP_FPS)
#     frame_count = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))

#     # Prepare VideoCapture for each processed subvideo
#     caps_processed = []
#     for pv in processed_video_paths:
#         c = cv2.VideoCapture(pv)
#         caps_processed.append(c)
    
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     frame_idx = 0
#     while True:
#         ret_orig, frame_orig = cap_orig.read()
#         if not ret_orig:
#             break

#         # For each bounding box, read one frame from the corresponding processed video
#         for i, (x1, y1, x2, y2) in enumerate(bboxes):
#             ret_proc, frame_proc = caps_processed[i].read()
#             if not ret_proc:
#                 print(f"No frame read for bounding box index={i}, frame_idx={frame_idx}")
#                 # If the processed video ended, skip
#                 continue
            
#             # Check dimension mismatch
#             expected_h = y2 - y1
#             expected_w = x2 - x1
#             proc_h, proc_w = frame_proc.shape[:2]  # shape = (height, width, channels)
            
#             if (proc_h != expected_h) or (proc_w != expected_w):
#                 # Resize to match bounding box
#                 print(
#                     f"[DEBUG] Resizing processed frame from "
#                     f"({proc_h}x{proc_w}) to ({expected_h}x{expected_w}) "
#                     f"for bounding box #{i}, frame_idx={frame_idx}"
#                 )
#                 frame_proc = cv2.resize(
#                     frame_proc,
#                     (expected_w, expected_h),
#                     interpolation=cv2.INTER_AREA
#                 )
            
#             # Paste the processed frame into the original
#             frame_orig[y1:y2, x1:x2] = frame_proc  # BGR
#             # debug_overlay = np.ones_like(frame_proc) * [0, 0, 255]  # bright red
#             # frame_orig[y1:y2, x1:x2] = debug_overlay

#         out.write(frame_orig)
#         frame_idx += 1

#     # Cleanup
#     cap_orig.release()
#     for c in caps_processed:
#         c.release()
#     out.release()

#     print("All done! Final video:", output_path)
    
def fol_num(directory_path):
    
    import os

    def count_files_in_directory(path):
        """
        Return the number of files (not directories) in the given path.
        """
        # Make sure the path is valid
        if not os.path.isdir(path):
            print(f"Error: The path '{path}' is not a valid directory.")
            return 0

        # Count only the items that are files
        return sum(
            1
            for entry in os.listdir(path)
            if os.path.isfile(os.path.join(path, entry))
        )

    # Example usage:
    file_count = count_files_in_directory(directory_path)
    print(f"There are {file_count} files in '{directory_path}'.")
    
def remove_contents(folder):
    """
    Remove all files and subdirectories inside the given folder.

    Args:
        folder (str): Path to the folder whose contents should be removed.
    """
    for entry in os.listdir(folder):
        path = os.path.join(folder, entry)
        try:
            # Remove file or symbolic link
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            # Remove directory and its contents
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(f"Failed to delete {path}. Reason: {e}")
            
from collections import defaultdict

def find_duplicate_videos_by_name(folder_path):
    # Dictionary to store filenames and their paths
    name_dict = defaultdict(list)
    
    # Supported video extensions
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')
    
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.split("_anon")[0].lower().endswith(video_extensions):
                filepath = os.path.join(root, filename)
                name_dict[filename].append(filepath)
    
    # Filter out non-duplicates (only keep entries with >1 file)
    duplicates = {name: paths for name, paths in name_dict.items() if len(paths) > 1}
    print(len(duplicates))
    return duplicates

# # Usage
# folder_path = '/path/to/your/videos'
# duplicates = find_duplicate_videos_by_name(folder_path)

# print(f"Found {len(duplicates)} duplicate filenames:")
# for name, paths in duplicates.items():
#     print(f"\nFilename: {name}")
#     for path in paths:
#         print(f" - {path}")

import os

def count_files_in_paths(vox_files_path, vox_route_path, vox_source_path):
    """
    Counts the total number of files in three specified paths (recursively).
    
    Args:
        vox_files_path (str): Path to the original VoxCeleb dataset
        vox_route_path (str): Path to the anonymized output images
        vox_source_path (str): Path to the source images
    
    Returns:
        dict: A dictionary containing counts for each path and the total
    """
    def count_files_recursive(directory):
        count = 0
        for root, dirs, files in os.walk(directory):
            count += len(files)
        return count
    
    try:
        original_count = count_files_recursive(vox_files_path)
    except Exception as e:
        print(f"Error counting files in {vox_files_path}: {e}")
        original_count = 0
        
    try:
        anon_count = count_files_recursive(vox_route_path)
    except Exception as e:
        print(f"Error counting files in {vox_route_path}: {e}")
        anon_count = 0
        
    try:
        source_count = count_files_recursive(vox_source_path)
    except Exception as e:
        print(f"Error counting files in {vox_source_path}: {e}")
        source_count = 0
    
    return {
        "original_files_count": original_count,
        "anonymized_files_count": anon_count,
        "source_files_count": source_count,
        "total_files_count": original_count + anon_count + source_count
    }


import shutil

def zip_folder(folder_path, output_path):
    """
    Zips a folder and all its contents.
    
    Args:
        folder_path (str): Path to the folder you want to zip
        output_path (str): Path where the zip file should be created 
                          (without .zip extension)
    """
    shutil.make_archive(output_path, 'zip', folder_path)

    
if __name__ == "__main__":
    
    #paste_subvideos_back()
    fol_num("/data/stars/share/celeba_hq/train/female")
    fol_num("/data/stars/share/celeba_hq/val/female")
    fol_num("/data/stars/share/celeba_hq/female_anon_abl")
    fol_num("/data/stars/share/celeba_hq/female_anon")
    
    fol_num("/data/stars/share/celeba_hq/train/male")
    fol_num("/data/stars/share/celeba_hq/val/male")
    fol_num("/data/stars/share/celeba_hq/male_anon_abl")
    fol_num("/data/stars/share/celeba_hq/male_anon")
    
    fol_num("/data/stars/share/LFW/lfw_anon")
    fol_num("/data/stars/share/LFW/lfw_anon_20")
    
    fol_num("/data/stars/share/CelebV_HQ_Anon/train_anon_final_im")
    fol_num("/data/stars/share/CelebV_HQ_Anon/train_anon_source")
    fol_num("/data/stars/share/CelebV_HQ_Anon/train_anon_videos")
    fol_num('/data/stars/share/CelebV-HQ/path_to_videos/train')
    
    _ = find_duplicate_videos_by_name("/data/stars/share/CelebV_HQ_Anon/train_anon_final_im")
    
    vox_files = '/data/stars/share/vox-celeb-dataset/vox-celeb-50000'
    vox_route = '/data/stars/share/vox-celeb-anon/anon_im'
    vox_source = '/data/stars/share/vox-celeb-anon/source_im'
    
    counts = count_files_in_paths(vox_files, vox_route, vox_source)
    print("File counts:")
    print(f"- Original VoxCeleb files: {counts['original_files_count']}")
    print(f"- Anonymized files: {counts['anonymized_files_count']}")
    print(f"- Source image files: {counts['source_files_count']}")
    print(f"- Total files: {counts['total_files_count']}")
    
    #zip_folder("/data/stars/share/CelebV_HQ_Anon/train_anon_final_im","/data/stars/share/CelebV_HQ_Anon/celebv_hq_anon")
    
    # remove_contents('/data/stars/share/CelebV_HQ_Anon/test_anon_im_celebv')
    # remove_contents('/data/stars/share/CelebV_HQ_Anon/test_anon_im_hdtf')
    # remove_contents('/data/stars/share/CelebV_HQ_Anon/test_anon_source_celebv')
    # remove_contents('/data/stars/share/CelebV_HQ_Anon/test_anon_source_hdtf')
    # remove_contents('/data/stars/share/celeba_hq/female_anon')