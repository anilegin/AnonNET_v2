import numpy as np 
import cv2
import torch
from PIL import Image
import argparse
import os

YOLO_MODEL = "models/yolov8x.pt"

from custom_yolo import CustomYOLO


class Segment:
    def load_YOLO(self, device: str, model: str):
        # There are many different sizes for the pretrained YOLOv8-model. Yolov8x is the largest one. 

        yolo = YOLO(model)

        yolo.to(device)

        return yolo
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = "models/yolov8x.pt"
        # self.yolo = self.load_YOLO(self.device, self.model)
        self.yolo = CustomYOLO()
        self.segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline(device=self.device)
    


    #def yolo_detect_and_annotate(self, img: Image.Image, margin: float = 0.1):
    def yolo_detect_and_annotate(self, img: str, margin: float = 0.1):
        """
        Args:
            img: A PIL image.
        
        Returns:
            A list of tuples: [(mask_image, cropped_image), ...].
            - mask_image: a PIL image (grayscale binary mask).
            - cropped_image: the original cropped region as a PIL image (RGB).
            Returns an empty list if no detections are found.
        """

        results = self.yolo.run(img)
        img = Image.open(img)
        
        
        # If YOLO returns no results or no boxes, just return an empty list
        if not results:
            return []

        # 'boxes.xyxy' is a Nx4 tensor: [x1, y1, x2, y2]
        # boxes_xyxy = results[0].boxes.xyxy # for ultralytics
        boxes_xyxy = results
        img_array = np.array(img.convert("RGB"))  # For cropping
        h, w = img_array.shape[:2]

        output = []
        for box in boxes_xyxy:
            #x1, y1, x2, y2 = box.tolist()  # floats from YOLO ultralytics
            x1, x2, y1, y2 = box
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            
            print(f"Original Box: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
            
            if x2 <= x1 or y2 <= y1:
                print("Skipping invalid bounding box:", box)
                continue

            box_width = x2 - x1
            box_height = y2 - y1
            margin_w = int(margin * box_width)
            margin_h = int(margin * box_height)

            x1 = max(0, x1 - margin_w)
            y1 = max(0, y1 - margin_h)
            x2 = min(w, x2 + margin_w)
            y2 = min(h, y2 + margin_h)

            if x2 <= x1 or y2 <= y1:
                print("Skipping invalid bounding box after margin adjustment:", (x1, x2, y1, y2))
                continue
            
            coordinates = (x1,x2,y1,y2)
            
            cropped_np = img_array[y1:y2, x1:x2]
            cropped_img_pil = Image.fromarray(cropped_np, mode="RGB")

            output.append(cropped_img_pil)

        return output
    
    def merge_crops(
            self,
            original_img: Image.Image,
            results: list[tuple[Image.Image, Image.Image, tuple[int,int,int,int]]]
        ) -> Image.Image:
        """
        Merges multiple (mask, crop, coords) outputs back into the original image.
        For each result, wherever the mask is white, the pixel from the cropped image 
        overwrites the original.
        
        Args:
            original_img: The PIL image we started with.
            results: A list of tuples in the form:
                    (mask_pil, cropped_img_pil, (x1, x2, y1, y2))
                    where 'mask_pil' is a binary (L-mode) segmentation mask,
                        'cropped_img_pil' is the corresponding original crop,
                        (x1, x2, y1, y2) are the bounding-box coordinates 
                        in that exact order.
                        
        Returns:
            A new PIL image (RGB) with all the masked crops pasted in place.
            Original pixels remain where the mask is black.
        """
        final_array = np.array(original_img.convert("RGB"))  # shape: (H, W, 3)
        H, W = final_array.shape[:2]

        for (mask_pil, cropped_pil, (x1, x2, y1, y2)) in results:
            
            mask_array = np.array(mask_pil)  # shape: (crop_height, crop_width), 0 or 255
            white_pixels = (mask_array > 127)  

            cropped_array = np.array(cropped_pil.convert("RGB"))  # shape: (crop_height, crop_width, 3)

            region = final_array[y1:y2, x1:x2]

            region[white_pixels] = cropped_array[white_pixels]

        final_image = Image.fromarray(final_array, mode="RGB")

        return final_image