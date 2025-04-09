from .head_segmentation import segmentation_pipeline as seg_pipeline
import numpy as np 
import cv2
import torch
# from ultralytics import YOLO
from PIL import Image
import argparse
import os
import sys
import torch
import torchvision.transforms as transforms

import gc

# YOLO_MODEL = "models/yolov8x.pt"

from .custom_yolo import CustomYOLO
from deepface import DeepFace

from .face_parsing.models.bisenet import BiSeNet
from .face_parsing.utils.common import ATTRIBUTES, COLOR_LIST, letterbox


class Segment:
    # def load_YOLO(self, device: str, model: str):
    #     # There are many different sizes for the pretrained YOLOv8-model. Yolov8x is the largest one. 

    #     yolo = YOLO(model)

    #     yolo.to(device)

    #     return yolo
    
    def load_face_seg(
        self,
        weight = "./AnonHead/face_parsing/weights/resnet18.pt",
        model = "resnet18"
    ):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = 19

        model = BiSeNet(num_classes, backbone_name=model)
        model.to(device)

        if os.path.exists(weight):
            model.load_state_dict(torch.load(weight))
        else:
            raise ValueError(f"Weights not found from given path ({weight})")
        
        model.eval()
        
        return model
    
    def __init__(self, load_det = True, load_seg = True, load_face = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = "models/yolov8x.pt"
        # self.yolo = self.load_YOLO(self.device, self.model)
        if load_det:
            self.yolo = CustomYOLO()
        if load_seg:
            self.segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline(device=self.device)
        if load_face:
            self.face_seg = self.load_face_seg()
    
    def fill_black_holes(self, mask: np.ndarray):

        mask_bin = np.where(mask > 127, 255, 0).astype(np.uint8)


        inverted = 255 - mask_bin

        # 3) Flood fill from the corners to label real background.
        h, w = inverted.shape[:2]
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        # Flood fill from each corner
        cv2.floodFill(inverted, flood_mask, (0, 0), 255)
        cv2.floodFill(inverted, flood_mask, (w - 1, 0), 255)
        cv2.floodFill(inverted, flood_mask, (0, h - 1), 255)
        cv2.floodFill(inverted, flood_mask, (w - 1, h - 1), 255)

        # 4) Invert back
        inverted_back = 255 - inverted

        # 5) Combine with the original to ensure enclosed holes are filled
        filled_mask = cv2.bitwise_or(mask_bin, inverted_back)

        return filled_mask
    
    def segment(self, image: np.ndarray):
        """
        Segments the input image, turning all non-black pixels in the segmented output to white.

        Args:
            image (numpy.ndarray): Input image as a NumPy array.

        Returns:
            numpy.ndarray: Output image with non-black pixels turned white.
        """
        segmented_image = image 

        # Create a mask for non-black pixels
        non_black_mask = (segmented_image[:, :, 0] != 0) | \
                        (segmented_image[:, :, 1] != 0) | \
                        (segmented_image[:, :, 2] != 0)

        # Turn all non-black pixels to white
        segmented_image[non_black_mask] = [255, 255, 255]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)

        return cleaned_image


    def annotate(self, img: Image.Image):
        """ Generates annotation image (segmentation) for the given image. 
        
        Arguments:
            image: image to segment. 
        Returns:
            mask_image: annotated segment(s) as a binary image or None if no people in img.
        """
        # Convert image to RGB
        img = img.convert("RGB")
        
        img_array = np.array(img)
        
        segmentation_map = self.segmentation_pipeline(img_array)
        final = img_array * cv2.cvtColor(segmentation_map, cv2.COLOR_GRAY2RGB)
        
        masked_image = self.segment(final)
        mask_image = Image.fromarray(masked_image)
        
        mask_image = mask_image.convert("L")  # Convert mask to grayscale
        mask_data = np.array(mask_image)
        binary_mask = np.where(mask_data > 127, 255, 0).astype(np.uint8)
        final_mask_image = Image.fromarray(binary_mask)

        return final_mask_image
    
    @torch.no_grad()
    def annotate_face(self, img: Image.Image, fill=True):
        
        def prepare_image(image):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            image_tensor = transform(image)
            image_batch = image_tensor.unsqueeze(0)

            return image_batch
        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        face_indices = [1, 2, 3, 4, 5, 9, 10, 11, 12, 13] #full face 
        
        resized_image = img.resize((512, 512), resample=Image.BILINEAR)
        transformed_image = prepare_image(resized_image)
        image_batch = transformed_image.to(device)
        
        output = self.face_seg(image_batch)[0]
        predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)
        face_mask = np.isin(predicted_mask, face_indices).astype(np.uint8) * 255
        face_mask = (face_mask == 255).astype(np.uint8) * 255  # Ensures only face is white
        original_size = img.size
        face_mask_image = Image.fromarray(face_mask).resize(original_size, Image.LANCZOS)
        
        if fill:
            mask_array = np.array(face_mask_image.convert("L"))
            mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
            mask_filled = self.fill_black_holes(mask_array)
            
            face_mask_image = Image.fromarray(mask_filled)
        
        torch.cuda.empty_cache()
        gc.collect()

        return face_mask_image


    # #def yolo_detect_and_annotate(self, img: Image.Image, margin: float = 0.1):
    # def yolo_detect_and_annotate_old(self, img: str, margin: float = 0.1):
    #     """
    #     Args:
    #         img: A PIL image.
        
    #     Returns:
    #         A list of tuples: [(mask_image, cropped_image), ...].
    #         - mask_image: a PIL image (grayscale binary mask).
    #         - cropped_image: the original cropped region as a PIL image (RGB).
    #         Returns an empty list if no detections are found.
    #     """

    #     # Run YOLO detection. Adjust 'classes=0' if you're detecting persons on class 0.
    #     #results = self.yolo.predict(img, save=False, classes=0)
    #     # results = self.yolo.detect(img)
    #     results = self.yolo.run(img)
    #     img = Image.open(img)
        
        
    #     # If YOLO returns no results or no boxes, just return an empty list
    #     if not results:
    #         return []

    #     # 'boxes.xyxy' is a Nx4 tensor: [x1, y1, x2, y2]
    #     # boxes_xyxy = results[0].boxes.xyxy # for ultralytics
    #     boxes_xyxy = results
    #     img_array = np.array(img.convert("RGB"))  # For cropping
    #     h, w = img_array.shape[:2]

    #     output = []
    #     for box in boxes_xyxy:
    #         #x1, y1, x2, y2 = box.tolist()  # floats from YOLO ultralytics
    #         x1, x2, y1, y2 = box
    #         x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            
    #         print(f"Original Box: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
            
    #         # Ensure box is valid
    #         if x2 <= x1 or y2 <= y1:
    #             print("Skipping invalid bounding box:", box)
    #             continue

    #         # Compute margin
    #         box_width = x2 - x1
    #         box_height = y2 - y1
    #         margin_w = int(margin * box_width)
    #         margin_h = int(margin * box_height)

    #         # Adjust bounding box with margins and clamp within image bounds
    #         x1 = max(0, x1 - margin_w)
    #         y1 = max(0, y1 - margin_h)
    #         x2 = min(w, x2 + margin_w)
    #         y2 = min(h, y2 + margin_h)

    #         # Validate adjusted box
    #         if x2 <= x1 or y2 <= y1:
    #             print("Skipping invalid bounding box after margin adjustment:", (x1, x2, y1, y2))
    #             continue
            
    #         coordinates = (x1,x2,y1,y2)
            
    #         # Crop the original image region
    #         cropped_np = img_array[y1:y2, x1:x2]
    #         cropped_img_pil = Image.fromarray(cropped_np, mode="RGB")

    #         # Run your annotate function on the cropped region
    #         mask_pil = self.annotate(cropped_img_pil)  # grayscale mask

    #         # Append the tuple
    #         output.append((mask_pil, cropped_img_pil, coordinates))

    #     return output
    
    @torch.no_grad()
    def retinaface_detect_and_annotate(self, img: str, margin: float = 0.1, method: str = 'head', fill = True ,only_boxes = False):
    
        output = []
        path_img = img
        img = Image.open(img)
        img_array = np.array(img.convert("RGB"))
        
        
        try:
            extracted_faces = DeepFace.extract_faces(
                img_path = path_img,        # can be a NumPy array
                detector_backend = 'retinaface',   # or 'mtcnn', 'retinaface', etc.
                enforce_detection = False,     # set to True if you want to raise errors on no-face
                expand_percentage = margin*100,
                align = True                  # whether to align face or not
            )
        except Exception as e:
            print(f"DeepFace extract failed: {e}")
            extracted_faces = []

        extracted_faces = [f for f in extracted_faces if f.get("confidence", 0) > 0.85]
        num_faces = len(extracted_faces)
        print(f"Detected {num_faces} face(s).")
        if num_faces == 0: return output #no face
        img_array = np.array(img.convert("RGB"))  # For cropping
        h, w = img_array.shape[:2]
        
        if only_boxes:
            coordinates = []
            for face_data in extracted_faces:
                # 'face' is the extracted face as a numpy array
                face_np = face_data["face"]
                facial_area = face_data.get("facial_area", None)
                x, y, w_face, h_face = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                x1 = x
                y1 = y
                x2 = x + w_face
                y2 = y + h_face
                coordinates.append((x1, x2, y1, y2))
            return coordinates
        
        for face_data in extracted_faces:
            # 'face' is the extracted face as a numpy array
            face_np = face_data["face"]
            facial_area = face_data.get("facial_area", None)
            x, y, w_face, h_face = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            x1, x2, y1, y2 = x, x+w_face, y, y+h_face
            # Make the box a square
            size = max(w_face, h_face)  # Take the maximum side length
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Adjust the coordinates to keep the square centered
            x1 = center_x - size // 2
            x2 = center_x + size // 2
            y1 = center_y - size // 2
            y2 = center_y + size // 2

            margin_size = int(margin * size) 
            x1 -= margin_size
            x2 += margin_size
            y1 -= margin_size
            y2 += margin_size

            x1 = max(0, x1)
            y1 = max(0, y1)
            # x2 = min(w - 1, x2)  
            # y2 = min(h - 1, y2)  
            x2 = min(w, x2)  
            y2 = min(h, y2) 
            
            coordinates = (x1, x2, y1, y2)
        
            cropped_np = img_array[y1:y2, x1:x2]
            cropped_img_pil = Image.fromarray(cropped_np, mode="RGB")
            if method not in ['face','head']:
                method = 'face'
                self.face_seg = self.load_face_seg()
            if method == 'face':
                mask_pil = self.annotate_face(cropped_img_pil, fill)
            if method == 'head':
                mask_pil = self.annotate(cropped_img_pil)
            output.append((mask_pil, cropped_img_pil, coordinates))
            
        torch.cuda.empty_cache()
        gc.collect()
        # print(output)
        return output
    
    @torch.no_grad()
    def retinaface_detect_and_annotate2(self, img: str, margin: float = 0.2, method: str = 'head', fill=True, only_boxes=False):
        from PIL import Image, ImageDraw
        output = []
        path_img = img
        img_pil = Image.open(path_img)
        annotated_img = img_pil.copy()  # Create a copy to draw boxes on
        draw = ImageDraw.Draw(annotated_img)
        img_array = np.array(img_pil.convert("RGB"))
        
        try:
            extracted_faces = DeepFace.extract_faces(
                img_path=path_img,
                detector_backend='retinaface',
                enforce_detection=False,
                expand_percentage=margin*100,
                align=True
            )
        except Exception as e:
            print(f"DeepFace extract failed: {e}")
            extracted_faces = []

        extracted_faces = [f for f in extracted_faces if f.get("confidence", 0) > 0.85]
        num_faces = len(extracted_faces)
        print(f"Detected {num_faces} face(s).")
        
        if num_faces == 0:
            return (output if not only_boxes else [], annotated_img)
        
        h, w = img_array.shape[:2]
        all_coordinates = []

        if only_boxes:
            coordinates = []
            for face_data in extracted_faces:
                facial_area = face_data.get("facial_area", {})
                x, y, w_face, h_face = facial_area.get('x', 0), facial_area.get('y', 0), facial_area.get('w', 0), facial_area.get('h', 0)
                x1, y1, x2, y2 = x, y, x + w_face, y + h_face
                coordinates.append((x1, x2, y1, y2))
                all_coordinates.append((x1, y1, x2, y2))
            # Draw original detection boxes
            for (x1, y1, x2, y2) in all_coordinates:
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            return coordinates, annotated_img
        else:
            for face_data in extracted_faces:
                facial_area = face_data.get("facial_area", {})
                x, y, w_face, h_face = facial_area.get('x', 0), facial_area.get('y', 0), facial_area.get('w', 0), facial_area.get('h', 0)
                # Convert to square with margin
                size = max(w_face, h_face)
                center_x = (x + x + w_face) // 2
                center_y = (y + y + h_face) // 2
                x1 = center_x - size // 2
                x2 = center_x + size // 2
                y1 = center_y - size // 2
                y2 = center_y + size // 2
                # Apply margin
                margin_size = int(margin * size)
                x1 -= margin_size
                x2 += margin_size
                y1 -= margin_size
                y2 += margin_size
                # Clamp coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                all_coordinates.append((x1, y1, x2, y2))
                # Crop and process
                cropped_np = img_array[y1:y2, x1:x2]
                cropped_img_pil = Image.fromarray(cropped_np, mode="RGB")
                # Annotate based on method
                if method not in ['face', 'head']:
                    method = 'face'
                    self.face_seg = self.load_face_seg()
                mask_pil = self.annotate_face(cropped_img_pil, fill) if method == 'face' else self.annotate(cropped_img_pil)
                output.append((mask_pil, cropped_img_pil, (x1, x2, y1, y2)))
            # Draw adjusted boxes with margin
            for (x1, y1, x2, y2) in all_coordinates:
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            return output, annotated_img

        torch.cuda.empty_cache()
        gc.collect()
        return output, annotated_img
    
    @torch.no_grad()
    def yolo_detect_and_annotate(self, img: str, margin: float = 0.1, force = True, bboxes = [], only_boxes = False):
        """
        Args:
            img: Path to image.
        
        Returns:
            A list of tuples: [(mask_image, cropped_image), ...].
            - mask_image: a PIL image (grayscale binary mask).
            - cropped_image: the original cropped region as a PIL image (RGB).
            Returns an empty list if no detections are found.
        """

        # Run YOLO detection. Adjust 'classes=0' if you're detecting persons on class 0.
        #results = self.yolo.predict(img, save=False, classes=0)
        # results = self.yolo.detect(img)
        if bboxes == []:
            bboxes = self.yolo.run(img)
            path_img = img
            img = Image.open(img)
            
            
            # If YOLO returns no bboxes or no boxes, try retina face
            if not bboxes and force:
                
                print("No faces found by YOLO. Trying RetinaFace to find face(s).")
                output = []
                output = self.retinaface_detect_and_annotate(path_img, method = 'head')
                #print(output)
                return output

            # 'boxes.xyxy' is a Nx4 tensor: [x1, y1, x2, y2]
            # boxes_xyxy = bboxes[0].boxes.xyxy # for ultralytics
        else:
            path_img = img
            img = Image.open(img)
            
        boxes_xyxy = bboxes
        if only_boxes:
            return boxes_xyxy
        img_array = np.array(img.convert("RGB"))  # For cropping
        h, w = img_array.shape[:2]

        output = []
        for box in boxes_xyxy:
            x1, x2, y1, y2 = box
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

            print(f"Original Box: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")

            # Ensure box is valid
            if x2 <= x1 or y2 <= y1:
                print("Skipping invalid bounding box:", box)
                continue

            # Compute width and height
            box_width = x2 - x1
            box_height = y2 - y1

            # Make the box a square
            size = max(box_width, box_height)  # Take the maximum side length
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Adjust the coordinates to keep the square centered
            x1 = center_x - size // 2
            x2 = center_x + size // 2
            y1 = center_y - size // 2
            y2 = center_y + size // 2

            margin_size = int(margin * size) 
            x1 -= margin_size
            x2 += margin_size
            y1 -= margin_size
            y2 += margin_size

            x1 = max(0, x1)
            y1 = max(0, y1)
            # x2 = min(w - 1, x2)  
            # y2 = min(h - 1, y2)  
            x2 = min(w, x2)  
            y2 = min(h, y2)

            # Validate adjusted box
            if x2 <= x1 or y2 <= y1:
                print("Skipping invalid bounding box after margin adjustment:", (x1, x2, y1, y2))
                continue
            
            coordinates = (x1,x2,y1,y2)
            
            # Crop the original image region
            cropped_np = img_array[y1:y2, x1:x2]
            cropped_img_pil = Image.fromarray(cropped_np, mode="RGB")

            # Run your annotate function on the cropped region
            mask_pil = self.annotate(cropped_img_pil)  # grayscale mask

            # Append the tuple
            output.append((mask_pil, cropped_img_pil, coordinates))
        
        torch.cuda.empty_cache()
        gc.collect()    
        
        return output
    
    
    # FACE SEGMENTATION
    
    # def annotate_face(self, img: Image.Image, fill=True):
    #     from .face_parsing.inference import inference

    #     mask_image = inference(image=img,
    #                            face_indices = [1, 2, 3, 4, 5, 9, 10, 11, 12, 13] #full face 
    #                            )
    #     final_mask_image = mask_image
        
    #     if fill:
    #         mask_array = np.array(mask_image.convert("L"))
    #         mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
    #         mask_filled = self.fill_black_holes(mask_array)
            
    #         final_mask_image = Image.fromarray(mask_filled)

    #     return final_mask_image
    
    @torch.no_grad()
    def bisenet_detect_and_annotate(self, img: str, margin: float = 0.1, force = True, bboxes = [], only_boxes = False, fill=True):
        """
        Args:
            img: Path to image.
        
        Returns:
            A list of tuples: [(mask_image, cropped_image), ...].
            - mask_image: a PIL image (grayscale binary mask).
            - cropped_image: the original cropped region as a PIL image (RGB).
            Returns an empty list if no detections are found.
        """

        # Run YOLO detection. Adjust 'classes=0' if you're detecting persons on class 0.
        #bboxes = self.yolo.predict(img, save=False, classes=0)
        # bboxes = self.yolo.detect(img)
        if bboxes == []:
            bboxes = self.yolo.run(img)
            path_img = img
            img = Image.open(img)
            
            
            # If YOLO returns no bboxes or no boxes, try retina face
            if not bboxes and force:
                
                print("No faces found by YOLO. Trying RetinaFace to find face(s).")
                output = []
                output = self.retinaface_detect_and_annotate(path_img, method = 'face')
                #print(output)
                return output

        # 'boxes.xyxy' is a Nx4 tensor: [x1, y1, x2, y2]
        # boxes_xyxy = bboxes[0].boxes.xyxy # for ultralytics
        else:
            path_img = img
            img = Image.open(img)
            
            
        boxes_xyxy = bboxes
        if only_boxes:
            return boxes_xyxy
        img_array = np.array(img.convert("RGB"))  # For cropping
        h, w = img_array.shape[:2]

        output = []
        for box in boxes_xyxy:
            x1, x2, y1, y2 = box
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

            print(f"Original Box: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")

            # Ensure box is valid
            if x2 <= x1 or y2 <= y1:
                print("Skipping invalid bounding box:", box)
                continue

            box_width = x2 - x1
            box_height = y2 - y1

            # Make the box a square
            size = max(box_width, box_height)  # Use the larger side as the new square size
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Adjust the coordinates to keep the square centered
            x1 = center_x - size // 2
            x2 = center_x + size // 2
            y1 = center_y - size // 2
            y2 = center_y + size // 2

            margin_size = int(margin * size) 
            x1 -= margin_size
            x2 += margin_size
            y1 -= margin_size
            y2 += margin_size

            x1 = max(0, x1)
            y1 = max(0, y1)
            # x2 = min(w - 1, x2)  
            # y2 = min(h - 1, y2)
            x2 = min(w, x2)  
            y2 = min(h, y2)  

            # Validate adjusted square box
            if x2 <= x1 or y2 <= y1:
                print("Skipping invalid square bounding box:", (x1, x2, y1, y2))
                continue

            coordinates = (x1, x2, y1, y2)

            # Crop the original image region
            cropped_np = img_array[y1:y2, x1:x2]
            cropped_img_pil = Image.fromarray(cropped_np, mode="RGB")

            # Run your annotate function on the cropped region
            mask_pil = self.annotate_face(cropped_img_pil,fill=fill)  # grayscale mask

            # Append the tuple
            output.append((mask_pil, cropped_img_pil, coordinates))
            
        torch.cuda.empty_cache()
        gc.collect()
            
        return output
        
         
    # def bisenet_detect_and_annotate(self, img: str, margin: float = 0.1, force = True):
    #     """
    #     Args:
    #         img: Path to image.
        
    #     Returns:
    #         A list of tuples: [(mask_image, cropped_image), ...].
    #         - mask_image: a PIL image (grayscale binary mask).
    #         - cropped_image: the original cropped region as a PIL image (RGB).
    #         Returns an empty list if no detections are found.
    #     """

    #     # Run YOLO detection. Adjust 'classes=0' if you're detecting persons on class 0.
    #     #bboxes = self.yolo.predict(img, save=False, classes=0)
    #     # bboxes = self.yolo.detect(img)
    #     bboxes = self.yolo.run(img)
    #     path_img = img
    #     img = Image.open(img)
        
        
    #     # If YOLO returns no bboxes or no boxes, try retina face
    #     if not bboxes and force:
            
    #         print("No faces found by YOLO. Trying RetinaFace to find face(s).")
    #         output = []
    #         img_array = np.array(img.convert("RGB"))
            
            
    #         try:
    #             extracted_faces = DeepFace.extract_faces(
    #                 img_path = path_img,        # can be a NumPy array
    #                 detector_backend = 'retinaface',   # or 'mtcnn', 'retinaface', etc.
    #                 enforce_detection = False,     # set to True if you want to raise errors on no-face
    #                 expand_percentage = 30,
    #                 align = True                  # whether to align face or not
    #             )
    #         except Exception as e:
    #             print(f"DeepFace extract failed: {e}")
    #             extracted_faces = []

    #         extracted_faces = [f for f in extracted_faces if f.get("confidence", 0) > 0.85]
    #         num_faces = len(extracted_faces)
    #         print(f"Detected {num_faces} face(s).")
    #         if num_faces == 0: sys.exit("No face detected. Exiting the anonymization pipeline.")
            
    #         for face_data in extracted_faces:
    #             # 'face' is the extracted face as a numpy array
    #             face_np = face_data["face"]
    #             facial_area = face_data.get("facial_area", None)
    #             x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
    #             coordinates = (x, x + w, y, y + h)
            
    #             cropped_np = img_array[y:y+h, x:x+w]
    #             cropped_img_pil = Image.fromarray(cropped_np, mode="RGB")
    #             mask_pil = self.annotate_face(cropped_img_pil)
    #             output.append((mask_pil, cropped_img_pil, coordinates))
    #         print(output)
    #         return output

    #     # 'boxes.xyxy' is a Nx4 tensor: [x1, y1, x2, y2]
    #     # boxes_xyxy = bboxes[0].boxes.xyxy # for ultralytics
    #     boxes_xyxy = bboxes
    #     img_array = np.array(img.convert("RGB"))  # For cropping
    #     h, w = img_array.shape[:2]

    #     output = []
    #     for box in boxes_xyxy:
    #         #x1, y1, x2, y2 = box.tolist()  # floats from YOLO ultralytics
    #         x1, x2, y1, y2 = box
    #         x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            
    #         print(f"Original Box: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
            
    #         # Ensure box is valid
    #         if x2 <= x1 or y2 <= y1:
    #             print("Skipping invalid bounding box:", box)
    #             continue

    #         # Compute margin
    #         box_width = x2 - x1
    #         box_height = y2 - y1
    #         margin_w = int(margin * box_width)
    #         margin_h = int(margin * box_height)

    #         # Adjust bounding box with margins and clamp within image bounds
    #         x1 = max(0, x1 - margin_w)
    #         y1 = max(0, y1 - margin_h)
    #         x2 = min(w, x2 + margin_w)
    #         y2 = min(h, y2 + margin_h)

    #         # Validate adjusted box
    #         if x2 <= x1 or y2 <= y1:
    #             print("Skipping invalid bounding box after margin adjustment:", (x1, x2, y1, y2))
    #             continue
            
    #         coordinates = (x1,x2,y1,y2)
            
    #         # Crop the original image region
    #         cropped_np = img_array[y1:y2, x1:x2]
    #         cropped_img_pil = Image.fromarray(cropped_np, mode="RGB")

    #         # Run your annotate function on the cropped region
    #         mask_pil = self.annotate_face(cropped_img_pil)  # grayscale mask

    #         # Append the tuple
    #         output.append((mask_pil, cropped_img_pil, coordinates))
            
    #     return output
    
    
    #safe on 21-03-25
    # def merge_crops(
    #         self,
    #         original_img: Image.Image,
    #         results: list[tuple[Image.Image, Image.Image, tuple[int,int,int,int]]]
    #     ) -> Image.Image:
    #     """
    #     Merges multiple (mask, crop, coords) outputs back into the original image.
    #     For each result, wherever the mask is white, the pixel from the cropped image 
    #     overwrites the original.
        
    #     Args:
    #         original_img: The PIL image we started with.
    #         results: A list of tuples in the form:
    #                 (mask_pil, cropped_img_pil, (x1, x2, y1, y2))
    #                 where 'mask_pil' is a binary (L-mode) segmentation mask,
    #                     'cropped_img_pil' is the corresponding original crop,
    #                     (x1, x2, y1, y2) are the bounding-box coordinates 
    #                     in that exact order.
                        
    #     Returns:
    #         A new PIL image (RGB) with all the masked crops pasted in place.
    #         Original pixels remain where the mask is black.
    #     """
    #     # Convert the original to a NumPy array for direct pixel manipulation
    #     final_array = np.array(original_img.convert("RGB"))  # shape: (H, W, 3)
    #     H, W = final_array.shape[:2]

    #     for (mask_pil, cropped_pil, (x1, x2, y1, y2)) in results:
            
    #         # Convert mask to NumPy array
    #         mask_array = np.array(mask_pil)  # shape: (crop_height, crop_width), 0 or 255
    #         # Create a boolean mask for white pixels
    #         white_pixels = (mask_array > 127)  # True where mask is white, False otherwise

    #         cropped_array = np.array(cropped_pil.convert("RGB"))  # shape: (crop_height, crop_width, 3)

    #         region = final_array[y1:y2, x1:x2]
            
    #         try:
    #             # Paste only where the mask is white
    #             region[white_pixels] = cropped_array[white_pixels]
    #         except IndexError:
    #             region = final_array[y1:y2+1, x1:x2+1]
    #             region[white_pixels] = cropped_array[white_pixels]
    #         except:
    #             print("Shape Mismatch!")

    #     final_image = Image.fromarray(final_array, mode="RGB")

    #     return final_image
    
    def merge_crops(
        self,
        original_img: Image.Image,
        results: list[tuple[Image.Image, Image.Image, tuple[int, int, int, int]]]
    ) -> Image.Image:
        """
        Merges anonymized crops back into the original image with robust dimension handling.
        """
        final_array = np.array(original_img.convert("RGB"))
        H, W = final_array.shape[:2]

        for (mask_pil, cropped_pil, (orig_x1, orig_x2, orig_y1, orig_y2)) in results:
            # 1. Sanitize coordinates to ensure they're within image bounds
            x1 = max(0, min(orig_x1, W))
            x2 = max(0, min(orig_x2, W))
            y1 = max(0, min(orig_y1, H))
            y2 = max(0, min(orig_y2, H))

            # 2. Calculate valid region dimensions
            region_width = x2 - x1
            region_height = y2 - y1
            if region_width <= 0 or region_height <= 0:
                continue

            # 3. Resize both mask and crop to match EXACT region dimensions
            try:
                resized_mask = mask_pil.resize((region_width, region_height), Image.LANCZOS)
                resized_crop = cropped_pil.resize((region_width, region_height), Image.LANCZOS)
            except ValueError as e:
                print(f"Resize error: {e}")
                continue

            # 4. Convert to arrays and create mask
            mask_array = np.array(resized_mask)
            cropped_array = np.array(resized_crop)
            white_pixels = mask_array > 127

            # 5. Validate dimensions before assignment
            if white_pixels.shape != (region_height, region_width):
                print(f"Dimension mismatch: Mask {white_pixels.shape} vs Region ({region_height}, {region_width})")
                continue

            # 6. Apply the anonymized pixels
            try:
                final_array[y1:y2, x1:x2][white_pixels] = cropped_array[white_pixels]
            except Exception as e:
                print(f"Merge error: {e}")
                print(f"Region dimensions: {final_array[y1:y2, x1:x2].shape}")
                print(f"Mask dimensions: {white_pixels.shape}")
                print(f"Crop dimensions: {cropped_array.shape}")

        return Image.fromarray(final_array, mode="RGB")
        
    # def merge_crops(
    #         self,
    #         original_img: Image.Image,
    #         results: list[tuple[Image.Image, Image.Image, tuple[int, int, int, int]]]
    #     ) -> Image.Image:
    #     """
    #     Merges anonymized crops back into the original image with robust dimension handling.
    #     """
    #     final_array = np.array(original_img.convert("RGB"))
    #     H, W = final_array.shape[:2]

    #     for (mask_pil, cropped_pil, (orig_x1, orig_x2, orig_y1, orig_y2)) in results:
    #         # 1. Sanitize and order coordinates
    #         x1, x2 = sorted([max(0, min(orig_x1, W)), max(0, min(orig_x2, W))])
    #         y1, y2 = sorted([max(0, min(orig_y1, H)), max(0, min(orig_y2, H))])
            
    #         # 2. Skip invalid regions
    #         region_width = x2 - x1
    #         region_height = y2 - y1
    #         if region_width <= 0 or region_height <= 0:
    #             continue

    #         # 3. Resize to exact region dimensions
    #         try:
    #             resized_mask = mask_pil.resize((region_width, region_height), Image.LANCZOS)
    #             resized_crop = cropped_pil.resize((region_width, region_height), Image.LANCZOS)
    #         except ValueError as e:
    #             print(f"Resize error: {e}")
    #             continue

    #         # 4. Convert to arrays and handle mask channels
    #         mask_array = np.array(resized_mask)
    #         cropped_array = np.array(resized_crop)
            
    #         # Handle grayscale mask
    #         if mask_array.ndim == 2:
    #             mask_array = mask_array[..., np.newaxis]  # Add channel dimension
            
    #         # Ensure mask matches image channels
    #         if mask_array.shape[-1] == 1 and final_array.shape[-1] == 3:
    #             mask_array = np.repeat(mask_array, 3, axis=-1)
            
    #         white_pixels = mask_array > 127

    #         # 5. Apply using np.where for clean replacement
    #         try:
    #             region = final_array[y1:y2, x1:x2]
    #             final_array[y1:y2, x1:x2] = np.where(white_pixels, cropped_array, region)
    #         except Exception as e:
    #             print(f"Merge error: {e}")
    #             print(f"Region shape: {final_array[y1:y2, x1:x2].shape}")
    #             print(f"Mask shape: {white_pixels.shape}")
    #             print(f"Crop shape: {cropped_array.shape}")

    #     return Image.fromarray(final_array, mode="RGB")

    # def merge_crops(
    #         self,
    #         original_img: Image.Image,
    #         results: list[tuple[Image.Image, Image.Image, tuple[int, int, int, int]]]
    #     ) -> Image.Image:
    #     """
    #     Merges multiple (mask, crop, coords) outputs back into the original image.
    #     For each result, wherever the mask is white, the pixel from the cropped image 
    #     overwrites the original.
    #     """
    #     final_array = np.array(original_img.convert("RGB"))  # shape: (H, W, 3)
    #     H, W = final_array.shape[:2]

    #     for i, (mask_pil, cropped_pil, (x1, x2, y1, y2)) in enumerate(results):
    #         mask_array = np.array(mask_pil)  # shape: (H_mask, W_mask)
    #         cropped_array = np.array(cropped_pil.convert("RGB"))  # shape: (H_crop, W_crop, 3)

    #         # Ensure dimensions match
    #         expected_height = y2 - y1
    #         expected_width = x2 - x1

    #         region = final_array[y1:y2, x1:x2]

    #         # Debug prints
    #         # print(f"---- Debug Info (Segment {i}) ----")
    #         # print(f"Original Image Size: {W} x {H}")
    #         # print(f"Bounding Box: x1={x1}, x2={x2}, y1={y1}, y2={y2}")
    #         # print(f"Expected Region Size: {expected_width} x {expected_height}")
    #         # print(f"Mask Size: {mask_array.shape}")
    #         # print(f"Cropped Size: {cropped_array.shape}")
    #         # print(f"Region Size: {region.shape}\n")

    #         # Fix dimension mismatch
    #         if mask_array.shape[:2] != region.shape[:2]:
    #             print(f"Warning: Mask shape {mask_array.shape[:2]} does not match region shape {region.shape[:2]}!")
    #             mask_array = np.resize(mask_array, region.shape[:2])  # Resize mask
    #         if cropped_array.shape[:2] != region.shape[:2]:
    #             print(f"Warning: Cropped shape {cropped_array.shape[:2]} does not match region shape {region.shape[:2]}!")
    #             cropped_array = np.resize(cropped_array, region.shape[:2])  # Resize crop

    #         white_pixels = (mask_array > 127)  # True where mask is white

    #         try:
    #             region[white_pixels] = cropped_array[white_pixels]
    #         except IndexError as e:
    #             print(f"IndexError: {e}")
    #             print(f"Region shape: {region.shape}")
    #             print(f"Mask shape: {mask_array.shape}")
    #             print(f"Cropped shape: {cropped_array.shape}")
    #             raise e  # Rethrow the error after printing debug info

    #     final_image = Image.fromarray(final_array, mode="RGB")
    #     return final_image
