import os
import cv2
import numpy as np
import torch
from scripts.detector import object_detection  # object detection
from segment_anything import sam_model_registry, SamPredictor
from bm3d import bm3d

def load(path):
    _, extension = os.path.splitext(path)
    if extension.lower() == '.png':
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Error loading PNG image.")

        pixel_size = None  # PNG does not have a standard metadata field like the others

        # Normalize the image to the range [0, 255]
        img = (img / img.max()) * 255

        # Ensure the image is in the proper format for YOLO: 3 channels (RGB)
        img = cv2.normalize(img, None, 0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Invert the image (for YOLO, it's often easier to work with positive values)
        img_x = cv2.bitwise_not(img)

    # Resize the image to the expected size for YOLO input (e.g., 640x640)
    img = cv2.resize(img, (640, 640))  # Resize to 640x640 or whatever your YOLO model uses

    return img_x, img, pixel_size

class BoundingBoxSAMProcessor:
    def __init__(self, input_root_dir, output_root_dir, sam_checkpoint="sam_vit_h_4b8939.pth"):
        self.input_root_dir = input_root_dir
        self.output_root_dir = output_root_dir
        
        # Create two new directories
        self.bbox_output_dir = os.path.join(self.output_root_dir, "bounding_boxes")
        self.segmentation_output_dir = os.path.join(self.output_root_dir, "segmentations")
        os.makedirs(self.bbox_output_dir, exist_ok=True)
        os.makedirs(self.segmentation_output_dir, exist_ok=True)

        self.output_txt = os.path.join(self.output_root_dir, 'detected_boxes.txt')
        
        # Load SAM model
        self.sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.sam_model.to(device="cuda" if torch.cuda.is_available() else "cpu")
        self.sam_predictor = SamPredictor(self.sam_model)
    
    def process_images(self):
        with open(self.output_txt, 'w') as f:
            for subdir in ['train', 'test', 'valid']:
                current_dir = os.path.join(self.input_root_dir, subdir)
                if not os.path.exists(current_dir):
                    continue
                for filename in os.listdir(current_dir):
                    if filename.endswith('.png') or filename.endswith('.jpg'):
                        path_to_image = os.path.join(current_dir, filename)
                        
                        img_x, img, pixel_size = load(path_to_image)
                        H_orig, W_orig = img.shape[:2]
                        
                        print(f"Processing {filename} from {subdir}")
                        print(f"Original Image Size: Width={W_orig}, Height={H_orig}")
                        
                        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

                        # Normalize grayscale image
                        img_float = img_gray.astype(np.float32) / 255.0

                        # Apply BM3D denoising
                        img_bm3d = bm3d(img_float, sigma_psd=0.2)
                        img_bm3d_uint8 = np.clip(img_bm3d * 255.0, 0, 255).astype(np.uint8)

                        # Apply CLAHE after denoising
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                        img_clahe = clahe.apply(img_bm3d_uint8)

                        # Convert to RGB for object detection
                        img_bm3d_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)

                        bbox_output_path = os.path.join(self.bbox_output_dir, filename)
                        boxes = object_detection(img_bm3d_rgb, img, bbox_output_path, img_size=1024, pred_score=0.1, overlap_thr=0.2, save=True)
                        
                        f.write(f"File: {filename}\n")
                        if boxes is not None:
                            f.write(f"Detected {len(boxes)} objects:\n")
                            for i, box in enumerate(boxes):
                                f.write(f"Box {i+1}: {box}\n")
                        else:
                            f.write("No objects detected.\n")
                        f.write("\n")
                        
                        if boxes is not None and len(boxes) > 0:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 else img
                            self.sam_predictor.set_image(img_rgb)
                            
                            masks = []
                            for box in boxes:
                                box_np = np.array(box, dtype=np.float32).reshape(1, 4)
                                mask, _, _ = self.sam_predictor.predict(box=box_np, multimask_output=False)
                                masks.append(mask)
                            
                            mask_overlay = np.zeros_like(img_rgb)
                            for mask in masks:
                                mask_overlay[mask[0] > 0] = (0, 255, 0)
                            
                            # Save segmentation map
                            segmentation_output_path = os.path.join(self.segmentation_output_dir, filename)
                            cv2.imwrite(segmentation_output_path, mask_overlay)
                            print(f"Saved segmentation mask: {segmentation_output_path}")
        
        print(f"All images processed. Box coordinates and segmentations saved in {self.output_txt}")

processor = BoundingBoxSAMProcessor(
    input_root_dir='/path/to/split_data',
    output_root_dir='/path/to/output_directory'
)
processor.process_images()
