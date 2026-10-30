#!/usr/bin/env python3
"""
⚠️  DEPRECATED — NOT MAINTAINED (post-training workflow only; see master_script_dinov2_PostTrain.py).
Known issues left unfixed: multiple detections of a class in one frame overwrite each other
(frame_{idx}.png), and --num_frames is not met after frame dedup.

YOLO model inference script that:
1. Processes video frames and collects annotations for specified classes
2. Smart samples frames to get evenly distributed 100 frames per class
3. Crops and saves bounding boxes from selected frames

Usage:
    python yolo_model_crop_bbox.py \
        --model path/to/model.pt \
        --video input.mp4 \
        --classes class1 class2 \
        --num_frames 100 \
        --output cropped_images

This script:
 - Loads a YOLOv11 model via ultralytics.YOLO
 - Runs inference per-frame on the provided video
 - For each detection whose class name is in the provided list:
     - crops the bounding box from the frame
     - saves crop to output_dir/<class_name>/<index>.png
"""

import os
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def collect_annotations(model, video_path, target_classes, conf_thresh, frame_stride):
    """
    Process video and collect all annotations for target classes.
    Args:
        model: YOLO model
        video_path: Path to video file
        target_classes: Set of class names to detect
        conf_thresh: Confidence threshold for detections
        frame_stride: Process every Nth frame (default=3 for 1/3 of frames)
    Returns: dict {frame_num: [(class_name, x1, y1, x2, y2), ...]}
    """
    annotations = defaultdict(list)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Collecting annotations (processing every {frame_stride}th frame)...")
    for frame_idx in tqdm(range(0, total_frames, frame_stride)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, conf=conf_thresh, verbose=False)[0]
        
        for box in results.boxes:
            cls = results.names[int(box.cls)]
            if cls in target_classes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                annotations[frame_idx].append((cls, int(x1), int(y1), int(x2), int(y2)))
                
    cap.release()
    return annotations

def sample_frames_per_class(annotations, target_classes, num_frames):
    """
    Smart sample frames to get even distribution for each class.
    Returns: dict {class_name: set(frame_numbers)}
    """
    class_frames = {cls: [] for cls in target_classes}
    
    # Collect all frame numbers per class
    for frame_num, boxes in annotations.items():
        for (cls, *_) in boxes:
            class_frames[cls].append(frame_num)
    
    selected_frames = {cls: set() for cls in target_classes}
    
    print("\nSampling frames per class:")
    for cls in target_classes:
        total = len(class_frames[cls])
        if total == 0:
            print(f"Warning: No annotations found for class {cls}")
            continue
            
        stride = max(1, total / num_frames)
        print(f"{cls}: Found {total} annotations, using stride {stride:.2f}")
        
        # Use numpy's linspace for even sampling
        if total > num_frames:
            indices = np.linspace(0, len(class_frames[cls])-1, num_frames, dtype=int)
            frames = [class_frames[cls][i] for i in indices]
        else:
            frames = class_frames[cls]
            
        selected_frames[cls].update(frames)
        
    return selected_frames

def crop_and_save_boxes(video_path, annotations, selected_frames, output_root):
    """
    Crop and save bounding boxes for selected frames.
    """
    output_root = Path(output_root)
    
    # Create output directories
    class_dirs = {cls: output_root / cls for cls in selected_frames.keys()}
    for d in class_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    # Process video
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    
    print("\nCropping and saving boxes...")
    with tqdm(total=sum(len(frames) for frames in selected_frames.values())) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Check if this frame was selected for any class
            process_frame = False
            for frames in selected_frames.values():
                if frame_idx in frames:
                    process_frame = True
                    break
                    
            if process_frame and frame_idx in annotations:
                for cls, x1, y1, x2, y2 in annotations[frame_idx]:
                    if frame_idx in selected_frames[cls]:
                        # Crop and save
                        crop = frame[y1:y2, x1:x2]
                        if 0 not in crop.shape:
                            out_path = class_dirs[cls] / f"frame_{frame_idx:06d}.png"
                            cv2.imwrite(str(out_path), crop)
                            pbar.update(1)
            
            frame_idx += 1
            
    cap.release()

def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 video -> smart sampled cropped images by class")
    p.add_argument("--model", required=True, help="Path to YOLO model")
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument("--classes", nargs="+", required=True, help="List of class names to detect and crop")
    p.add_argument("--num_frames", type=int, default=100, help="Number of frames to sample per class")
    p.add_argument("--output", default="cropped_images", help="Output directory")
    p.add_argument("--conf_thresh", type=float, default=0.4, help="Confidence threshold (0-1)")
    p.add_argument("--frame_stride", type=int, default=3, help="Process every Nth frame (default=3)")
    return p.parse_args()

def main():
    args = parse_args()
    
    # Prepare output folder
    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model {args.model}...")
    model = YOLO(args.model)
    
    # Process video and collect annotations
    annotations = collect_annotations(
        model=model,
        video_path=args.video,
        target_classes=set(args.classes),
        conf_thresh=args.conf_thresh,
        frame_stride=args.frame_stride
    )
    print(f"Collected annotations from {len(annotations)} frames (every {args.frame_stride}th frame)")
    
    # Smart sample frames for each class
    selected_frames = sample_frames_per_class(annotations, args.classes, args.num_frames)
    
    # Crop and save
    crop_and_save_boxes(args.video, annotations, selected_frames, args.output)
    print("\nDone! Cropped images saved in:", args.output)

    # After crop_and_save_boxes in main():
    print("\n" + "="*60)
    print("SUMMARY:")
    for cls in args.classes:
        saved_count = len(list((output_dir / cls).glob("*.png")))
        print(f"  {cls}: {saved_count} images saved")
    print("="*60)

if __name__ == "__main__":
    main()