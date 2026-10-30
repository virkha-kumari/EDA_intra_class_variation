"""
⚠️  DEPRECATED — NOT MAINTAINED. Kept for reference only. This post-training workflow (and
postannotation_scripts/1. yolo_model_crop_bbox_per_class.py) is no longer used, updated, or
tested; it predates the current fixes and has known issues (crops overwriting each other,
sample count not met). Do not use its output for analysis.

Master script to run entire intra and inter class variation EDA pipeline.
Includes memory management and cooling breaks for laptop GPUs.

TIME-TAKEN:
~ 18 to 20 minutes end-to-end on RTX 4060 laptop GPU for a ~22-minute video (10.8k frames sampled every 3rd frame).
"""

import subprocess
import sys
import gc
import time
import psutil
import os
import torch

print("⚠️  DEPRECATED: master_script_dinov2_PostTrain.py is not maintained — reference only, see docstring.")


def clear_memory():
    """Aggressive memory cleanup."""
    print("\n🧹 Clearing memory...")
    
    # Python garbage collection
    gc.collect()
    
    # Clear PyTorch cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Get GPU memory stats
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    # Get system memory stats
    mem = psutil.virtual_memory()
    print(f"   RAM Usage: {mem.percent}% ({mem.used/1e9:.1f}GB / {mem.total/1e9:.1f}GB)")
    print("   ✅ Memory cleared\n")

def cooling_break(seconds):
    """Pause execution to let GPU cool down."""
    print(f"\n❄️  Cooling break: {seconds} seconds...")
    print("   This prevents thermal throttling on your laptop GPU")
    
    for remaining in range(seconds, 0, -5):
        print(f"   ⏳ {remaining} seconds remaining...", end='\r')
        time.sleep(5)
    
    print("\n   ✅ Cooling break complete\n")

def check_system_resources():
    """Check if system has enough resources to continue."""
    mem = psutil.virtual_memory()
    
    if mem.percent > 90:
        print("⚠️  WARNING: RAM usage is very high (>90%)")
        print("   Consider closing other applications")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        usage_percent = (allocated / total) * 100
        
        if usage_percent > 85:
            print("⚠️  WARNING: GPU memory usage is very high (>85%)")
            print("   The script will attempt to clear cache")
            clear_memory()

def run_step(script, args, cool_down_after=True):
    """Run a pipeline step with memory management."""
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print(f"{'='*60}\n")
    
    # Check resources before starting
    check_system_resources()
    
    # Run the script
    result = subprocess.run([sys.executable, script] + args)
    
    if result.returncode != 0:
        print(f"❌ Error in {script}")
        clear_memory()  # Clean up even on failure
        sys.exit(1)
    
    print(f"\n✅ {script} completed successfully")
    
    # Post-execution cleanup
    if cool_down_after:
        clear_memory()
        cooling_break(10)  # 10 second break between steps

def main():
    ## ---------------------------------##
    ##         GLOBAL VARIABLES         ##
    ## ---------------------------------##
    RESULTS_DIR = "posttrain_results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\n📂  All results will be saved under: {os.path.abspath(RESULTS_DIR)}/")

    model_path = r"path/to/your/model.pt"  # Replace with your YOLO model path
    video_path = r"path/to/your/video.mp4"  # Replace with your video file path
    classes_space_separated = ["class0", "class1", "class2"]  # Replace with your class names
    per_class_num_frames = 1000
    conf_thresh = 0.4
    frame_stride_per_video = 3
    cropped_bbox_dir = os.path.join(RESULTS_DIR, "cropped_images")

    batch_size = 32
    save_suffix = "embeddings_dinov2.npy"

    epsilon = 0.15
    min_pts = 3
    output_cluster_dir = os.path.join(RESULTS_DIR, "clustering_results")
    max_cluster_samples = 5


    # GPU specs print
    print("="*60)
    print("INTRA-CLASS VARIATION EDA PIPELINE")
    print("="*60)
    
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {gpu_props.name}")
        print(f"  VRAM: {gpu_props.total_memory/1e9:.1f}GB")
    else:
        print("  GPU: Not available or PyTorch not installed")
    
    print("="*60)

    # Confirm before starting
    print("\nPipeline Configuration:")
    print(f"  Video: {os.path.basename(video_path)} (22 mins)")
    print(f"  Classes: {', '.join(classes_space_separated)}")
    print(f"  Frames per class: {per_class_num_frames}")
    print(f"  Total estimated frames: {int(per_class_num_frames) * len(classes_space_separated)}")


    # Step 1: Crop YOLO bboxes
    print("\n" + "="*60)
    print("STEP 1/3: EXTRACTING BOUNDING BOXES FROM VIDEO")
    print("⚠️  This step is time-consuming")
    print("="*60)
    run_step("postannotation_scripts/1. yolo_model_crop_bbox_per_class.py", [
        "--model", model_path,
        "--video", video_path,
        "--classes"] + classes_space_separated + [
        "--num_frames", str(per_class_num_frames),
        "--output", cropped_bbox_dir,
        "--conf_thresh", str(conf_thresh),
        "--frame_stride", str(frame_stride_per_video)
    ], cool_down_after=True)

    
    # Step 2: Embed the cropped YOLO bbox
    print("\n" + "="*60)
    print("STEP 2/3: GENERATING DINOV2 EMBEDDINGS")
    print("="*60)
    print("⚠️  This step is GPU-intensive")
    print(f"   Processing ~{int(per_class_num_frames) * len(classes_space_separated)} images")
    
    run_step("postannotation_scripts/2. save_dinov2_embeddings_per_class.py", [
        "--root", cropped_bbox_dir,
        "--batch", str(batch_size),
        "--save_suffix", save_suffix
    ], cool_down_after=True)

    
    # Step 3: Cluster using DBSCAN to visualize
    print("\n" + "="*60)
    print("STEP 3/3: CLUSTERING ANALYSIS")
    print("="*60)
    run_step("postannotation_scripts/3. clustering_of_classes_embeddings.py", [
        "--root", cropped_bbox_dir,
        "--eps", str(epsilon),
        "--min_samples", str(min_pts),
        "--output_dir", output_cluster_dir,
        "--max_samples", str(max_cluster_samples),
        "--save_suffix", save_suffix,
        "--auto_tune",
        "--save_montage",
    ], cool_down_after=False)  # No cooling needed after last step

    # Final cleanup
    clear_memory()
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)
    print("\nOutput Locations:")
    print(f"  Cropped Images: {cropped_bbox_dir}/")
    print(f"  Clustering Results: {output_cluster_dir}/")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        clear_memory()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        clear_memory()
        sys.exit(1)