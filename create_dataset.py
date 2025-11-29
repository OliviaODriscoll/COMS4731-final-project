"""
Create organized dataset with train/val/test splits based on timestamp.

Splits data at 1:59 (dancer change), processes all frames, and organizes
2D skeletons from each view plus 3D triangulated skeletons.
"""

import cv2
import numpy as np
import pandas as pd
import os
import shutil
import argparse
from pathlib import Path
from typing import Dict, Tuple
from mediapipe_2d_skeleton import MediaPipe2DSkeleton, process_video_2d_skeleton
from triangulate_3d_skeleton import MultiViewTriangulator, load_2d_skeleton


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert timestamp string (MM:SS or MM:SS.S) to seconds."""
    parts = timestamp.split(':')
    return int(parts[0]) * 60 + float(parts[1])


def seconds_to_frame(seconds: float, fps: float) -> int:
    """Convert seconds to frame number."""
    return int(seconds * fps)


def get_video_fps(video_path: str) -> float:
    """Get FPS from video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 30.0  # Default
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0 or np.isnan(fps):
        return 30.0
    return fps


def process_all_frames_2d(video_path: str, output_dir: str, start_frame: int = 0, 
                          end_frame: int = None, model_complexity: int = 2,
                          output_frame_offset: int = 0):
    """
    Process all frames (not just every 10th) to extract 2D skeletons.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save skeleton files
        start_frame: Frame number to start processing from (in video)
        end_frame: Frame number to end processing at (in video)
        model_complexity: MediaPipe model complexity
        output_frame_offset: Offset to subtract from frame numbers when saving files
                            (used to align frames across videos)
    """
    import cv2
    os.makedirs(output_dir, exist_ok=True)
    
    skeleton_extractor = MediaPipe2DSkeleton(
        model_complexity=model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    fps = get_video_fps(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if end_frame is None:
        end_frame = total_frames
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    processed_count = 0
    no_pose_count = 0
    
    print(f"Processing frames {start_frame} to {end_frame}...")
    if output_frame_offset != 0:
        print(f"Output frame offset: {output_frame_offset} (saving as frame_{output_frame_offset:05d} onwards)")
    
    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        landmarks_2d, coords_array, frame_with_pose, visibility = skeleton_extractor.process_frame(
            frame, min_visibility=0.5, normalized=False
        )
        
        if landmarks_2d is None:
            no_pose_count += 1
            frame_count += 1
            continue
        
        # Save coordinates with aligned frame number
        output_frame_num = frame_count - output_frame_offset
        if output_frame_num >= 0:  # Only save if aligned frame number is non-negative
            coords_path = os.path.join(output_dir, f"skeleton_2d_coords_frame_{output_frame_num:05d}.npy")
            np.save(coords_path, coords_array)
            processed_count += 1
        
        if processed_count % 100 == 0:
            print(f"  Processed {processed_count} frames...")
        
        frame_count += 1
    
    cap.release()
    
    print(f"  ✓ Processed {processed_count} frames with pose detected")
    print(f"  Frames without pose: {no_pose_count}")
    return processed_count


def find_time_offset_manual(video_name: str, reference_video: str = "left") -> float:
    """
    Find time offset using manual alignment points.
    
    Alignment: arabesque_left frame 833 = barre_right frame 1071 = barre_tripod frame 1218
    
    Alignment offsets (relative to arabesque_left):
    - barre_right: ~7.94s (calculated from frame correspondence)
    - barre_tripod: ~12.83s (calculated from frame correspondence)
    
    Returns:
        Offset in seconds (positive means video starts later than reference)
    """
    # Direct offsets relative to arabesque_left (left camera)
    # Calculated from: left frame 833 = right frame 1071 = tripod frame 1218
    # At their respective FPS: left 27.80s = right 36.07s = tripod 40.60s
    offsets = {
        "left": 0.0,
        "right": 8.27,  # 36.07s - 27.80s
        "middle": 12.80  # 40.60s - 27.80s
    }
    
    return offsets.get(video_name, 0.0)


def create_dataset(
    video_paths: Dict[str, str],
    calibration_files: Dict[str, str],
    csv_path: str,
    output_base: str,
    split_time: str = "01:59",
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    reference_video: str = "left"
):
    """
    Create organized dataset with train/val/test splits.
    
    Args:
        video_paths: Dictionary mapping camera names to video paths
        calibration_files: Dictionary mapping camera names to calibration files
        csv_path: Path to label CSV file
        output_base: Base directory for dataset
        split_time: Timestamp to split train/test (format: MM:SS)
        train_split: Fraction for training (from frames before split_time)
        val_split: Fraction for validation (from frames before split_time)
        test_split: Fraction for testing (from frames after split_time)
        reference_video: Reference camera name (others will be offset relative to this)
    """
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Get FPS from reference video
    reference_video_path = video_paths.get(reference_video, list(video_paths.values())[0])
    fps = get_video_fps(reference_video_path)
    print(f"Video FPS: {fps:.2f}")
    
    # Calculate frame offsets directly from alignment point
    # Alignment: arabesque_left frame 833 = barre_right frame 1071 = barre_tripod frame 1218
    print("\n" + "="*60)
    print("Calculating frame offsets for synchronization...")
    print("="*60)
    
    # Alignment point frame numbers
    alignment_frames = {
        "left": 833,
        "right": 1071,
        "middle": 1218
    }
    
    # Calculate frame offsets (how many frames ahead each video is at alignment point)
    frame_offsets = {}
    reference_alignment_frame = alignment_frames.get(reference_video, 0)
    
    for cam_name in video_paths.keys():
        if cam_name == reference_video:
            frame_offsets[cam_name] = 0
        else:
            cam_alignment_frame = alignment_frames.get(cam_name, 0)
            # How many frames ahead is this camera at the alignment point?
            frame_offset = cam_alignment_frame - reference_alignment_frame
            frame_offsets[cam_name] = frame_offset
            print(f"{cam_name}: alignment frame {cam_alignment_frame}, offset: {frame_offset:+d} frames from reference")
    
    # Convert frame offsets to time offsets for display
    offsets = {}
    for cam_name, frame_offset in frame_offsets.items():
        video_fps = get_video_fps(video_paths[cam_name])
        time_offset = frame_offset / video_fps
        offsets[cam_name] = time_offset
        print(f"{cam_name}: {time_offset:.2f} seconds ({frame_offset:+d} frames at {video_fps:.2f} fps)")
    
    # Convert split time to frame number
    split_seconds = timestamp_to_seconds(split_time)
    split_frame = seconds_to_frame(split_seconds, fps)
    print(f"\nSplit time: {split_time} = {split_seconds}s = frame {split_frame}")
    
    # Create temporary directories for processing
    temp_2d_dir = output_base / "temp_2d_skeletons"
    temp_2d_dir.mkdir(exist_ok=True)
    
    # Step 1: Extract 2D skeletons from all views for all frames
    print("\n" + "="*60)
    print("Step 1: Extracting 2D skeletons from all views")
    print("="*60)
    
    camera_2d_dirs = {}
    for cam_name, video_path in video_paths.items():
        print(f"\nProcessing {cam_name} camera...")
        cam_2d_dir = temp_2d_dir / cam_name
        cam_2d_dir.mkdir(exist_ok=True)
        
        # Get frame offset for this camera (from alignment point)
        frame_offset = frame_offsets[cam_name]
        video_fps = get_video_fps(video_path)
        
        # Get alignment frame for this camera
        alignment_frame = alignment_frames.get(cam_name, 0)
        
        # Start processing from alignment frame (this is where all videos sync)
        # We'll save as dataset frame 0, so we need to subtract the alignment frame
        start_frame = alignment_frame
        
        print(f"  Alignment frame: {alignment_frame}")
        print(f"  Frame offset from reference: {frame_offset:+d} frames")
        print(f"  Starting from video frame: {start_frame}")
        print(f"  Output frame offset: {alignment_frame} (saving video frame {start_frame} as dataset frame 0)")
        
        process_all_frames_2d(
            video_path=video_path,
            output_dir=str(cam_2d_dir),
            start_frame=start_frame,
            end_frame=None,
            model_complexity=2,
            output_frame_offset=alignment_frame  # Subtract alignment frame to align at dataset frame 0
        )
        camera_2d_dirs[cam_name] = cam_2d_dir
    
    # Step 2: Triangulate 3D skeletons
    print("\n" + "="*60)
    print("Step 2: Triangulating 3D skeletons")
    print("="*60)
    
    temp_3d_dir = output_base / "temp_3d_skeletons"
    temp_3d_dir.mkdir(exist_ok=True)
    
    # Initialize triangulator
    triangulator = MultiViewTriangulator(calibration_files, camera_poses=None)
    
    # Find all frame numbers
    all_frames = set()
    for cam_dir in camera_2d_dirs.values():
        skeleton_files = sorted(cam_dir.glob("skeleton_2d_coords_frame_*.npy"))
        for f in skeleton_files:
            frame_num = int(f.stem.split('_')[-1])
            all_frames.add(frame_num)
    
    all_frames = sorted(all_frames)
    print(f"Found {len(all_frames)} frames across all cameras")
    
    # Triangulate all frames
    processed_3d = 0
    for frame_num in all_frames:
        # Load 2D skeletons from all views
        skeletons_2d = {}
        for cam_name, cam_dir in camera_2d_dirs.items():
            skeleton_file = cam_dir / f"skeleton_2d_coords_frame_{frame_num:05d}.npy"
            if skeleton_file.exists():
                skeleton = load_2d_skeleton(skeleton_file)
                if skeleton is not None:
                    skeletons_2d[cam_name] = skeleton
        
        # Triangulate if we have at least 2 views
        if len(skeletons_2d) >= 2:
            skeleton_3d = triangulator.triangulate_skeleton(skeletons_2d, min_views=2)
            output_path = temp_3d_dir / f"skeleton_3d_coords_frame_{frame_num:05d}.npy"
            np.save(output_path, skeleton_3d)
            processed_3d += 1
        
        if processed_3d % 100 == 0:
            print(f"  Triangulated {processed_3d} frames...")
    
    print(f"  ✓ Triangulated {processed_3d} 3D skeletons")
    
    # Step 3: Split into train/val/test
    print("\n" + "="*60)
    print("Step 3: Splitting into train/val/test")
    print("="*60)
    
    # Split frames: before split_time = train/val, after = test
    train_val_frames = [f for f in all_frames if f < split_frame]
    test_frames = [f for f in all_frames if f >= split_frame]
    
    print(f"Train/Val frames (before {split_time}): {len(train_val_frames)}")
    print(f"Test frames (after {split_time}): {len(test_frames)}")
    
    # Split train_val into train and val
    train_size = int(len(train_val_frames) * train_split)
    val_size = len(train_val_frames) - train_size
    
    train_frames = train_val_frames[:train_size]
    val_frames = train_val_frames[train_size:]
    
    print(f"Train frames: {len(train_frames)}")
    print(f"Val frames: {len(val_frames)}")
    print(f"Test frames: {len(test_frames)}")
    
    # Step 4: Organize into dataset structure
    print("\n" + "="*60)
    print("Step 4: Organizing dataset structure")
    print("="*60)
    
    splits = {
        'train': train_frames,
        'val': val_frames,
        'test': test_frames
    }
    
    for split_name, frames in splits.items():
        print(f"\nOrganizing {split_name} set ({len(frames)} frames)...")
        
        # Create directories
        split_dir = output_base / split_name
        split_2d_dir = split_dir / "2d_skeletons"
        split_3d_dir = split_dir / "3d_skeletons"
        
        split_2d_dir.mkdir(parents=True, exist_ok=True)
        split_3d_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each camera view
        for cam_name in camera_2d_dirs.keys():
            (split_2d_dir / cam_name).mkdir(exist_ok=True)
        
        # Copy files
        for frame_num in frames:
            # Copy 2D skeletons from each view
            for cam_name, cam_dir in camera_2d_dirs.items():
                src = cam_dir / f"skeleton_2d_coords_frame_{frame_num:05d}.npy"
                dst = split_2d_dir / cam_name / f"skeleton_2d_coords_frame_{frame_num:05d}.npy"
                if src.exists():
                    shutil.copy2(src, dst)
            
            # Copy 3D skeleton
            src = temp_3d_dir / f"skeleton_3d_coords_frame_{frame_num:05d}.npy"
            dst = split_3d_dir / f"skeleton_3d_coords_frame_{frame_num:05d}.npy"
            if src.exists():
                shutil.copy2(src, dst)
        
        print(f"  ✓ Created {split_name} set")
    
    # Step 5: Copy labels and create frame_labels.csv
    print("\n" + "="*60)
    print("Step 5: Organizing labels")
    print("="*60)
    
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        
        # Get alignment frame for reference video (left/arabesque_left)
        reference_alignment_frame = alignment_frames.get(reference_video, 0)
        print(f"Reference alignment frame: {reference_alignment_frame}")
        print(f"Dataset frame 0 = original video frame {reference_alignment_frame}")
        
        for split_name, frames in splits.items():
            split_dir = output_base / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Filter labels for this split (in original video frame numbers)
            split_labels = []
            for _, row in df.iterrows():
                start_sec = timestamp_to_seconds(row['start'])
                end_sec = timestamp_to_seconds(row['end'])
                start_frame_video = seconds_to_frame(start_sec, fps)
                end_frame_video = seconds_to_frame(end_sec, fps)
                
                # Convert to dataset frame numbers
                start_frame_dataset = start_frame_video - reference_alignment_frame
                end_frame_dataset = end_frame_video - reference_alignment_frame
                
                # Check if this label overlaps with split frames (in dataset frame numbers)
                if split_name == 'test':
                    if start_frame_dataset >= split_frame:
                        split_labels.append(row)
                else:  # train or val
                    if end_frame_dataset < split_frame:
                        split_labels.append(row)
            
            if split_labels:
                split_df = pd.DataFrame(split_labels)
                labels_path = split_dir / "labels.csv"
                split_df.to_csv(labels_path, index=False)
                print(f"  {split_name}: {len(split_df)} labels saved to {labels_path}")
            
            # Create frame_labels.csv for this split
            print(f"\n  Creating frame_labels.csv for {split_name}...")
            frame_label_list = []
            
            # Create mapping from dataset frame to label
            frame_to_label = {}
            for _, row in split_df.iterrows():
                start_sec = timestamp_to_seconds(row['start'])
                end_sec = timestamp_to_seconds(row['end'])
                start_frame_video = seconds_to_frame(start_sec, fps)
                end_frame_video = seconds_to_frame(end_sec, fps)
                
                # Convert to dataset frame numbers
                start_frame_dataset = start_frame_video - reference_alignment_frame
                end_frame_dataset = end_frame_video - reference_alignment_frame
                
                # Label all frames in this range
                for frame in range(start_frame_dataset, end_frame_dataset + 1):
                    if frame in frames:  # Only label frames that exist in this split
                        if frame not in frame_to_label:
                            frame_to_label[frame] = row['step']
            
            # Create frame_labels.csv with all frames in this split
            for frame in sorted(frames):
                label = frame_to_label.get(frame, "no_label")
                # Calculate reference timestamp for this frame (in arabesque_left video)
                frame_video = frame + reference_alignment_frame
                time_sec = frame_video / fps
                minutes = int(time_sec // 60)
                seconds = time_sec % 60
                reference_timestamp = f"{minutes:02d}:{seconds:05.2f}"
                
                frame_label_list.append({
                    "frame": frame,
                    "label": label,
                    "reference_timestamp": reference_timestamp
                })
            
            frame_labels_df = pd.DataFrame(frame_label_list)
            frame_labels_path = split_dir / "frame_labels.csv"
            frame_labels_df.to_csv(frame_labels_path, index=False)
            
            labeled_count = sum(1 for f in frame_to_label.keys() if f in frames)
            print(f"    Saved {len(frame_labels_df)} frame labels ({labeled_count} labeled, {len(frame_labels_df) - labeled_count} unlabeled)")
            print(f"    Saved to {frame_labels_path}")
    
    # Cleanup temporary directories
    print("\n" + "="*60)
    print("Cleaning up temporary files...")
    print("="*60)
    shutil.rmtree(temp_2d_dir)
    shutil.rmtree(temp_3d_dir)
    print("  ✓ Cleaned up")
    
    print("\n" + "="*60)
    print("Dataset creation complete!")
    print("="*60)
    print(f"\nDataset structure:")
    print(f"  {output_base}/")
    print(f"    train/")
    print(f"      2d_skeletons/ (left/, middle/, right/)")
    print(f"      3d_skeletons/")
    print(f"      labels.csv")
    print(f"      frame_labels.csv (frame, label, reference_timestamp)")
    print(f"    val/")
    print(f"      2d_skeletons/ (left/, middle/, right/)")
    print(f"      3d_skeletons/")
    print(f"      labels.csv")
    print(f"      frame_labels.csv")
    print(f"    test/")
    print(f"      2d_skeletons/ (left/, middle/, right/)")
    print(f"      3d_skeletons/")
    print(f"      labels.csv")
    print(f"      frame_labels.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Create organized dataset with train/val/test splits"
    )
    
    parser.add_argument("--video-dir", type=str,
                       default="/Users/olivia/Documents/COMS4731/COMS4731-final-project/Raw Data",
                       help="Directory containing video files")
    parser.add_argument("--calibration-dir", type=str,
                       default="/Users/olivia/Desktop/COMS4731-final-project/calibration",
                       help="Directory containing calibration files")
    parser.add_argument("--csv", type=str,
                       default="/Users/olivia/Desktop/COMS4731-final-project/barre_tripod.csv",
                       help="Path to label CSV file")
    parser.add_argument("--output", type=str,
                       default="/Users/olivia/Desktop/COMS4731-final-project/dataset",
                       help="Output dataset directory")
    parser.add_argument("--split-time", type=str, default="01:59",
                       help="Timestamp to split train/test (MM:SS format)")
    parser.add_argument("--train-split", type=float, default=0.7,
                       help="Fraction of pre-split data for training")
    parser.add_argument("--val-split", type=float, default=0.15,
                       help="Fraction of pre-split data for validation")
    parser.add_argument("--reference-video", type=str, default="arabesque_left",
                       help="Reference video name (arabesque_left, barre_right, or barre_tripod)")
    
    args = parser.parse_args()
    
    # Map video names to camera names
    video_mapping = {
        "arabesque_left": "left",
        "barre_tripod": "middle",
        "barre_right": "right"
    }
    
    video_dir = Path(args.video_dir)
    calibration_dir = Path(args.calibration_dir)
    
    # Find videos
    video_paths = {}
    calibration_files = {}
    
    for video_name, cam_name in video_mapping.items():
        # Find video file
        video_file = None
        for ext in ['.mov', '.MOV', '.mp4', '.MP4']:
            candidate = video_dir / f"{video_name}{ext}"
            if candidate.exists():
                video_file = candidate
                break
        
        if video_file is None:
            print(f"Warning: Video not found for {video_name}")
            continue
        
        # Find calibration file
        calib_file = calibration_dir / f"calibration_{cam_name}.npz"
        if not calib_file.exists():
            print(f"Warning: Calibration not found for {cam_name}")
            continue
        
        video_paths[cam_name] = str(video_file)
        calibration_files[cam_name] = str(calib_file)
    
    if len(video_paths) < 2:
        print("Error: Need at least 2 camera views")
        return
    
    print("Videos found:")
    for cam_name, path in video_paths.items():
        print(f"  {cam_name}: {path}")
    
    print("\nCalibration files:")
    for cam_name, path in calibration_files.items():
        print(f"  {cam_name}: {path}")
    
    # Map reference video name to camera name
    reference_camera = video_mapping.get(args.reference_video, "left")
    
    # Create dataset
    create_dataset(
        video_paths=video_paths,
        calibration_files=calibration_files,
        csv_path=args.csv,
        output_base=args.output,
        split_time=args.split_time,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=1.0 - args.train_split - args.val_split,
        reference_video=reference_camera
    )


if __name__ == "__main__":
    main()

