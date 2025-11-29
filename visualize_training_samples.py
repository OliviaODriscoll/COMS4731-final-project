"""
Visualize training samples with 2D skeletons from 3 views, 3D skeleton, and labels.

Shows a comprehensive view of the training data including:
- 2D skeletons from each camera angle
- 3D triangulated skeleton
- Ground truth movement label
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path
import os
from typing import Optional, Tuple

# MediaPipe landmark connections
LANDMARK_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Torso
    (23, 24),  # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
]


def draw_skeleton_2d(image: np.ndarray, skeleton_2d: np.ndarray, 
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2, original_image_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Draw 2D skeleton on image.
    
    Args:
        image: Image to draw on (or None to create blank)
        skeleton_2d: Skeleton coordinates (shape: [num_joints, 2]) in pixel coordinates from original video
        color: Color for skeleton lines
        thickness: Line thickness
        original_image_size: (width, height) of original video frame when skeleton was extracted.
                           If None, assumes skeleton coords match current image size.
    """
    if image is not None:
        img = image.copy()
        img_h, img_w = img.shape[:2]
        
        # Scale skeleton coordinates if image size differs from original
        valid_points = skeleton_2d[~np.isnan(skeleton_2d).any(axis=1)]
        if len(valid_points) > 0:
            if original_image_size is not None:
                orig_w, orig_h = original_image_size
                # Scale from original size to current image size
                scale_x = img_w / orig_w
                scale_y = img_h / orig_h
                skeleton_scaled = skeleton_2d.copy()
                skeleton_scaled[:, 0] *= scale_x
                skeleton_scaled[:, 1] *= scale_y
                # Debug output for middle view
                if scale_x != 1.0 or scale_y != 1.0:
                    print(f"    Scaling skeleton: {orig_w}x{orig_h} -> {img_w}x{img_h} (scale: {scale_x:.3f}x, {scale_y:.3f}x)")
            else:
                # Check if coordinates are normalized (0-1) or pixel coordinates
                max_coord = valid_points.max()
                min_coord = valid_points.min()
                if max_coord <= 1.0 and min_coord >= 0.0:
                    # Normalized coordinates, scale to image size
                    skeleton_scaled = skeleton_2d.copy()
                    skeleton_scaled[:, 0] *= img_w
                    skeleton_scaled[:, 1] *= img_h
                else:
                    # Assume pixel coordinates - check if they match image size
                    max_x = valid_points[:, 0].max()
                    max_y = valid_points[:, 1].max()
                    if max_x > img_w * 1.1 or max_y > img_h * 1.1:
                        # Coordinates are larger than image - need to scale down
                        scale_x = img_w / max_x if max_x > 0 else 1.0
                        scale_y = img_h / max_y if max_y > 0 else 1.0
                        skeleton_scaled = skeleton_2d.copy()
                        skeleton_scaled[:, 0] *= scale_x
                        skeleton_scaled[:, 1] *= scale_y
                        print(f"    Auto-scaling skeleton: coords max ({max_x:.1f}, {max_y:.1f}) > image ({img_w}, {img_h})")
                    else:
                        # Assume pixel coordinates match current image
                        skeleton_scaled = skeleton_2d.copy()
        else:
            skeleton_scaled = skeleton_2d.copy()
    else:
        # Create blank image with skeleton bounds
        valid_points = skeleton_2d[~np.isnan(skeleton_2d).any(axis=1)]
        if len(valid_points) > 0:
            min_x, min_y = valid_points.min(axis=0).astype(int)
            max_x, max_y = valid_points.max(axis=0).astype(int)
            width = max(640, max_x - min_x + 100)
            height = max(480, max_y - min_y + 100)
            img = np.zeros((height, width, 3), dtype=np.uint8)
            skeleton_scaled = skeleton_2d.copy()
        else:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            skeleton_scaled = skeleton_2d.copy()
    
    # Draw connections
    for start_idx, end_idx in LANDMARK_CONNECTIONS:
        if (start_idx < len(skeleton_scaled) and end_idx < len(skeleton_scaled)):
            pt1 = skeleton_scaled[start_idx]
            pt2 = skeleton_scaled[end_idx]
            
            if not np.isnan(pt1).any() and not np.isnan(pt2).any():
                pt1 = (int(pt1[0]), int(pt1[1]))
                pt2 = (int(pt2[0]), int(pt2[1]))
                # Make sure points are within image bounds
                if (0 <= pt1[0] < img.shape[1] and 0 <= pt1[1] < img.shape[0] and
                    0 <= pt2[0] < img.shape[1] and 0 <= pt2[1] < img.shape[0]):
                    cv2.line(img, pt1, pt2, color, thickness)
    
    # Draw joints
    for i, pt in enumerate(skeleton_scaled):
        if not np.isnan(pt).any():
            pt = (int(pt[0]), int(pt[1]))
            # Make sure point is within image bounds
            if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
                cv2.circle(img, pt, 5, (0, 0, 255), -1)
    
    return img


def visualize_3d_skeleton(skeleton_3d: np.ndarray, ax, title: str = "3D Skeleton"):
    """Visualize 3D skeleton on given axes."""
    # Extract valid points
    valid_points = []
    valid_indices = []
    for i, pt in enumerate(skeleton_3d):
        if not np.isnan(pt).any():
            valid_points.append(pt)
            valid_indices.append(i)
    
    if len(valid_points) == 0:
        ax.text(0.5, 0.5, 0.5, "No valid points", fontsize=12)
        return
    
    valid_points = np.array(valid_points)
    
    # Plot points
    ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], 
               c='red', s=50, alpha=0.8)
    
    # Draw connections
    for start_idx, end_idx in LANDMARK_CONNECTIONS:
        if (start_idx < len(skeleton_3d) and end_idx < len(skeleton_3d)):
            pt1 = skeleton_3d[start_idx]
            pt2 = skeleton_3d[end_idx]
            
            if not np.isnan(pt1).any() and not np.isnan(pt2).any():
                x_line = [pt1[0], pt2[0]]
                y_line = [pt1[1], pt2[1]]
                z_line = [pt1[2], pt2[2]]
                ax.plot(x_line, y_line, z_line, 'b-', linewidth=2, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect
    if len(valid_points) > 0:
        max_range = np.array([
            valid_points[:, 0].max() - valid_points[:, 0].min(),
            valid_points[:, 1].max() - valid_points[:, 1].min(),
            valid_points[:, 2].max() - valid_points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (valid_points[:, 0].max() + valid_points[:, 0].min()) * 0.5
        mid_y = (valid_points[:, 1].max() + valid_points[:, 1].min()) * 0.5
        mid_z = (valid_points[:, 2].max() + valid_points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert timestamp string (MM:SS or MM:SS.S) to seconds."""
    parts = timestamp.split(':')
    return int(parts[0]) * 60 + float(parts[1])


def find_time_offset(csv1_path: str, csv2_path: str, reference_step: str = 'fifth position') -> float:
    """Find time offset between two videos by matching a reference step."""
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    
    step1 = df1[df1['step'] == reference_step]
    step2 = df2[df2['step'] == reference_step]
    
    if len(step1) == 0 or len(step2) == 0:
        return 0.0
    
    time1 = timestamp_to_seconds(step1.iloc[0]['start'])
    time2 = timestamp_to_seconds(step2.iloc[0]['start'])
    
    return time2 - time1


def get_label_for_frame(frame_num: int, labels_df: pd.DataFrame, fps: float = 30.0) -> Optional[str]:
    """Get ground truth label for a frame number."""
    def seconds_to_frame(sec: float) -> int:
        return int(sec * fps)
    
    for _, row in labels_df.iterrows():
        start_sec = timestamp_to_seconds(row['start'])
        end_sec = timestamp_to_seconds(row['end'])
        start_frame = seconds_to_frame(start_sec)
        end_frame = seconds_to_frame(end_sec)
        
        if start_frame <= frame_num <= end_frame:
            return row['step']
    
    return None


def get_video_frame_offsets(csv_dir: Path, reference_video: str = "arabesque_left", fps: float = 30.0) -> dict:
    """
    Calculate frame offsets for each camera view.
    
    Returns offsets that need to be ADDED to dataset frame numbers to get actual video frame numbers.
    Dataset frames are aligned (all start at 0), but videos have different start times.
    
    Alignment: arabesque_left frame 833 = barre_right frame 1071 = barre_tripod frame 1218
    
    Alignment offsets (determined by frame correspondence):
    - left (arabesque_left): offset = 0
    - right (barre_right): offset = +8.27 seconds
    - middle (barre_tripod): offset = +12.80 seconds
    """
    video_to_camera = {
        "arabesque_left": "left",
        "barre_tripod": "middle",
        "barre_right": "right"
    }
    
    offsets = {}
    
    # Use the exact alignment offsets (determined by visual inspection):
    # barre_right: +8.30s relative to arabesque_left
    # barre_tripod: +12.80s relative to arabesque_left
    
    for video_name, cam_name in video_to_camera.items():
        if video_name == reference_video:
            offsets[cam_name] = 0
        elif video_name == "barre_right":
            # barre_right offset: +8.27s (from frame correspondence: left 833 = right 1071)
            offset_sec = 8.27
            offsets[cam_name] = int(offset_sec * fps)
        elif video_name == "barre_tripod":
            # barre_tripod offset: +12.80s (from frame correspondence: left 833 = tripod 1218)
            offset_sec = 12.80
            offsets[cam_name] = int(offset_sec * fps)
    
    return offsets


def visualize_training_sample(
    dataset_dir: Path,
    split: str,
    frame_num: int,
    video_paths: Optional[dict] = None,
    save_path: Optional[str] = None
):
    """
    Visualize a single training sample with all views.
    
    Args:
        dataset_dir: Base dataset directory
        split: 'train', 'val', or 'test'
        frame_num: Frame number to visualize
        video_paths: Optional dict mapping camera names to video paths (for original images)
        save_path: Optional path to save visualization
    """
    split_dir = dataset_dir / split
    
    # Load 2D skeletons from each view
    # Note: Skeletons should already be aligned in the dataset (same frame_num = same moment in time)
    # This is ensured by create_dataset.py which uses output_frame_offset to align frame numbers
    skeleton_2d_left = None
    skeleton_2d_middle = None
    skeleton_2d_right = None
    
    left_file = split_dir / "2d_skeletons" / "left" / f"skeleton_2d_coords_frame_{frame_num:05d}.npy"
    middle_file = split_dir / "2d_skeletons" / "middle" / f"skeleton_2d_coords_frame_{frame_num:05d}.npy"
    right_file = split_dir / "2d_skeletons" / "right" / f"skeleton_2d_coords_frame_{frame_num:05d}.npy"
    
    print(f"\n  Loading skeletons for dataset frame {frame_num}:")
    if left_file.exists():
        skeleton_2d_left = np.load(left_file)
        print(f"    Left: {left_file.name}")
    else:
        print(f"    Warning: Left skeleton file not found: {left_file}")
    if middle_file.exists():
        skeleton_2d_middle = np.load(middle_file)
        print(f"    Middle: {middle_file.name}")
    else:
        print(f"    Warning: Middle skeleton file not found: {middle_file}")
    if right_file.exists():
        skeleton_2d_right = np.load(right_file)
        print(f"    Right: {right_file.name}")
    else:
        print(f"    Warning: Right skeleton file not found: {right_file}")
    
    # Load 3D skeleton
    skeleton_3d = None
    skeleton_3d_file = split_dir / "3d_skeletons" / f"skeleton_3d_coords_frame_{frame_num:05d}.npy"
    if skeleton_3d_file.exists():
        skeleton_3d = np.load(skeleton_3d_file)
    
    # Load label
    label = None
    labels_file = split_dir / "labels.csv"
    if labels_file.exists():
        labels_df = pd.read_csv(labels_file)
        label = get_label_for_frame(frame_num, labels_df)
    
    # Try to load original images if video paths provided
    # Need to account for time offsets between videos
    # Dataset frames are aligned, but videos have different start times
    images = {}
    original_sizes = {}  # Store original video dimensions for skeleton scaling
    if video_paths:
        # Get frame offsets (assuming reference is middle/barre_tripod)
        csv_dir = dataset_dir.parent  # Assuming CSVs are in parent directory
        
        # Get actual FPS from one of the videos
        fps = 30.0
        if video_paths:
            cap_temp = cv2.VideoCapture(list(video_paths.values())[0])
            if cap_temp.isOpened():
                fps = cap_temp.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or np.isnan(fps):
                    fps = 30.0
                cap_temp.release()
        
        try:
            offsets = get_video_frame_offsets(csv_dir, reference_video="arabesque_left", fps=fps)
        except Exception as e:
            print(f"  Warning: Could not calculate offsets: {e}")
            offsets = {"left": 0, "middle": 0, "right": 0}
        
        # Get original video dimensions for skeleton scaling
        original_sizes = {}
        for cam_name, video_path in video_paths.items():
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                original_sizes[cam_name] = (orig_w, orig_h)
                cap.release()
        
        for cam_name, video_path in video_paths.items():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"  Warning: Could not open video {video_path} for {cam_name}")
                continue
                
            # Dataset frame numbers are aligned (all start at 0)
            # But actual video frames need offset added back
            offset = offsets.get(cam_name, 0)
            video_frame = frame_num + offset
            video_frame = max(0, video_frame)  # Don't go negative
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if video_frame >= total_frames:
                print(f"  Warning: Frame {video_frame} exceeds video length ({total_frames}) for {cam_name}")
                cap.release()
                continue
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
            ret, frame = cap.read()
            if ret and frame is not None:
                images[cam_name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                print(f"  Loaded video frame {video_frame} from {cam_name} (dataset frame {frame_num}, offset {offset} frames)")
                print(f"    This should match skeleton from dataset frame {frame_num}")
            else:
                print(f"  Warning: Could not read frame {video_frame} from {cam_name} (dataset frame {frame_num}, offset {offset})")
            cap.release()
    
    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    
    # Top row: 2D skeletons from 3 views
    if skeleton_2d_left is not None:
        ax1 = fig.add_subplot(2, 3, 1)
        img = images.get('left') if images else None
        orig_size = original_sizes.get('left') if 'original_sizes' in locals() else None
        if img is not None and orig_size is not None:
            print(f"  Left: image size {img.shape[1]}x{img.shape[0]}, original size {orig_size[0]}x{orig_size[1]}")
        img_with_skeleton = draw_skeleton_2d(img, skeleton_2d_left, original_image_size=orig_size)
        ax1.imshow(img_with_skeleton)
        ax1.set_title('Left View (2D Skeleton)', fontsize=12, fontweight='bold')
        ax1.axis('off')
    
    if skeleton_2d_middle is not None:
        ax2 = fig.add_subplot(2, 3, 2)
        img = images.get('middle') if images else None
        orig_size = original_sizes.get('middle') if 'original_sizes' in locals() else None
        if img is not None and orig_size is not None:
            print(f"  Middle: image size {img.shape[1]}x{img.shape[0]}, original size {orig_size[0]}x{orig_size[1]}")
            # Debug: check skeleton coordinate range
            valid_points = skeleton_2d_middle[~np.isnan(skeleton_2d_middle).any(axis=1)]
            if len(valid_points) > 0:
                print(f"  Middle skeleton coords range: x=[{valid_points[:, 0].min():.1f}, {valid_points[:, 0].max():.1f}], "
                      f"y=[{valid_points[:, 1].min():.1f}, {valid_points[:, 1].max():.1f}]")
        img_with_skeleton = draw_skeleton_2d(img, skeleton_2d_middle, original_image_size=orig_size)
        ax2.imshow(img_with_skeleton)
        ax2.set_title('Middle View (2D Skeleton)', fontsize=12, fontweight='bold')
        ax2.axis('off')
    
    if skeleton_2d_right is not None:
        ax3 = fig.add_subplot(2, 3, 3)
        img = images.get('right') if images else None
        orig_size = original_sizes.get('right') if 'original_sizes' in locals() else None
        if img is not None and orig_size is not None:
            print(f"  Right: image size {img.shape[1]}x{img.shape[0]}, original size {orig_size[0]}x{orig_size[1]}")
        img_with_skeleton = draw_skeleton_2d(img, skeleton_2d_right, original_image_size=orig_size)
        ax3.imshow(img_with_skeleton)
        ax3.set_title('Right View (2D Skeleton)', fontsize=12, fontweight='bold')
        ax3.axis('off')
    
    # Bottom row: 3D skeleton (spanning 2 columns)
    if skeleton_3d is not None:
        ax4 = fig.add_subplot(2, 3, (4, 6), projection='3d')
        visualize_3d_skeleton(skeleton_3d, ax4, "3D Triangulated Skeleton")
    
    # Add label text
    label_text = f"Frame: {frame_num}\n"
    if label:
        label_text += f"Movement: {label}"
    else:
        label_text += "Movement: (no label)"
    
    fig.suptitle(label_text, fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_multiple_samples(
    dataset_dir: Path,
    split: str = "train",
    num_samples: int = 6,
    video_paths: Optional[dict] = None,
    output_dir: Optional[str] = None
):
    """Visualize multiple training samples."""
    split_dir = dataset_dir / split
    
    # Find available frames
    skeleton_3d_dir = split_dir / "3d_skeletons"
    skeleton_files = sorted(
        skeleton_3d_dir.glob("skeleton_3d_coords_frame_*.npy"),
        key=lambda x: int(x.stem.split('_')[-1])
    )
    
    if len(skeleton_files) == 0:
        print(f"No skeleton files found in {skeleton_3d_dir}")
        return
    
    # Select evenly spaced samples
    if len(skeleton_files) <= num_samples:
        selected_files = skeleton_files
    else:
        indices = np.linspace(0, len(skeleton_files) - 1, num_samples, dtype=int)
        selected_files = [skeleton_files[i] for i in indices]
    
    print(f"Visualizing {len(selected_files)} samples from {split} set...")
    
    for i, skeleton_file in enumerate(selected_files):
        frame_num = int(skeleton_file.stem.split('_')[-1])
        
        save_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{split}_sample_{i+1}_frame_{frame_num:05d}.png")
        
        print(f"\nSample {i+1}/{len(selected_files)}: Frame {frame_num}")
        visualize_training_sample(
            dataset_dir=dataset_dir,
            split=split,
            frame_num=frame_num,
            video_paths=video_paths,
            save_path=save_path
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training samples with 2D/3D skeletons and labels"
    )
    
    parser.add_argument("--dataset-dir", type=str,
                       default="/Users/olivia/Desktop/COMS4731-final-project/dataset",
                       help="Dataset directory")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "val", "test"],
                       help="Which split to visualize")
    parser.add_argument("--frame-num", type=int, default=None,
                       help="Specific frame number to visualize")
    parser.add_argument("--num-samples", type=int, default=6,
                       help="Number of samples to visualize (if frame-num not specified)")
    parser.add_argument("--video-dir", type=str, default=None,
                       help="Directory with original videos (for showing images)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    
    # Map camera names to video files if video directory provided
    video_paths = None
    if args.video_dir:
        video_dir = Path(args.video_dir)
        video_mapping = {
            "left": "arabesque_left",
            "middle": "barre_tripod",
            "right": "barre_right"
        }
        video_paths = {}
        for cam_name, video_name in video_mapping.items():
            for ext in ['.mov', '.MOV', '.mp4', '.MP4']:
                video_file = video_dir / f"{video_name}{ext}"
                if video_file.exists():
                    video_paths[cam_name] = str(video_file)
                    break
        
        # Print alignment info
        csv_dir = dataset_dir.parent
        # Get actual FPS
        fps = 30.0
        if video_paths:
            cap_temp = cv2.VideoCapture(list(video_paths.values())[0])
            if cap_temp.isOpened():
                fps = cap_temp.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or np.isnan(fps):
                    fps = 30.0
                cap_temp.release()
        
        try:
            offsets = get_video_frame_offsets(csv_dir, reference_video="arabesque_left", fps=fps)
            print(f"\nVideo frame offsets (for alignment, FPS={fps:.2f}):")
            print("  (Dataset frames are aligned; offsets show actual video frame numbers)")
            for cam_name, offset in offsets.items():
                print(f"  {cam_name}: +{offset} frames")
        except Exception as e:
            print(f"Warning: Could not calculate offsets: {e}")
    
    if args.frame_num is not None:
        # Visualize single frame
        visualize_training_sample(
            dataset_dir=dataset_dir,
            split=args.split,
            frame_num=args.frame_num,
            video_paths=video_paths,
            save_path=None
        )
    else:
        # Visualize multiple samples
        visualize_multiple_samples(
            dataset_dir=dataset_dir,
            split=args.split,
            num_samples=args.num_samples,
            video_paths=video_paths,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()

