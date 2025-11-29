"""
Visualize 2D skeletons with their corresponding labels from all camera views.

Shows 2D skeletons from left, middle, and right camera views with ground truth labels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import argparse
from typing import Optional


# MediaPipe landmark connections for drawing skeleton
LANDMARK_CONNECTIONS = [
    # Face
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Upper body
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Torso
    # Lower body
    (23, 24),  # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
]


def draw_skeleton_2d(skeleton_2d: np.ndarray, image_size: tuple = (640, 480)):
    """Draw 2D skeleton on a blank image."""
    img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    
    # Scale skeleton if needed (check if coordinates are normalized)
    valid_points = skeleton_2d[~np.isnan(skeleton_2d).any(axis=1)]
    if len(valid_points) > 0:
        max_coord = valid_points.max()
        if max_coord <= 1.0:
            # Normalized coordinates, scale to image size
            skeleton_scaled = skeleton_2d.copy()
            skeleton_scaled[:, 0] *= image_size[0]
            skeleton_scaled[:, 1] *= image_size[1]
        else:
            # Pixel coordinates - check if they fit
            max_x = valid_points[:, 0].max()
            max_y = valid_points[:, 1].max()
            if max_x > image_size[0] or max_y > image_size[1]:
                # Scale down to fit
                scale = min(image_size[0] / max_x, image_size[1] / max_y)
                skeleton_scaled = skeleton_2d.copy()
                skeleton_scaled[:, 0] *= scale
                skeleton_scaled[:, 1] *= scale
            else:
                skeleton_scaled = skeleton_2d.copy()
    else:
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
                if (0 <= pt1[0] < image_size[0] and 0 <= pt1[1] < image_size[1] and
                    0 <= pt2[0] < image_size[0] and 0 <= pt2[1] < image_size[1]):
                    cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    
    # Draw joints
    for i, pt in enumerate(skeleton_scaled):
        if not np.isnan(pt).any():
            pt = (int(pt[0]), int(pt[1]))
            if 0 <= pt[0] < image_size[0] and 0 <= pt[1] < image_size[1]:
                cv2.circle(img, pt, 5, (0, 0, 255), -1)
    
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def get_label_for_frame(frame_num: int, labels_df: pd.DataFrame, fps: float = 30.0,
                       alignment_frame: int = 833) -> Optional[str]:
    """
    Get ground truth label for a frame number.
    
    Args:
        frame_num: Dataset frame number (starts at 0, aligned at original video frame 833)
        labels_df: DataFrame with labels in timestamps
        fps: Frames per second
        alignment_frame: Original video frame number where dataset frame 0 corresponds
    """
    def timestamp_to_seconds(timestamp: str) -> float:
        parts = timestamp.split(':')
        return int(parts[0]) * 60 + float(parts[1])
    
    def seconds_to_frame(sec: float) -> int:
        return int(sec * fps)
    
    # Convert dataset frame number to original video frame number
    # Dataset frame 0 = original video frame 833 (alignment point)
    original_video_frame = frame_num + alignment_frame
    
    for _, row in labels_df.iterrows():
        start_sec = timestamp_to_seconds(row['start'])
        end_sec = timestamp_to_seconds(row['end'])
        start_frame = seconds_to_frame(start_sec)
        end_frame = seconds_to_frame(end_sec)
        
        if start_frame <= original_video_frame <= end_frame:
            return row['step']
    
    return None


def visualize_2d_skeletons_with_label(
    skeleton_2d_left: Optional[np.ndarray],
    skeleton_2d_middle: Optional[np.ndarray],
    skeleton_2d_right: Optional[np.ndarray],
    label: str,
    frame_num: int,
    save_path: Optional[str] = None
):
    """Visualize 2D skeletons from all three views with label."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    views = [
        ("Left", skeleton_2d_left),
        ("Middle", skeleton_2d_middle),
        ("Right", skeleton_2d_right)
    ]
    
    for ax, (view_name, skeleton) in zip(axes, views):
        if skeleton is not None:
            img = draw_skeleton_2d(skeleton)
            ax.imshow(img)
            ax.set_title(f"{view_name} View", fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f"{view_name} View\nNo skeleton", 
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Add label text
    label_text = f"Frame {frame_num}\nLabel: {label}" if label else f"Frame {frame_num}\nLabel: (no label)"
    fig.suptitle(label_text, fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_multiple_2d_skeletons(
    dataset_dir: Path,
    split: str,
    num_samples: int = 12,
    output_dir: Optional[str] = None,
    fps: float = 30.0,
    alignment_frame: int = 833
):
    """Visualize multiple 2D skeletons with labels in a grid."""
    split_dir = dataset_dir / split
    
    # Load labels - prefer frame_labels.csv if available (has labels for every frame)
    frame_labels_csv = split_dir / "frame_labels.csv"
    labels_csv = split_dir / "labels.csv"
    
    use_frame_labels = False
    if frame_labels_csv.exists():
        print(f"Using frame_labels.csv (has labels for every frame)")
        frame_labels_df = pd.read_csv(frame_labels_csv)
        use_frame_labels = True
        labels_df = None  # Not needed when using frame_labels
    elif labels_csv.exists():
        print(f"Using labels.csv (timestamp-based, may have gaps)")
        labels_df = pd.read_csv(labels_csv)
    else:
        print(f"Warning: No label files found")
        labels_df = None
        use_frame_labels = False
    
    # Get all skeleton files from one view to determine frame numbers
    skeleton_dir_left = split_dir / "2d_skeletons" / "left"
    if not skeleton_dir_left.exists():
        print(f"Error: {skeleton_dir_left} does not exist")
        return
    
    skeleton_files = sorted(
        skeleton_dir_left.glob("skeleton_2d_coords_frame_*.npy"),
        key=lambda x: int(x.stem.split('_')[-1])
    )
    
    if len(skeleton_files) == 0:
        print(f"No skeleton files found in {skeleton_dir_left}")
        return
    
    # Sample skeletons
    if len(skeleton_files) > num_samples:
        step = len(skeleton_files) // num_samples
        sampled_files = skeleton_files[::step][:num_samples]
    else:
        sampled_files = skeleton_files[:num_samples]
    
    print(f"Visualizing {len(sampled_files)} skeletons from {split} split")
    
    # Create grid layout (3 columns for 3 views, multiple rows)
    cols = 3
    rows = len(sampled_files)
    
    fig = plt.figure(figsize=(18, 6 * rows))
    
    for idx, skeleton_file in enumerate(sampled_files):
        frame_num = int(skeleton_file.stem.split('_')[-1])
        
        # Load skeletons from all views
        skeleton_2d_left = None
        skeleton_2d_middle = None
        skeleton_2d_right = None
        
        left_file = split_dir / "2d_skeletons" / "left" / f"skeleton_2d_coords_frame_{frame_num:05d}.npy"
        middle_file = split_dir / "2d_skeletons" / "middle" / f"skeleton_2d_coords_frame_{frame_num:05d}.npy"
        right_file = split_dir / "2d_skeletons" / "right" / f"skeleton_2d_coords_frame_{frame_num:05d}.npy"
        
        if left_file.exists():
            skeleton_2d_left = np.load(left_file)
        if middle_file.exists():
            skeleton_2d_middle = np.load(middle_file)
        if right_file.exists():
            skeleton_2d_right = np.load(right_file)
        
        # Get label
        if use_frame_labels:
            # Direct lookup from frame_labels.csv
            frame_row = frame_labels_df[frame_labels_df['frame'] == frame_num]
            if len(frame_row) > 0:
                label = frame_row.iloc[0]['label']
                if label == 'no_label':
                    label = None
            else:
                label = None
        elif labels_df is not None:
            # Lookup from timestamp-based labels (accounting for alignment frame offset)
            label = get_label_for_frame(frame_num, labels_df, fps, alignment_frame=833)
        else:
            label = None
        
        # Create subplots for this sample (3 views side by side)
        views = [
            ("Left", skeleton_2d_left),
            ("Middle", skeleton_2d_middle),
            ("Right", skeleton_2d_right)
        ]
        
        for col, (view_name, skeleton) in enumerate(views):
            ax = fig.add_subplot(rows, cols, idx * cols + col + 1)
            
            if skeleton is not None:
                img = draw_skeleton_2d(skeleton)
                ax.imshow(img)
                title = f"{view_name}"
                if col == 0:  # Add frame and label info to first column
                    title += f"\nFrame {frame_num}"
                    if label:
                        title += f"\n{label}"
                ax.set_title(title, fontsize=10, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f"{view_name}\nNo skeleton", 
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            ax.axis('off')
    
    plt.suptitle(f"{split.upper()} Split - 2D Skeletons with Labels", 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if output_dir:
        output_path = Path(output_dir) / f"{split}_2d_skeletons_with_labels.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 2D skeletons with their labels from all camera views"
    )
    
    parser.add_argument("--dataset-dir", type=str,
                       default="/Users/olivia/Desktop/COMS4731-final-project/dataset",
                       help="Dataset directory")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "val", "test"],
                       help="Which split to visualize")
    parser.add_argument("--num-samples", type=int, default=8,
                       help="Number of skeleton samples to visualize")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save visualizations")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="Frames per second (for label lookup)")
    parser.add_argument("--alignment-frame", type=int, default=833,
                       help="Original video frame where dataset frame 0 corresponds")
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = None
    
    visualize_multiple_2d_skeletons(
        dataset_dir=dataset_dir,
        split=args.split,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        fps=args.fps,
        alignment_frame=args.alignment_frame
    )


if __name__ == "__main__":
    main()

