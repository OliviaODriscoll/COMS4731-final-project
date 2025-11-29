"""
Visualize 3D skeletons with their corresponding labels.

Shows skeletons from the dataset with ground truth labels displayed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
        ax.text(0.5, 0.5, 0.5, "No valid points", fontsize=12, ha='center')
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
    ax.set_title(title, fontsize=12, fontweight='bold')
    
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


def visualize_skeleton_with_label(
    skeleton_3d: np.ndarray,
    label: str,
    frame_num: int,
    save_path: Optional[str] = None
):
    """Visualize a single 3D skeleton with its label."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    visualize_3d_skeleton(skeleton_3d, ax, f"Frame {frame_num}")
    
    # Add label text
    label_text = f"Label: {label}" if label else "Label: (no label)"
    fig.suptitle(label_text, fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_multiple_skeletons(
    dataset_dir: Path,
    split: str,
    num_samples: int = 12,
    output_dir: Optional[str] = None,
    fps: float = 30.0
):
    """Visualize multiple skeletons with labels in a grid."""
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
    
    # Get all skeleton files
    skeleton_dir = split_dir / "3d_skeletons"
    if not skeleton_dir.exists():
        print(f"Error: {skeleton_dir} does not exist")
        return
    
    skeleton_files = sorted(
        skeleton_dir.glob("skeleton_3d_coords_frame_*.npy"),
        key=lambda x: int(x.stem.split('_')[-1])
    )
    
    if len(skeleton_files) == 0:
        print(f"No skeleton files found in {skeleton_dir}")
        return
    
    # Sample skeletons
    if len(skeleton_files) > num_samples:
        step = len(skeleton_files) // num_samples
        sampled_files = skeleton_files[::step][:num_samples]
    else:
        sampled_files = skeleton_files[:num_samples]
    
    print(f"Visualizing {len(sampled_files)} skeletons from {split} split")
    
    # Create grid layout
    cols = 4
    rows = (len(sampled_files) + cols - 1) // cols
    
    fig = plt.figure(figsize=(20, 5 * rows))
    
    for idx, skeleton_file in enumerate(sampled_files):
        frame_num = int(skeleton_file.stem.split('_')[-1])
        skeleton_3d = np.load(skeleton_file)
        
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
        
        # Create subplot
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        
        # Visualize skeleton
        title = f"Frame {frame_num}"
        if label:
            title += f"\n{label}"
        visualize_3d_skeleton(skeleton_3d, ax, title)
    
    plt.suptitle(f"{split.upper()} Split - Sample Skeletons with Labels", 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if output_dir:
        output_path = Path(output_dir) / f"{split}_skeletons_with_labels.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D skeletons with their labels"
    )
    
    parser.add_argument("--dataset-dir", type=str,
                       default="/Users/olivia/Desktop/COMS4731-final-project/dataset",
                       help="Dataset directory")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "val", "test"],
                       help="Which split to visualize")
    parser.add_argument("--num-samples", type=int, default=12,
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
    
    visualize_multiple_skeletons(
        dataset_dir=dataset_dir,
        split=args.split,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        fps=args.fps
    )


if __name__ == "__main__":
    main()

