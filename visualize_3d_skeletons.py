"""
Visualize 3D skeletons from saved numpy arrays.

Loads and displays 3D skeleton coordinates with interactive 3D plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path
import os


# MediaPipe Pose landmark connections (33 landmarks)
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

# Key landmark names for labeling
KEY_LANDMARKS = {
    0: 'Nose',
    11: 'L.Shoulder',
    12: 'R.Shoulder',
    15: 'L.Wrist',
    16: 'R.Wrist',
    23: 'L.Hip',
    24: 'R.Hip',
    27: 'L.Ankle',
    28: 'R.Ankle'
}


def visualize_skeleton_3d(skeleton_3d, title="3D Skeleton", show_labels=True, ax=None):
    """
    Visualize a single 3D skeleton.
    
    Args:
        skeleton_3d: NumPy array of shape (33, 3) with 3D coordinates
        title: Plot title
        show_labels: Whether to show landmark labels
        ax: Optional axes to plot on (if None, creates new figure)
    
    Returns:
        fig, ax: Figure and axes objects
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False
    
    # Extract valid points
    valid_indices = []
    valid_points = []
    
    for i in range(len(skeleton_3d)):
        pt = skeleton_3d[i]
        if not np.isnan(pt).any():
            valid_indices.append(i)
            valid_points.append(pt)
    
    if len(valid_points) == 0:
        print("Warning: No valid points in skeleton")
        return fig, ax
    
    valid_points = np.array(valid_points)
    
    # Plot points
    ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], 
               c='red', s=50, alpha=0.8, label='Joints')
    
    # Draw connections
    for start_idx, end_idx in LANDMARK_CONNECTIONS:
        if (start_idx < len(skeleton_3d) and end_idx < len(skeleton_3d)):
            pt1 = skeleton_3d[start_idx]
            pt2 = skeleton_3d[end_idx]
            
            # Check if both points are valid
            if not np.isnan(pt1).any() and not np.isnan(pt2).any():
                x_line = [pt1[0], pt2[0]]
                y_line = [pt1[1], pt2[1]]
                z_line = [pt1[2], pt2[2]]
                ax.plot(x_line, y_line, z_line, 'b-', linewidth=2, alpha=0.6)
    
    # Label key points
    if show_labels:
        for idx, label in KEY_LANDMARKS.items():
            if idx < len(skeleton_3d):
                pt = skeleton_3d[idx]
                if not np.isnan(pt).any():
                    ax.text(pt[0], pt[1], pt[2], label, fontsize=8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio for better visualization
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
    
    if created_fig:
        plt.tight_layout()
    
    return fig, ax


def visualize_multiple_skeletons(skeleton_dir, num_skeletons=5, frame_indices=None, save_path=None):
    """
    Visualize multiple skeletons from a directory.
    
    Args:
        skeleton_dir: Directory containing skeleton_3d_coords_frame_*.npy files
        num_skeletons: Number of skeletons to visualize (if frame_indices not provided)
        frame_indices: Specific frame indices to visualize (overrides num_skeletons)
        save_path: Optional path to save the visualization
    """
    skeleton_dir = Path(skeleton_dir)
    
    # Find all skeleton files
    skeleton_files = sorted(
        skeleton_dir.glob("skeleton_3d_coords_frame_*.npy"),
        key=lambda x: int(x.stem.split('_')[-1])
    )
    
    if len(skeleton_files) == 0:
        print(f"No skeleton files found in {skeleton_dir}")
        return
    
    print(f"Found {len(skeleton_files)} skeleton files")
    
    # Select frames to visualize
    if frame_indices is None:
        # Select evenly spaced frames
        if len(skeleton_files) <= num_skeletons:
            selected_files = skeleton_files
        else:
            indices = np.linspace(0, len(skeleton_files) - 1, num_skeletons, dtype=int)
            selected_files = [skeleton_files[i] for i in indices]
    else:
        # Use specified frame indices
        selected_files = []
        for idx in frame_indices:
            # Find file with matching frame number
            for f in skeleton_files:
                frame_num = int(f.stem.split('_')[-1])
                if frame_num == idx:
                    selected_files.append(f)
                    break
    
    print(f"Visualizing {len(selected_files)} skeletons")
    
    # Create subplots
    n = len(selected_files)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    
    for i, skeleton_file in enumerate(selected_files):
        frame_num = int(skeleton_file.stem.split('_')[-1])
        skeleton = np.load(skeleton_file)
        
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        visualize_skeleton_3d(
            skeleton, 
            title=f"Frame {frame_num}",
            show_labels=(i == 0),  # Only show labels on first plot
            ax=ax
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_skeleton_sequence(skeleton_dir, start_frame=0, num_frames=10, save_path=None):
    """
    Visualize a sequence of skeletons as an animation (static frames).
    
    Args:
        skeleton_dir: Directory containing skeleton files
        start_frame: Starting frame number
        num_frames: Number of frames to visualize
        save_path: Optional path to save the visualization
    """
    skeleton_dir = Path(skeleton_dir)
    
    # Load skeletons
    skeletons = []
    frame_numbers = []
    
    for i in range(num_frames):
        frame_num = start_frame + i * 10  # Assuming 10-frame interval
        skeleton_file = skeleton_dir / f"skeleton_3d_coords_frame_{frame_num:05d}.npy"
        
        if skeleton_file.exists():
            skeleton = np.load(skeleton_file)
            skeletons.append(skeleton)
            frame_numbers.append(frame_num)
    
    if len(skeletons) == 0:
        print(f"No skeletons found starting from frame {start_frame}")
        return
    
    print(f"Visualizing sequence: {len(skeletons)} frames")
    
    # Create subplots
    cols = min(5, len(skeletons))
    rows = (len(skeletons) + cols - 1) // cols
    
    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    
    for i, (skeleton, frame_num) in enumerate(zip(skeletons, frame_numbers)):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        visualize_skeleton_3d(
            skeleton,
            title=f"Frame {frame_num}",
            show_labels=(i == 0),
            ax=ax
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved sequence visualization to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D skeletons from numpy arrays")
    
    parser.add_argument("--skeleton-dir", type=str, required=True,
                       help="Directory containing skeleton_3d_coords_frame_*.npy files")
    parser.add_argument("--num-skeletons", type=int, default=5,
                       help="Number of skeletons to visualize (default: 5)")
    parser.add_argument("--frame-indices", type=int, nargs='+', default=None,
                       help="Specific frame indices to visualize (e.g., 0 10 20)")
    parser.add_argument("--sequence", action="store_true",
                       help="Visualize as a sequence (consecutive frames)")
    parser.add_argument("--start-frame", type=int, default=0,
                       help="Starting frame for sequence visualization")
    parser.add_argument("--num-frames", type=int, default=10,
                       help="Number of frames in sequence")
    parser.add_argument("--save", type=str, default=None,
                       help="Path to save visualization (optional)")
    
    args = parser.parse_args()
    
    if args.sequence:
        visualize_skeleton_sequence(
            args.skeleton_dir,
            start_frame=args.start_frame,
            num_frames=args.num_frames,
            save_path=args.save
        )
    else:
        visualize_multiple_skeletons(
            args.skeleton_dir,
            num_skeletons=args.num_skeletons,
            frame_indices=args.frame_indices,
            save_path=args.save
        )


if __name__ == "__main__":
    main()

