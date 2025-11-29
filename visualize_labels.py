"""
Visualize ballet movement labels from the dataset.

Creates various visualizations:
1. Label distribution (bar chart)
2. Timeline of labels
3. Sample frames with labels
4. Label frequency over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import argparse
from collections import Counter


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert timestamp string (MM:SS or MM:SS.S) to seconds."""
    parts = timestamp.split(':')
    return int(parts[0]) * 60 + float(parts[1])


def seconds_to_frame(seconds: float, fps: float) -> int:
    """Convert seconds to frame number."""
    return int(seconds * fps)


def visualize_label_distribution(labels_df: pd.DataFrame, output_path: str = None):
    """Create bar chart of label distribution."""
    label_counts = labels_df['step'].value_counts()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(label_counts)), label_counts.values, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(label_counts)))
    ax.set_xticklabels(label_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Number of Occurrences', fontsize=12)
    ax.set_xlabel('Dance Step', fontsize=12)
    ax.set_title('Label Distribution (Number of Occurrences)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved label distribution to {output_path}")
    else:
        plt.show()
    plt.close()


def visualize_label_timeline(labels_df: pd.DataFrame, fps: float = 30.0, 
                             output_path: str = None, max_time: float = None):
    """Create timeline visualization showing when each label occurs."""
    # Create color map for labels
    unique_labels = sorted(labels_df['step'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    fig, ax = plt.subplots(figsize=(16, 4))
    
    y_pos = 0
    for _, row in labels_df.iterrows():
        start_sec = timestamp_to_seconds(row['start'])
        end_sec = timestamp_to_seconds(row['end'])
        duration = end_sec - start_sec
        label = row['step']
        
        # Draw rectangle for this label
        rect = mpatches.Rectangle((start_sec, y_pos - 0.4), duration, 0.8,
                                 facecolor=label_to_color[label], 
                                 edgecolor='black', linewidth=0.5, alpha=0.7)
        ax.add_patch(rect)
        
        # Add label text if duration is long enough
        if duration > 0.5:
            ax.text(start_sec + duration/2, y_pos, label, 
                   ha='center', va='center', fontsize=8, fontweight='bold')
    
    if max_time is None:
        max_time = max(timestamp_to_seconds(row['end']) for _, row in labels_df.iterrows())
    
    ax.set_xlim(0, max_time)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    ax.set_title('Label Timeline', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    
    # Add legend
    legend_elements = [mpatches.Patch(facecolor=label_to_color[label], 
                                      edgecolor='black', label=label)
                      for label in unique_labels]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), 
             fontsize=8, ncol=1)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved label timeline to {output_path}")
    else:
        plt.show()
    plt.close()


def visualize_label_frequency(labels_df: pd.DataFrame, fps: float = 30.0,
                              output_path: str = None, bin_size_sec: float = 5.0):
    """Create histogram showing label frequency over time."""
    # Create time bins
    max_time = max(timestamp_to_seconds(row['end']) for _, row in labels_df.iterrows())
    bins = np.arange(0, max_time + bin_size_sec, bin_size_sec)
    
    # Count labels in each bin
    unique_labels = sorted(labels_df['step'].unique())
    label_counts_per_bin = {label: np.zeros(len(bins) - 1) for label in unique_labels}
    
    for _, row in labels_df.iterrows():
        start_sec = timestamp_to_seconds(row['start'])
        end_sec = timestamp_to_seconds(row['end'])
        label = row['step']
        
        # Find which bins this label overlaps with
        start_bin = np.searchsorted(bins, start_sec) - 1
        end_bin = np.searchsorted(bins, end_sec)
        
        for bin_idx in range(max(0, start_bin), min(len(bins) - 1, end_bin)):
            label_counts_per_bin[label][bin_idx] += 1
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(16, 6))
    
    bottom = np.zeros(len(bins) - 1)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        counts = label_counts_per_bin[label]
        ax.bar(bins[:-1], counts, bin_size_sec, label=label, 
              bottom=bottom, color=colors[i], alpha=0.7)
        bottom += counts
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Number of Labels', fontsize=12)
    ax.set_title(f'Label Frequency Over Time (binned every {bin_size_sec}s)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved label frequency to {output_path}")
    else:
        plt.show()
    plt.close()


def visualize_frame_labels(frame_labels_df: pd.DataFrame, output_path: str = None,
                           sample_size: int = 1000):
    """Visualize frame-by-frame labels (if frame_labels.csv exists)."""
    if len(frame_labels_df) == 0:
        print("No frame labels to visualize")
        return
    
    # Sample if too many frames
    if len(frame_labels_df) > sample_size:
        step = len(frame_labels_df) // sample_size
        sampled_df = frame_labels_df.iloc[::step]
    else:
        sampled_df = frame_labels_df
    
    # Create color map
    unique_labels = sorted(frame_labels_df['label'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: colors[i] if label != 'no_label' else 'gray' 
                      for i, label in enumerate(unique_labels)}
    
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # Plot each frame as a colored point
    for label in unique_labels:
        mask = sampled_df['label'] == label
        frames = sampled_df[mask]['frame']
        ax.scatter(frames, [0] * len(frames), c=[label_to_color[label]], 
                  label=label, alpha=0.6, s=10)
    
    ax.set_xlabel('Frame Number', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    ax.set_title('Frame-by-Frame Labels', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved frame labels visualization to {output_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ballet movement labels"
    )
    
    parser.add_argument("--dataset-dir", type=str,
                       default="/Users/olivia/Desktop/COMS4731-final-project/dataset",
                       help="Dataset directory")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "val", "test", "all"],
                       help="Which split to visualize (or 'all' for all splits)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save visualizations (default: dataset_dir/label_visualizations)")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="Frames per second (for timeline visualization)")
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = dataset_dir / "label_visualizations"
    output_dir.mkdir(exist_ok=True)
    
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Visualizing {split} split")
        print(f"{'='*60}")
        
        labels_csv = dataset_dir / split / "labels.csv"
        if not labels_csv.exists():
            print(f"Warning: {labels_csv} not found, skipping {split}")
            continue
        
        labels_df = pd.read_csv(labels_csv)
        print(f"Loaded {len(labels_df)} labels from {labels_csv}")
        
        # Create visualizations
        split_output_dir = output_dir / split
        split_output_dir.mkdir(exist_ok=True)
        
        # 1. Label distribution
        print("\n1. Creating label distribution chart...")
        visualize_label_distribution(
            labels_df, 
            output_path=str(split_output_dir / "label_distribution.png")
        )
        
        # 2. Label timeline
        print("\n2. Creating label timeline...")
        visualize_label_timeline(
            labels_df, 
            fps=args.fps,
            output_path=str(split_output_dir / "label_timeline.png")
        )
        
        # 3. Label frequency over time
        print("\n3. Creating label frequency chart...")
        visualize_label_frequency(
            labels_df,
            fps=args.fps,
            output_path=str(split_output_dir / "label_frequency.png")
        )
        
        # 4. Frame-by-frame labels (if available)
        frame_labels_csv = dataset_dir / split / "frame_labels.csv"
        if frame_labels_csv.exists():
            print("\n4. Creating frame-by-frame visualization...")
            frame_labels_df = pd.read_csv(frame_labels_csv)
            visualize_frame_labels(
                frame_labels_df,
                output_path=str(split_output_dir / "frame_labels.png")
            )
        else:
            print(f"\n4. Frame labels not found at {frame_labels_csv}, skipping frame-by-frame visualization")
            print("   (Run create_frame_labels.py first to generate frame_labels.csv)")
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"{'='*60}")
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()

