"""
Visualize clustering of 3D skeletons by label.

For each label, this script:
- collects 3D skeletons from the dataset (triangulated 3D coords)
- aligns each skeleton on a chosen anchor joint
- normalizes scale
- flattens aligned skeletons to feature vectors
- runs PCA to 2D
- creates a figure showing the clustering of samples, colored by label
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_frame_labels(split_dir: Path) -> pd.DataFrame:
    """
    Load per-frame labels for a split.

    Prefers frame_labels.csv (has a row for every frame).
    Falls back to labels.csv if needed (but that is timestamp-based).
    """
    frame_labels_csv = split_dir / "frame_labels.csv"
    labels_csv = split_dir / "labels.csv"

    if frame_labels_csv.exists():
        df = pd.read_csv(frame_labels_csv)
        if "frame" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{frame_labels_csv} must have 'frame' and 'label' columns")
        return df

    if labels_csv.exists():
        raise RuntimeError(
            "frame_labels.csv not found. Please regenerate the dataset so that "
            "frame_labels.csv is created, or adapt this script to expand labels.csv."
        )

    raise FileNotFoundError(f"No frame_labels.csv or labels.csv found in {split_dir}")


def load_skeleton3d_for_frame(split_dir: Path, frame_num: int) -> np.ndarray:
    """
    Load a 3D skeleton numpy array for a given frame.
    """
    skeleton_path = split_dir / "3d_skeletons" / f"skeleton_3d_coords_frame_{frame_num:05d}.npy"
    if not skeleton_path.exists():
        raise FileNotFoundError(f"3D skeleton file not found: {skeleton_path}")
    return np.load(skeleton_path)


def collect_aligned_features_3d(
    split_dir: Path,
    anchor_joint: int = 23,
    max_per_label: int = 500,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Collect aligned 3D skeleton feature vectors and their labels.

    Args:
        split_dir: directory of the split (e.g., dataset/train)
        anchor_joint: index of joint to align on (e.g., 23 ~ left hip)
        max_per_label: max number of frames to sample per label

    Returns:
        features: (N, D) array of flattened aligned skeletons
        label_indices: (N,) integer indices for labels
        label_names: list mapping index -> label string
    """
    frame_labels_df = load_frame_labels(split_dir)

    # Filter out unlabeled frames
    labeled_df = frame_labels_df[frame_labels_df["label"] != "no_label"].copy()
    if labeled_df.empty:
        raise ValueError(f"No labeled frames (non-'no_label') found in {split_dir}")

    # Build label mapping
    label_names = sorted(labeled_df["label"].unique())
    label_to_idx: Dict[str, int] = {lbl: i for i, lbl in enumerate(label_names)}

    # Group frames by label
    frames_by_label: Dict[str, List[int]] = {lbl: [] for lbl in label_names}
    for _, row in labeled_df.iterrows():
        frames_by_label[row["label"]].append(int(row["frame"]))

    features: List[np.ndarray] = []
    label_indices: List[int] = []

    for label, frames in frames_by_label.items():
        if not frames:
            continue

        # Subsample if too many
        if len(frames) > max_per_label:
            step = max(1, len(frames) // max_per_label)
            frames = frames[::step][:max_per_label]

        print(f"Collecting 3D skeletons for label '{label}' ({len(frames)} frames)")

        for frame_num in frames:
            try:
                skel3d = load_skeleton3d_for_frame(split_dir, frame_num=frame_num)
            except FileNotFoundError:
                continue

            # Expect shape (num_joints, 3)
            if skel3d.ndim != 2 or skel3d.shape[1] != 3:
                raise ValueError(
                    f"Unexpected 3D skeleton shape {skel3d.shape} for frame {frame_num}; "
                    f"expected (num_joints, 3)."
                )

            num_joints = skel3d.shape[0]
            if anchor_joint >= num_joints:
                raise ValueError(
                    f"anchor_joint index {anchor_joint} >= num_joints {num_joints}"
                )

            # Handle NaNs: if anchor is NaN, skip this frame
            anchor = skel3d[anchor_joint].copy()
            if np.isnan(anchor).any():
                continue

            # Align: subtract anchor joint from all joints
            aligned = skel3d - anchor[None, :]

            # Normalize scale: use average distance from anchor to all valid joints
            diffs = aligned[~np.isnan(aligned).any(axis=1)]
            if diffs.size > 0:
                dists = np.linalg.norm(diffs, axis=1)
                scale = np.mean(dists)
                if scale > 1e-6:
                    aligned = aligned / scale

            # Replace remaining NaNs with 0
            aligned = np.nan_to_num(aligned, nan=0.0)

            feat = aligned.flatten()
            features.append(feat)
            label_indices.append(label_to_idx[label])

    if not features:
        raise ValueError(
            "No features were collected for 3D skeletons "
            "(check that 3D skeleton files and labels exist)."
        )

    features_arr = np.stack(features, axis=0)
    label_indices_arr = np.array(label_indices, dtype=np.int64)
    return features_arr, label_indices_arr, label_names


def pca(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple PCA using SVD.

    Returns:
        X_proj: (N, n_components) projected data
        mean: (D,) mean vector
        components: (n_components, D) principal directions
    """
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    X_centered = X - mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components]
    X_proj = X_centered @ components.T
    return X_proj, mean, components


def plot_label_clusters(
    X_proj: np.ndarray,
    label_indices: np.ndarray,
    label_names: List[str],
    title: str,
    save_path: Path = None,
):
    """
    Scatter plot of PCA projections colored by label.
    Supports 2D (n_components=2) and 3D (n_components=3).
    """
    n_components = X_proj.shape[1]

    if n_components == 2:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
    elif n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        raise ValueError(f"plot_label_clusters only supports 2 or 3 components, got {n_components}")

    num_labels = len(label_names)
    cmap = plt.get_cmap("tab20")

    for idx, label in enumerate(label_names):
        mask = label_indices == idx
        if not np.any(mask):
            continue
        color = cmap(idx % 20)
        if n_components == 2:
            ax.scatter(
                X_proj[mask, 0],
                X_proj[mask, 1],
                s=10,
                alpha=0.6,
                label=label,
                color=color,
            )
        else:
            ax.scatter(
                X_proj[mask, 0],
                X_proj[mask, 1],
                X_proj[mask, 2],
                s=10,
                alpha=0.6,
                label=label,
                color=color,
            )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    if n_components == 3:
        ax.set_zlabel("PC 3")
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved 3D clustering figure to {save_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cluster 3D skeletons by label using PCA after alignment on an anchor joint."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/Users/olivia/Desktop/COMS4731-final-project/dataset",
        help="Base dataset directory (with train/val/test subdirs).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--anchor-joint",
        type=int,
        default=23,
        help="Joint index to align on (e.g., 23 ~ left hip).",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=500,
        help="Maximum number of frames to sample per label.",
    )
    parser.add_argument(
        "--num-pcs",
        type=int,
        default=2,
        help="Number of principal components to project onto (2 or 3).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the clustering figure (PNG). "
        "If not provided, the plot is shown interactively.",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    split_dir = dataset_dir / args.split

    print(f"Using split directory: {split_dir}")
    print(f"Anchor joint: {args.anchor_joint}, num_pcs: {args.num_pcs}")

    features, label_indices, label_names = collect_aligned_features_3d(
        split_dir=split_dir,
        anchor_joint=args.anchor_joint,
        max_per_label=args.max_per_label,
    )

    print(f"Collected {features.shape[0]} 3D samples, feature dim = {features.shape[1]}")
    print(f"Labels: {label_names}")

    X_proj, _, _ = pca(features, n_components=args.num_pcs)

    title = f"3D Skeleton Clusters ({args.split} split, PCs={args.num_pcs})"

    save_path = Path(args.output_path) if args.output_path else None

    plot_label_clusters(
        X_proj=X_proj,
        label_indices=label_indices,
        label_names=label_names,
        title=title,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()


