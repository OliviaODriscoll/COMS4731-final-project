"""
Visualize clustering of 2D skeletons by label.

For each label, this script:
- collects 2D skeletons from the specified camera view (default: middle)
- aligns each skeleton on a chosen anchor joint
- flattens aligned skeletons to feature vectors
- runs PCA to 2D
- creates a scatter plot showing clusters for all labels
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
        # Fallback: expand timestamp-based labels using simple assumptions.
        # NOTE: in your current pipeline, create_dataset.py already generated frame_labels.csv,
        # so this branch should normally NOT be used.
        raise RuntimeError(
            "frame_labels.csv not found. Please regenerate the dataset so that "
            "frame_labels.csv is created, or adapt this script to expand labels.csv."
        )

    raise FileNotFoundError(f"No frame_labels.csv or labels.csv found in {split_dir}")


def load_skeleton_for_frame(
    split_dir: Path,
    view: str,
    frame_num: int,
) -> np.ndarray:
    """
    Load a 2D skeleton numpy array for a given frame and view.
    """
    skeleton_path = (
        split_dir / "2d_skeletons" / view / f"skeleton_2d_coords_frame_{frame_num:05d}.npy"
    )
    if not skeleton_path.exists():
        raise FileNotFoundError(f"Skeleton file not found: {skeleton_path}")
    return np.load(skeleton_path)


def collect_aligned_features(
    split_dir: Path,
    view: str = "middle",
    anchor_joint: int = 23,
    max_per_label: int = 500,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Collect aligned skeleton feature vectors and their labels.

    Args:
        split_dir: directory of the split (e.g., dataset/train)
        view: which 2D camera view to use ('left', 'middle', 'right')
        anchor_joint: index of joint to align on (MediaPipe: 23 ~ left hip)
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
            # Uniform subsample for temporal coverage
            step = max(1, len(frames) // max_per_label)
            frames = frames[::step][:max_per_label]

        print(f"Collecting skeletons for label '{label}' ({len(frames)} frames)")

        for frame_num in frames:
            try:
                skel = load_skeleton_for_frame(split_dir, view=view, frame_num=frame_num)
            except FileNotFoundError:
                # Skip if this frame doesn't have the requested view
                continue

            # Expect shape (num_joints, 2) or (num_joints, 3); if 3, ignore z
            if skel.ndim != 2 or skel.shape[1] not in (2, 3):
                raise ValueError(
                    f"Unexpected skeleton shape {skel.shape} in {view} view "
                    f"(frame {frame_num}). Expected (num_joints, 2 or 3)."
                )

            if skel.shape[1] == 3:
                skel = skel[:, :2]

            num_joints = skel.shape[0]
            if anchor_joint >= num_joints:
                raise ValueError(
                    f"anchor_joint index {anchor_joint} >= num_joints {num_joints}"
                )

            # Handle NaNs: if anchor is NaN, skip this frame
            anchor = skel[anchor_joint].copy()
            if np.isnan(anchor).any():
                continue

            # Align: subtract anchor joint from all joints
            aligned = skel - anchor[None, :]

            # Optionally normalize scale: divide by torso length (shoulder-hip)
            # Here we use distance between shoulders (11,12) if available, else L2 norm of all joints.
            valid_points = aligned[~np.isnan(aligned).any(axis=1)]
            if valid_points.size > 0:
                # torso scale heuristic
                scale = np.linalg.norm(valid_points, axis=1).mean()
                if scale > 1e-6:
                    aligned = aligned / scale

            # Replace remaining NaNs (if any joints were missing) with 0
            aligned = np.nan_to_num(aligned, nan=0.0)

            feat = aligned.flatten()
            features.append(feat)
            label_indices.append(label_to_idx[label])

    if not features:
        raise ValueError("No features were collected (check that skeletons exist for the chosen view).")

    features_arr = np.stack(features, axis=0)
    label_indices_arr = np.array(label_indices, dtype=np.int64)
    return features_arr, label_indices_arr, label_names


def pca_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple PCA to 2D using SVD.

    Returns:
        X_2d: (N, 2) projected data
        mean: (D,) mean vector
        components: (2, D) principal directions
    """
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    X_centered = X - mean

    # SVD on covariance-equivalent
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:2]  # (2, D)
    X_2d = X_centered @ components.T
    return X_2d, mean, components


def plot_label_clusters(
    X_2d: np.ndarray,
    label_indices: np.ndarray,
    label_names: List[str],
    title: str,
    save_path: Path = None,
):
    """
    Scatter plot of 2D PCA projections colored by label.
    """
    plt.figure(figsize=(10, 8))

    num_labels = len(label_names)
    cmap = plt.get_cmap("tab20")

    for idx, label in enumerate(label_names):
        mask = label_indices == idx
        if not np.any(mask):
            continue
        color = cmap(idx % 20)
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=10,
            alpha=0.6,
            label=label,
            color=color,
        )

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(title)
    plt.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved clustering figure to {save_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cluster 2D skeletons by label using PCA after alignment on an anchor joint "
            "from a specific camera view."
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
        "--view",
        type=str,
        default="middle",
        choices=["left", "middle", "right"],
        help="2D camera view to use.",
    )
    parser.add_argument(
        "--anchor-joint",
        type=int,
        default=23,
        help="Joint index to align on (MediaPipe: 23 ~ left hip).",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=500,
        help="Maximum number of frames to sample per label.",
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
    print(f"View: {args.view}, anchor_joint: {args.anchor_joint}")

    features, label_indices, label_names = collect_aligned_features(
        split_dir=split_dir,
        view=args.view,
        anchor_joint=args.anchor_joint,
        max_per_label=args.max_per_label,
    )

    print(f"Collected {features.shape[0]} samples, feature dim = {features.shape[1]}")
    print(f"Labels: {label_names}")

    X_2d, mean, components = pca_2d(features)
    print("PCA done. Explained variance (approx via singular values):")
    # approximate explained variance ratio from singular values
    # note: singular values S correspond to sqrt of eigenvalues (for centered X)
    # but we didn't normalize by (n-1), so this is just qualitative
    # we won't print exact ratios to keep things simple

    title = f"2D Skeleton Clusters ({args.split} split, view={args.view})"

    if args.output_path:
        save_path = Path(args.output_path)
    else:
        save_path = None

    plot_label_clusters(
        X_2d=X_2d,
        label_indices=label_indices,
        label_names=label_names,
        title=title,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()


