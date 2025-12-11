#!/usr/bin/env python3
"""
Train HOG-based classifiers on multiple views and depth data, then ensemble and compare.

Runs:
1. HOG -> PCA -> SVM -> Markov on left view
2. HOG -> PCA -> SVM -> Markov on middle view
3. HOG -> PCA -> SVM -> Markov on right view
4. Ensemble of the three views
5. HOG on depth data (depth as additional channel) -> PCA -> SVM -> Markov
6. Comparison table of all results
"""

import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
import pickle
import os
import time
from markov_smoothing import fit_markov_from_training_data
from tqdm import tqdm


def load_frame_from_video(video_path, frame_number):
    """Load a specific frame from a video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame {frame_number} from {video_path}")
    
    return frame


def load_depth_map(depth_path):
    """Load a depth map from file"""
    if depth_path.endswith('.npy'):
        depth = np.load(depth_path)
    elif depth_path.endswith('.png'):
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None
        # Convert to float if needed
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 65535.0
        elif depth.dtype == np.uint8:
            depth = depth.astype(np.float32) / 255.0
    else:
        return None
    
    return depth


def extract_hog_features(image, orientations=9, pixels_per_cell=(16, 16), 
                         cells_per_block=(2, 2), target_size=(320, 240)):
    """Extract HOG features from an image"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize to target size
    if target_size is not None:
        gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    
    # Extract HOG features
    features = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    
    return features


def extract_hog_from_depth(rgb_image, depth_map, orientations=9, 
                           pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                           target_size=(320, 240)):
    """
    Extract HOG features using RGB + depth as additional channel.
    
    Args:
        rgb_image: RGB frame
        depth_map: Depth map (will be resized to match RGB)
        orientations: HOG orientations
        pixels_per_cell: HOG pixels per cell
        cells_per_block: HOG cells per block
        target_size: Target image size
    
    Returns:
        HOG feature vector
    """
    # Resize both to target size
    if target_size is not None:
        rgb_resized = cv2.resize(rgb_image, target_size, interpolation=cv2.INTER_AREA)
        depth_resized = cv2.resize(depth_map, target_size, interpolation=cv2.INTER_AREA)
    else:
        rgb_resized = rgb_image
        depth_resized = depth_map
    
    # Convert RGB to grayscale
    if len(rgb_resized.shape) == 3:
        gray = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = rgb_resized
    
    # Normalize depth to 0-255 range for HOG
    if depth_resized.max() > 0:
        depth_normalized = (depth_resized / depth_resized.max() * 255).astype(np.uint8)
    else:
        depth_normalized = np.zeros_like(gray, dtype=np.uint8)
    
    # Stack RGB grayscale and depth as two channels
    # HOG can work with multi-channel images
    multi_channel = np.stack([gray, depth_normalized], axis=-1)
    
    # Extract HOG features from multi-channel image
    # For multi-channel, we can either:
    # 1. Extract HOG from each channel separately and concatenate
    # 2. Use channel_axis parameter (if supported)
    try:
        # Try with channel_axis (newer skimage)
        features = hog(
            multi_channel,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True,
            channel_axis=-1
        )
    except TypeError:
        # Fallback: extract HOG from each channel and concatenate
        features_list = []
        for channel_idx in range(multi_channel.shape[-1]):
            channel_features = hog(
                multi_channel[:, :, channel_idx],
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm='L2-Hys',
                visualize=False,
                feature_vector=True
            )
            features_list.append(channel_features)
        features = np.concatenate(features_list)
    
    return features


def save_features_cache(X, y, frame_info, cache_file):
    """Save extracted features to cache file"""
    try:
        np.savez(cache_file, X=X, y=y, frame_info=frame_info)
        return True
    except Exception as e:
        print(f"  Warning: Could not save cache: {e}")
        return False


def load_features_cache(cache_file):
    """Load extracted features from cache file"""
    try:
        data = np.load(cache_file, allow_pickle=True)
        X = data['X']
        y = data['y']
        
        # Handle frame_info - it might be stored as an array or as a list
        frame_info = data['frame_info']
        if isinstance(frame_info, np.ndarray):
            # If it's a numpy array, try to convert it
            if frame_info.dtype == object:
                # Object array - convert to list
                frame_info = frame_info.tolist()
            elif hasattr(frame_info, 'item'):
                # Try item() if it's a 0-d array
                try:
                    frame_info = frame_info.item()
                except (ValueError, TypeError):
                    # If item() fails, convert to list
                    frame_info = frame_info.tolist()
            else:
                frame_info = frame_info.tolist()
        
        # Ensure frame_info is a list of tuples
        if isinstance(frame_info, list):
            # Check if elements need conversion
            frame_info = [(int(f[0]), f[1]) if isinstance(f, (list, tuple, np.ndarray)) else f 
                         for f in frame_info]
        
        if X is None or len(X) == 0:
            print(f"    Warning: Cache file is empty")
            return None, None, None
        return X, y, frame_info
    except Exception as e:
        print(f"    Error loading cache: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def load_hog_features_from_view(dataset_dir, view, split='train', 
                                 base_dir=None, sync_offsets=None,
                                 video_paths=None, target_size=(320, 240),
                                 pixels_per_cell=(16, 16), cache_dir=None,
                                 force_regenerate=False):
    """
    Load HOG features from a specific camera view.
    
    Args:
        dataset_dir: Path to dataset_midas_3d directory
        view: 'left', 'middle', or 'right'
        split: 'train', 'val', or 'test'
        base_dir: Base directory for video paths
        sync_offsets: Dictionary of sync offsets
        video_paths: Dictionary of video paths
        target_size: Target image size for HOG
        pixels_per_cell: HOG pixels per cell
    
    Returns:
        X: Feature array
        y: Labels
        frame_info: List of (frame_num, view) tuples
    """
    # Check cache first (before doing anything else)
    if cache_dir and not force_regenerate:
        cache_file = Path(cache_dir) / f"{view}_{split}_hog_features.npz"
        if cache_file.exists():
            print(f"  Loading cached features from {cache_file}...")
            X, y, frame_info = load_features_cache(str(cache_file))
            if X is not None and len(X) > 0:
                print(f"  ✓ Loaded {len(X)} cached samples (skipping extraction)")
                return X, y, frame_info
            else:
                print(f"  Warning: Cache file exists but could not be loaded, will re-extract")
        else:
            print(f"  No cache found at {cache_file}, will extract features...")
    
    view_dir = Path(dataset_dir) / view
    split_dir = view_dir / split
    frames_csv = split_dir / "frames.csv"
    
    if not frames_csv.exists():
        print(f"Warning: {frames_csv} not found")
        return None, None, None
    
    frames_df = pd.read_csv(frames_csv)
    
    # Map view names to camera names
    camera_map = {
        'left': 'arabesque_left',
        'middle': 'barre_tripod',
        'right': 'barre_right'
    }
    camera_name = camera_map.get(view, view)
    
    X_list = []
    y_list = []
    frame_info = []
    
    print(f"  Extracting HOG features from {len(frames_df)} frames...")
    
    for _, row in tqdm(frames_df.iterrows(), total=len(frames_df), desc=f"  {view}"):
        frame_num = row['frame']
        label = row['label']
        
        if label == 'unlabeled' or label == 'no_label' or pd.isna(label):
            continue
        
        # Calculate raw video frame number
        if sync_offsets and camera_name in sync_offsets:
            raw_frame_num = frame_num + sync_offsets[camera_name]
        else:
            raw_frame_num = frame_num
        
        # Load frame from video
        if video_paths and camera_name in video_paths and base_dir:
            video_path = os.path.join(base_dir, video_paths[camera_name])
            try:
                frame = load_frame_from_video(video_path, raw_frame_num)
                
                # Extract HOG features
                hog_features = extract_hog_features(
                    frame,
                    orientations=9,
                    pixels_per_cell=pixels_per_cell,
                    cells_per_block=(2, 2),
                    target_size=target_size
                )
                
                X_list.append(hog_features)
                y_list.append(label)
                frame_info.append((frame_num, view))
                
            except Exception as e:
                continue
    
    if len(X_list) == 0:
        return None, None, None
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Save to cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = Path(cache_dir) / f"{view}_{split}_hog_features.npz"
        if save_features_cache(X, y, frame_info, str(cache_file)):
            print(f"  ✓ Cached features to {cache_file}")
    
    return X, y, frame_info


def load_hog_features_from_depth(dataset_dir, view, split='train',
                                 base_dir=None, depth_dir=None,
                                 sync_offsets=None, video_paths=None,
                                 target_size=(320, 240),
                                 pixels_per_cell=(16, 16),
                                 cache_dir=None, force_regenerate=False):
    """
    Load HOG features from RGB + depth data.
    
    Args:
        dataset_dir: Path to dataset_midas_3d directory
        view: 'left', 'middle', or 'right'
        split: 'train', 'val', or 'test'
        base_dir: Base directory for video paths
        depth_dir: Directory containing depth maps
        sync_offsets: Dictionary of sync offsets
        video_paths: Dictionary of video paths
        target_size: Target image size
        pixels_per_cell: HOG pixels per cell
    
    Returns:
        X: Feature array
        y: Labels
        frame_info: List of (frame_num, view) tuples
    """
    view_dir = Path(dataset_dir) / view
    split_dir = view_dir / split
    frames_csv = split_dir / "frames.csv"
    
    if not frames_csv.exists():
        print(f"Warning: {frames_csv} not found")
        return None, None, None
    
    frames_df = pd.read_csv(frames_csv)
    
    camera_map = {
        'left': 'arabesque_left',
        'middle': 'barre_tripod',
        'right': 'barre_right'
    }
    camera_name = camera_map.get(view, view)
    
    X_list = []
    y_list = []
    frame_info = []
    
    print(f"  Extracting HOG features from RGB+Depth for {len(frames_df)} frames...")
    
    for _, row in tqdm(frames_df.iterrows(), total=len(frames_df), desc=f"  {view}+depth"):
        frame_num = row['frame']
        label = row['label']
        
        if label == 'unlabeled' or label == 'no_label' or pd.isna(label):
            continue
        
        # Calculate raw video frame number
        if sync_offsets and camera_name in sync_offsets:
            raw_frame_num = frame_num + sync_offsets[camera_name]
        else:
            raw_frame_num = frame_num
        
        # Load RGB frame
        rgb_frame = None
        if video_paths and camera_name in video_paths and base_dir:
            video_path = os.path.join(base_dir, video_paths[camera_name])
            try:
                rgb_frame = load_frame_from_video(video_path, raw_frame_num)
            except:
                continue
        
        # Load depth map
        depth_map = None
        if depth_dir:
            # Try different depth file naming conventions and directory structures
            depth_patterns = [
                # MiDaS output format: depth_frame_XXXXX.npy
                f"depth_frame_{raw_frame_num:05d}.npy",
                f"frame_{raw_frame_num:05d}.npy",
                f"depth_{raw_frame_num:05d}.npy",
                f"{raw_frame_num:05d}.npy",
                f"depth_frame_{raw_frame_num:05d}.png",
                f"frame_{raw_frame_num:05d}.png",
            ]
            
            # Try camera-specific subdirectory first
            # MiDaS stores depth in directories like: midas_depth_arabesque_left/
            camera_depth_dirs = [
                Path(depth_dir) / f"midas_depth_{camera_name}",
                Path(depth_dir) / camera_name,
                Path(depth_dir),
            ]
            
            for depth_base_dir in camera_depth_dirs:
                if not depth_base_dir.exists():
                    continue
                for pattern in depth_patterns:
                    depth_path = depth_base_dir / pattern
                    if depth_path.exists():
                        depth_map = load_depth_map(str(depth_path))
                        if depth_map is not None:
                            break
                if depth_map is not None:
                    break
        
        if rgb_frame is None or depth_map is None:
            continue
        
        # Extract HOG features from RGB + depth
        try:
            hog_features = extract_hog_from_depth(
                rgb_frame, depth_map,
                orientations=9,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=(2, 2),
                target_size=target_size
            )
            
            X_list.append(hog_features)
            y_list.append(label)
            frame_info.append((frame_num, view))
            
        except Exception as e:
            continue
    
    if len(X_list) == 0:
        return None, None, None
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Save to cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = Path(cache_dir) / f"{view}_{split}_hog_depth_features.npz"
        if save_features_cache(X, y, frame_info, str(cache_file)):
            print(f"  ✓ Cached features to {cache_file}")
    
    return X, y, frame_info


def train_hog_pca_svm_markov(X_train, y_train, X_test, y_test,
                             frame_info_train, frame_info_test,
                             n_components=500, use_markov=True):
    """
    Train HOG -> PCA -> SVM -> Markov pipeline.
    
    Returns:
        Dictionary with results
    """
    results = {}
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA
    print(f"    Applying PCA with {n_components} components...")
    if X_train_scaled.shape[1] > 50000:
        pca = IncrementalPCA(n_components=n_components, batch_size=100)
    else:
        pca = PCA(n_components=n_components)
    
    X_train_scaled = pca.fit_transform(X_train_scaled)
    X_test_scaled = pca.transform(X_test_scaled)
    
    print(f"    Reduced to {X_train_scaled.shape[1]} components")
    
    # Train SVM
    print(f"    Training SVM...")
    if X_train_scaled.shape[1] > 10000:
        clf = LinearSVC(C=1.0, max_iter=10000, random_state=42, dual=False)
    else:
        clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=False,
                 cache_size=1000, max_iter=-1)
    
    clf.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_svm = clf.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, y_pred_svm)
    
    results['svm_accuracy'] = svm_acc
    results['y_pred_svm'] = y_pred_svm
    
    # Apply Markov smoothing
    y_pred_markov = None
    markov_acc = None
    markov_model = None
    
    if use_markov:
        print(f"    Fitting Markov model...")
        try:
            markov_model = fit_markov_from_training_data(
                frame_info_train, list(y_train)
            )
            
            print(f"    Applying Markov smoothing (only to frames within 5 frames of each other)...")
            # IMPORTANT: Sort by frame number to maintain temporal order for Markov smoothing
            if frame_info_test is not None and len(frame_info_test) > 0:
                # Sort by frame number
                sort_indices = np.argsort([f[0] for f in frame_info_test])
                frame_nums_sorted = np.array([f[0] for f in frame_info_test])[sort_indices]
                y_test_sorted = np.array(y_test)[sort_indices]
                y_pred_svm_sorted = y_pred_svm[sort_indices]
                
                # Apply Markov smoothing to the entire sequence (removed 5-frame constraint)
                # The 5-frame constraint was breaking sequences and causing poor performance
                # Instead, apply to full sequence but handle large gaps gracefully
                y_pred_markov_sorted = markov_model.smooth_predictions(
                    list(y_pred_svm_sorted), method='viterbi'
                )
                y_pred_markov_sorted = np.array(y_pred_markov_sorted)
                
                # Unsort back to original order for evaluation
                unsort_indices = np.argsort(sort_indices)
                y_pred_markov = y_pred_markov_sorted[unsort_indices]
                
                markov_acc = accuracy_score(y_test, y_pred_markov)
            else:
                # No frame info - apply directly (may not work well if data is shuffled)
                print(f"    Warning: No frame_info_test available, Markov smoothing may not work correctly on shuffled data")
                y_pred_markov = markov_model.smooth_predictions(
                    list(y_pred_svm), method='viterbi'
                )
                y_pred_markov = np.array(y_pred_markov)
                markov_acc = accuracy_score(y_test, y_pred_markov)
            
            results['markov_accuracy'] = markov_acc
            results['y_pred_markov'] = y_pred_markov
            results['markov_model'] = markov_model
            
        except Exception as e:
            print(f"    Warning: Markov smoothing failed: {e}")
            results['markov_accuracy'] = None
            results['y_pred_markov'] = None
    
    results['classifier'] = clf
    results['scaler'] = scaler
    results['pca'] = pca
    results['y_pred'] = y_pred_markov if y_pred_markov is not None else y_pred_svm
    results['y_pred_svm'] = y_pred_svm  # Always keep SVM predictions
    
    return results


def align_predictions_by_frame(predictions_dict, frame_info_dict):
    """
    Align predictions from different views by frame number.
    
    Args:
        predictions_dict: {view: predictions_array}
        frame_info_dict: {view: frame_info_list}
    
    Returns:
        aligned_predictions: {view: aligned_predictions}, common_frame_nums
    """
    # Get all unique frame numbers across all views
    all_frame_nums = set()
    for view, frame_info in frame_info_dict.items():
        if frame_info:
            frame_nums = [f[0] for f in frame_info]
            all_frame_nums.update(frame_nums)
    
    common_frame_nums = sorted(all_frame_nums)
    
    # Create frame number to index mapping for each view
    frame_to_idx = {}
    for view, frame_info in frame_info_dict.items():
        if frame_info:
            frame_to_idx[view] = {f[0]: i for i, f in enumerate(frame_info)}
    
    # Align predictions
    aligned_predictions = {}
    for view, preds in predictions_dict.items():
        if view in frame_to_idx:
            aligned = []
            for frame_num in common_frame_nums:
                if frame_num in frame_to_idx[view]:
                    idx = frame_to_idx[view][frame_num]
                    if idx < len(preds):
                        aligned.append(preds[idx])
                    else:
                        aligned.append(None)  # Missing frame
                else:
                    aligned.append(None)  # Frame not in this view
            aligned_predictions[view] = aligned
        else:
            # No frame info - just truncate to minimum length
            min_len = min(len(p) for p in predictions_dict.values())
            aligned_predictions[view] = preds[:min_len]
            common_frame_nums = list(range(min_len))
    
    return aligned_predictions, common_frame_nums


def majority_vote_ensemble(predictions_list, weights=None):
    """Ensemble predictions using majority voting, handling None values (missing frames)"""
    if weights is None:
        weights = [1.0] * len(predictions_list)
    
    min_len = min(len(p) for p in predictions_list)
    predictions_list = [p[:min_len] for p in predictions_list]
    
    ensemble_pred = []
    for i in range(min_len):
        votes = {}
        vote_count = 0
        for pred_array, weight in zip(predictions_list, weights):
            pred = pred_array[i]
            if pred is not None:  # Skip None values (missing frames)
                votes[pred] = votes.get(pred, 0) + weight
                vote_count += 1
        
        if votes:
            # Use majority vote if we have at least one vote
            ensemble_pred.append(max(votes, key=votes.get))
        else:
            # No valid predictions at this position - use first available or None
            # Try to get any non-None prediction from any view
            for pred_array in predictions_list:
                if i < len(pred_array) and pred_array[i] is not None:
                    ensemble_pred.append(pred_array[i])
                    break
            else:
                ensemble_pred.append(None)  # No valid predictions anywhere
    
    return np.array(ensemble_pred)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train HOG ensemble on multiple views and depth data"
    )
    parser.add_argument("--dataset-dir", type=str, default="dataset_midas_3d",
                       help="Path to dataset_midas_3d directory")
    parser.add_argument("--split", type=str, default="train", choices=['train', 'val', 'test'],
                       help="Which split to use for training")
    parser.add_argument("--use-val-for-test", action="store_true",
                       help="Use val split for testing")
    parser.add_argument("--pca-components", type=int, default=500,
                       help="Number of PCA components")
    parser.add_argument("--use-markov", action="store_true", default=True,
                       help="Apply Markov smoothing")
    parser.add_argument("--depth-dir", type=str, default=None,
                       help="Directory containing depth maps (e.g., 'midas_depth_results')")
    parser.add_argument("--target-size", type=int, nargs=2, default=[320, 240],
                       help="Target image size for HOG")
    parser.add_argument("--pixels-per-cell", type=int, nargs=2, default=[16, 16],
                       help="HOG pixels per cell")
    parser.add_argument("--output-dir", type=str, default="hog_ensemble_comparison",
                       help="Output directory for results")
    parser.add_argument("--cache-dir", type=str, default="hog_ensemble_cache",
                       help="Directory to cache extracted features")
    parser.add_argument("--force-regenerate", action="store_true",
                       help="Force regeneration of features (ignore cache)")
    parser.add_argument("--save-models", action="store_true",
                       help="Save trained models (classifier, PCA, scaler, Markov)")
    parser.add_argument("--load-models", action="store_true",
                       help="Load saved models instead of training new ones")
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, args.dataset_dir)
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to load previous progress if it exists
    progress_file = os.path.join(output_dir, 'progress_results.pkl')
    all_results = {}
    all_predictions = {}
    all_test_labels = None
    
    if os.path.exists(progress_file) and not args.force_regenerate:
        try:
            print(f"Loading previous progress from {progress_file}...")
            with open(progress_file, 'rb') as f:
                progress_data = pickle.load(f)
                all_results = progress_data.get('all_results', {})
                all_predictions = progress_data.get('all_predictions', {})
                all_test_labels = progress_data.get('all_test_labels', None)
            print(f"  Loaded {len(all_results)} previous results")
            print(f"  Loaded predictions for: {list(all_predictions.keys())}")
        except Exception as e:
            print(f"  Warning: Could not load previous progress: {e}")
            all_results = {}
            all_predictions = {}
            all_test_labels = None
    
    # Sync offsets and video paths (from hog_classifier.py)
    SYNC_OFFSETS = {
        'arabesque_left': 833,
        'barre_right': 1071,
        'barre_tripod': 1218
    }
    
    VIDEO_PATHS = {
        'arabesque_left': 'Raw Data/arabesque_left.mov',
        'barre_right': 'Raw Data/barre_right.mov',
        'barre_tripod': 'Raw Data/barre_tripod.mov'
    }
    
    views = ['left', 'middle', 'right']
    # all_results, all_predictions, all_test_labels already initialized above from progress file
    
    print("="*70)
    print("HOG ENSEMBLE COMPARISON")
    print("="*70)
    print(f"Dataset: {dataset_dir}")
    print(f"Training on: {args.split} split")
    print(f"PCA components: {args.pca_components}")
    print(f"Markov smoothing: {args.use_markov}")
    print(f"Target size: {args.target_size}")
    print(f"Pixels per cell: {args.pixels_per_cell}")
    print("="*70)
    
    # Train on each view with HOG
    for view in views:
        print(f"\n{'='*70}")
        print(f"HOG ON {view.upper()} VIEW")
        print(f"{'='*70}")
        
        # Load HOG features
        print(f"\nLoading HOG features from {view} view...")
        cache_dir = os.path.join(base_dir, args.cache_dir)
        X_train, y_train, frame_info_train = load_hog_features_from_view(
            dataset_dir, view, split=args.split,
            base_dir=base_dir,
            sync_offsets=SYNC_OFFSETS,
            video_paths=VIDEO_PATHS,
            target_size=tuple(args.target_size),
            pixels_per_cell=tuple(args.pixels_per_cell),
            cache_dir=cache_dir,
            force_regenerate=args.force_regenerate
        )
        
        if X_train is None or len(X_train) == 0:
            print(f"  No data found for {view} view, skipping...")
            continue
        
        print(f"  Loaded {len(X_train)} training samples")
        print(f"  Feature dimension: {X_train.shape[1]}")
        
        # Get test data
        if args.use_val_for_test:
            X_test, y_test, frame_info_test = load_hog_features_from_view(
                dataset_dir, view, split='val',
                base_dir=base_dir,
                sync_offsets=SYNC_OFFSETS,
                video_paths=VIDEO_PATHS,
                target_size=tuple(args.target_size),
                pixels_per_cell=tuple(args.pixels_per_cell),
                cache_dir=cache_dir,
                force_regenerate=args.force_regenerate
            )
        else:
            # Split without shuffling to maintain temporal order for Markov model
            # Use last 20% as test set (temporal split, not random)
            n_samples = len(X_train)
            n_test = int(n_samples * 0.2)
            n_train = n_samples - n_test
            
            # Take first 80% for train, last 20% for test (maintains temporal order)
            X_test = X_train[n_train:]
            y_test = y_train[n_train:]
            X_train = X_train[:n_train]
            y_train = y_train[:n_train]
            
            # Update frame_info
            frame_info_test = frame_info_train[n_train:]
            frame_info_train = frame_info_train[:n_train]
            
            print(f"  Split: {n_train} train, {n_test} test (temporal, unshuffled)")
        
        if X_test is None or len(X_test) == 0:
            print(f"  No test data for {view} view, skipping...")
            continue
        
        # Sort test data by frame number to ensure temporal order (important for Markov model)
        # This doesn't affect training, just ensures test evaluation is on ordered data
        if frame_info_test and len(frame_info_test) > 0:
            sort_indices = np.argsort([f[0] for f in frame_info_test])
            X_test = X_test[sort_indices]
            y_test = y_test[sort_indices]
            frame_info_test = [frame_info_test[i] for i in sort_indices]
            print(f"  Test samples: {len(X_test)} (sorted by frame number for temporal order)")
        else:
            print(f"  Test samples: {len(X_test)}")
        
        if all_test_labels is None:
            all_test_labels = y_test
        
        # Try to load saved model or train new one
        model_file = os.path.join(output_dir, f'{view}_hog_model.pkl')
        
        if args.load_models and os.path.exists(model_file):
            print(f"  Loading saved model from {model_file}...")
            try:
                with open(model_file, 'rb') as f:
                    saved_data = pickle.load(f)
                
                clf = saved_data['classifier']
                scaler = saved_data['scaler']
                pca = saved_data['pca']
                markov_model = saved_data.get('markov_model')
                
                # Apply preprocessing
                X_train_scaled = scaler.transform(X_train)
                X_train_scaled = pca.transform(X_train_scaled)
                X_test_scaled = scaler.transform(X_test)
                X_test_scaled = pca.transform(X_test_scaled)
                
                # Predictions
                y_pred_svm = clf.predict(X_test_scaled)
                svm_acc = accuracy_score(y_test, y_pred_svm)
                
                view_results = {
                    'svm_accuracy': svm_acc,
                    'y_pred_svm': y_pred_svm,
                    'classifier': clf,
                    'scaler': scaler,
                    'pca': pca
                }
                
                # Apply Markov smoothing if available (only to frames within 5 frames)
                if markov_model and args.use_markov:
                    if frame_info_test is not None and len(frame_info_test) > 0:
                        # Sort by frame number for temporal order
                        sort_indices = np.argsort([f[0] for f in frame_info_test])
                        frame_nums_sorted = np.array([f[0] for f in frame_info_test])[sort_indices]
                        y_test_sorted = np.array(y_test)[sort_indices]
                        y_pred_svm_sorted = y_pred_svm[sort_indices]
                        
                        # Apply Markov smoothing to the entire sequence (removed 5-frame constraint)
                        y_pred_markov_sorted = markov_model.smooth_predictions(
                            list(y_pred_svm_sorted), method='viterbi'
                        )
                        y_pred_markov_sorted = np.array(y_pred_markov_sorted)
                        
                        # Unsort back to original order
                        unsort_indices = np.argsort(sort_indices)
                        y_pred_markov = y_pred_markov_sorted[unsort_indices]
                    else:
                        y_pred_markov = markov_model.smooth_predictions(
                            list(y_pred_svm), method='viterbi'
                        )
                        y_pred_markov = np.array(y_pred_markov)
                    
                    markov_acc = accuracy_score(y_test, y_pred_markov)
                    view_results['markov_accuracy'] = markov_acc
                    view_results['y_pred_markov'] = y_pred_markov
                    view_results['markov_model'] = markov_model
                    view_results['y_pred'] = y_pred_markov
                else:
                    view_results['y_pred'] = y_pred_svm
                
                elapsed = 0.0  # No training time
                print(f"  ✓ Model loaded successfully")
                
            except Exception as e:
                print(f"  Warning: Could not load model: {e}")
                print(f"  Training new model...")
                start_time = time.time()
                view_results = train_hog_pca_svm_markov(
                    X_train, y_train, X_test, y_test,
                    frame_info_train, frame_info_test,
                    n_components=args.pca_components,
                    use_markov=args.use_markov
                )
                elapsed = time.time() - start_time
        else:
            # Train new model
            start_time = time.time()
            view_results = train_hog_pca_svm_markov(
                X_train, y_train, X_test, y_test,
                frame_info_train, frame_info_test,
                n_components=args.pca_components,
                use_markov=args.use_markov
            )
            elapsed = time.time() - start_time
        
        # Save model if requested
        if args.save_models:
            try:
                model_data = {
                    'classifier': view_results['classifier'],
                    'scaler': view_results['scaler'],
                    'pca': view_results['pca'],
                    'markov_model': view_results.get('markov_model'),
                    'view': view,
                    'n_components': args.pca_components
                }
                with open(model_file, 'wb') as f:
                    pickle.dump(model_data, f)
                print(f"  ✓ Model saved to {model_file}")
            except Exception as e:
                print(f"  Warning: Could not save model: {e}")
        
        all_results[f'{view}_hog'] = {
            'method': 'HOG',
            'view': view,
            'svm_accuracy': view_results['svm_accuracy'],
            'markov_accuracy': view_results.get('markov_accuracy'),
            'elapsed_time': elapsed,
            'y_pred': view_results['y_pred'],
            'y_pred_svm': view_results.get('y_pred_svm'),  # Keep SVM predictions for ensemble
            'frame_info_test': frame_info_test  # Store for frame alignment
        }
        
        all_predictions[f'{view}_hog'] = view_results['y_pred']
        # Also store SVM predictions for ensemble
        if 'y_pred_svm' in view_results:
            all_predictions[f'{view}_hog_svm'] = view_results['y_pred_svm']
        
        # Save progress after each view
        try:
            progress_file = os.path.join(output_dir, 'progress_results.pkl')
            with open(progress_file, 'wb') as f:
                pickle.dump({
                    'all_results': all_results,
                    'all_predictions': all_predictions,
                    'all_test_labels': all_test_labels
                }, f)
        except Exception as e:
            print(f"  Warning: Could not save progress: {e}")
        
        print(f"\n  ✓ {view} view complete ({elapsed:.1f}s)")
        print(f"    SVM accuracy: {view_results['svm_accuracy']:.4f}")
        if view_results.get('markov_accuracy'):
            print(f"    Markov accuracy: {view_results['markov_accuracy']:.4f}")
        print(f"    Progress saved to {output_dir}/progress_results.pkl")
    
    # Create ensemble of RGB HOG views
    if len([k for k in all_predictions.keys() if k.endswith('_hog') and not k.startswith('depth')]) > 1:
        print(f"\n{'='*70}")
        print("ENSEMBLING RGB HOG VIEWS")
        print(f"{'='*70}")
        
        # Use SVM predictions for ensemble (not Markov-smoothed, which are degraded)
        # Align by frame number
        rgb_predictions_svm_dict = {v: all_predictions[f'{v}_hog_svm'] 
                                   for v in views if f'{v}_hog_svm' in all_predictions}
        rgb_frame_info_dict = {v: all_results[f'{v}_hog'].get('frame_info_test')
                              for v in views if f'{v}_hog' in all_results}
        
        # Also get Markov predictions for comparison
        rgb_predictions_markov = [all_predictions[f'{v}_hog'] for v in views 
                                 if f'{v}_hog' in all_predictions]
        
        if len(rgb_predictions_svm_dict) > 1:
            # Map view names to camera names for sync offsets
            view_to_camera = {
                'left': 'arabesque_left',
                'middle': 'barre_tripod',
                'right': 'barre_right'
            }
            
            # Find frames that exist in ALL views (intersection)
            # frame_info stores dataset frame numbers, which should already be synchronized
            # But different views might have different sets of frames due to processing issues
            common_frames = None
            frame_counts = {}
            for view in views:
                if view in rgb_frame_info_dict and rgb_frame_info_dict[view]:
                    view_frames = set([f[0] for f in rgb_frame_info_dict[view]])
                    frame_counts[view] = len(view_frames)
                    if common_frames is None:
                        common_frames = view_frames
                    else:
                        common_frames = common_frames.intersection(view_frames)
            
            # If intersection is too small, use union and handle missing frames gracefully
            if len(common_frames) < 10:  # Threshold: if less than 10 common frames
                print(f"  Warning: Only {len(common_frames)} common frames found across views")
                print(f"  Frame counts per view: {frame_counts}")
                print(f"  Using union of frames and handling missing frames with majority voting")
                
                # Use union instead - include all frames from all views
                all_frames = set()
                for view in views:
                    if view in rgb_frame_info_dict and rgb_frame_info_dict[view]:
                        all_frames.update([f[0] for f in rgb_frame_info_dict[view]])
                common_frames = sorted(list(all_frames))
            
            if common_frames and len(common_frames) > 0:
                # Sort common frames
                common_frames = sorted(list(common_frames))
                print(f"  Found {len(common_frames)} frames common to all views")
                print(f"  Frame range: {min(common_frames)} to {max(common_frames)}")
                
                # Debug: show frame counts per view
                for view in views:
                    if view in rgb_frame_info_dict and rgb_frame_info_dict[view]:
                        total_frames = len(rgb_frame_info_dict[view])
                        common_count = len([f for f in rgb_frame_info_dict[view] if f[0] in common_frames])
                        print(f"    {view}: {common_count}/{total_frames} frames in common set")
                
                # Align predictions to common frames only
                aligned_svm_lists = []
                for view in views:
                    if view in rgb_predictions_svm_dict and view in rgb_frame_info_dict:
                        preds = rgb_predictions_svm_dict[view]
                        frame_info = rgb_frame_info_dict[view]
                        # Create frame number to prediction mapping
                        frame_to_pred = {f[0]: preds[i] for i, f in enumerate(frame_info)}
                        # Get predictions for common frames only
                        # Keep None values for missing frames - we'll handle them in voting
                        aligned = [frame_to_pred.get(f) for f in common_frames]
                        if len([p for p in aligned if p is not None]) > 0:  # At least some valid predictions
                            aligned_svm_lists.append(aligned)
                
                if len(aligned_svm_lists) > 1:
                    # All should have same length now (common frames)
                    min_len = min(len(p) for p in aligned_svm_lists)
                    aligned_svm_lists = [p[:min_len] for p in aligned_svm_lists]
                    
                    # Ensemble SVM predictions
                    ensemble_pred_svm = majority_vote_ensemble(aligned_svm_lists)
                    
                    # Align test labels to match ensemble predictions
                    # Since we've aligned predictions to common frames, just truncate labels to match
                    # Use the minimum of min_len and all_test_labels length to avoid index errors
                    if all_test_labels is not None and len(all_test_labels) > 0:
                        actual_len = min(min_len, len(all_test_labels))
                        test_labels_aligned = all_test_labels[:actual_len]
                        if actual_len < min_len:
                            print(f"  Warning: Truncating ensemble predictions from {min_len} to {actual_len} to match labels")
                            ensemble_pred_svm = ensemble_pred_svm[:actual_len]
                            if ensemble_pred_markov is not None:
                                ensemble_pred_markov = ensemble_pred_markov[:actual_len]
                            min_len = actual_len
                    else:
                        test_labels_aligned = None
                        print(f"  Warning: all_test_labels is None or empty (length={len(all_test_labels) if all_test_labels is not None else 0})")
                else:
                    # Fallback to simple truncation
                    min_len = min(len(p) for p in rgb_predictions_svm_dict.values())
                    rgb_predictions_svm = [p[:min_len] for p in rgb_predictions_svm_dict.values()]
                    ensemble_pred_svm = majority_vote_ensemble(rgb_predictions_svm)
                    test_labels_aligned = all_test_labels[:min_len] if all_test_labels is not None and len(all_test_labels) >= min_len else None
            else:
                # No common frames - fallback to simple truncation
                print(f"  Warning: No common frames found, using simple truncation")
                min_len = min(len(p) for p in rgb_predictions_svm_dict.values())
                rgb_predictions_svm = [p[:min_len] for p in rgb_predictions_svm_dict.values()]
                ensemble_pred_svm = majority_vote_ensemble(rgb_predictions_svm)
                test_labels_aligned = all_test_labels[:min_len] if all_test_labels is not None and len(all_test_labels) >= min_len else None
            
            # Also ensemble Markov predictions for comparison
            ensemble_pred_markov = None
            if len(rgb_predictions_markov) > 1:
                # Align Markov predictions the same way
                if common_frames and len(common_frames) > 0:
                    aligned_markov_lists = []
                    for view in views:
                        if view in all_predictions and f'{view}_hog' in all_predictions:
                            preds = all_predictions[f'{view}_hog']
                            if view in rgb_frame_info_dict and rgb_frame_info_dict[view]:
                                frame_info = rgb_frame_info_dict[view]
                                frame_to_pred = {f[0]: preds[i] for i, f in enumerate(frame_info)}
                                aligned = [frame_to_pred.get(f) for f in common_frames[:min_len]]
                                aligned = [p for p in aligned if p is not None]
                                if len(aligned) > 0:
                                    aligned_markov_lists.append(aligned)
                    
                    if len(aligned_markov_lists) > 1:
                        aligned_markov_lists = [p[:min_len] for p in aligned_markov_lists]
                        ensemble_pred_markov = majority_vote_ensemble(aligned_markov_lists)
                else:
                    rgb_predictions_markov = [p[:min_len] for p in rgb_predictions_markov]
                    ensemble_pred_markov = majority_vote_ensemble(rgb_predictions_markov)
            
            # Compute accuracies
            ensemble_acc_svm = None
            ensemble_acc_markov = None
            
            if test_labels_aligned is not None and len(test_labels_aligned) > 0:
                if len(test_labels_aligned) == len(ensemble_pred_svm):
                    ensemble_acc_svm = accuracy_score(test_labels_aligned, ensemble_pred_svm)
                    if ensemble_pred_markov is not None and len(ensemble_pred_markov) == len(test_labels_aligned):
                        ensemble_acc_markov = accuracy_score(test_labels_aligned, ensemble_pred_markov)
                else:
                    print(f"  Warning: Label length mismatch: {len(test_labels_aligned)} labels vs {len(ensemble_pred_svm)} predictions")
                    # Try to align by truncating to minimum
                    min_align_len = min(len(test_labels_aligned), len(ensemble_pred_svm))
                    if min_align_len > 0:
                        ensemble_acc_svm = accuracy_score(test_labels_aligned[:min_align_len], ensemble_pred_svm[:min_align_len])
                        if ensemble_pred_markov is not None:
                            ensemble_acc_markov = accuracy_score(test_labels_aligned[:min_align_len], ensemble_pred_markov[:min_align_len])
            else:
                print(f"  Warning: No test labels available for ensemble evaluation (test_labels_aligned={test_labels_aligned}, all_test_labels length={len(all_test_labels) if all_test_labels is not None else 0})")
            
            # Use SVM ensemble as primary (better performance)
            all_results['ensemble_rgb_hog'] = {
                'method': 'Ensemble (RGB HOG)',
                'view': 'all',
                'svm_accuracy': ensemble_acc_svm,
                'markov_accuracy': ensemble_acc_markov,
                'elapsed_time': None,
                'y_pred': ensemble_pred_svm  # Use SVM ensemble, not Markov
            }
            
            if ensemble_acc_svm is not None:
                print(f"  Ensemble SVM accuracy: {ensemble_acc_svm:.4f} (on {min_len} samples)")
                if ensemble_acc_markov is not None:
                    print(f"  Ensemble Markov accuracy: {ensemble_acc_markov:.4f} (for comparison)")
            else:
                print(f"  Ensemble predictions generated for {min_len} samples")
    
    # Train on depth data if available
    if args.depth_dir:
        depth_dir = os.path.join(base_dir, args.depth_dir)
        if os.path.exists(depth_dir):
            print(f"\n{'='*70}")
            print("HOG ON DEPTH DATA (RGB + Depth)")
            print(f"{'='*70}")
            
            # Try each view with depth
            for view in views:
                # Skip if already completed
                if f'{view}_depth_hog' in all_results:
                    print(f"\nSkipping {view} depth view (already completed)")
                    continue
                
                print(f"\nProcessing {view} view with depth...")
                
                X_train, y_train, frame_info_train = load_hog_features_from_depth(
                    dataset_dir, view, split=args.split,
                    base_dir=base_dir,
                    depth_dir=depth_dir,
                    sync_offsets=SYNC_OFFSETS,
                    video_paths=VIDEO_PATHS,
                    target_size=tuple(args.target_size),
                    pixels_per_cell=tuple(args.pixels_per_cell),
                    cache_dir=cache_dir,
                    force_regenerate=args.force_regenerate
                )
                
                if X_train is None or len(X_train) == 0:
                    print(f"  No depth data found for {view} view, skipping...")
                    continue
                
                print(f"  Loaded {len(X_train)} training samples")
                print(f"  Feature dimension: {X_train.shape[1]}")
                
                # Get test data
                if args.use_val_for_test:
                    X_test, y_test, frame_info_test = load_hog_features_from_depth(
                        dataset_dir, view, split='val',
                        base_dir=base_dir,
                        depth_dir=depth_dir,
                        sync_offsets=SYNC_OFFSETS,
                        video_paths=VIDEO_PATHS,
                        target_size=tuple(args.target_size),
                        pixels_per_cell=tuple(args.pixels_per_cell),
                        cache_dir=cache_dir,
                        force_regenerate=args.force_regenerate
                    )
                else:
                    # Split without shuffling to maintain temporal order for Markov model
                    # Use last 20% as test set (temporal split, not random)
                    n_samples = len(X_train)
                    n_test = int(n_samples * 0.2)
                    n_train = n_samples - n_test
                    
                    # Take first 80% for train, last 20% for test (maintains temporal order)
                    X_test = X_train[n_train:]
                    y_test = y_train[n_train:]
                    X_train = X_train[:n_train]
                    y_train = y_train[:n_train]
                    
                    # Update frame_info
                    frame_info_test = frame_info_train[n_train:]
                    frame_info_train = frame_info_train[:n_train]
                    
                    print(f"  Split: {n_train} train, {n_test} test (temporal, unshuffled)")
                
                if X_test is None or len(X_test) == 0:
                    continue
                
                # Sort test data by frame number to ensure temporal order (important for Markov model)
                # This doesn't affect training, just ensures test evaluation is on ordered data
                if frame_info_test and len(frame_info_test) > 0:
                    sort_indices = np.argsort([f[0] for f in frame_info_test])
                    X_test = X_test[sort_indices]
                    y_test = y_test[sort_indices]
                    frame_info_test = [frame_info_test[i] for i in sort_indices]
                    print(f"  Test samples: {len(X_test)} (sorted by frame number for temporal order)")
                else:
                    print(f"  Test samples: {len(X_test)}")
                
                # Try to load saved model or train new one
                model_file = os.path.join(output_dir, f'{view}_depth_hog_model.pkl')
                
                if args.load_models and os.path.exists(model_file):
                    print(f"  Loading saved depth model from {model_file}...")
                    try:
                        with open(model_file, 'rb') as f:
                            saved_data = pickle.load(f)
                        
                        clf = saved_data['classifier']
                        scaler = saved_data['scaler']
                        pca = saved_data['pca']
                        markov_model = saved_data.get('markov_model')
                        
                        # Apply preprocessing
                        X_train_scaled = scaler.transform(X_train)
                        X_train_scaled = pca.transform(X_train_scaled)
                        X_test_scaled = scaler.transform(X_test)
                        X_test_scaled = pca.transform(X_test_scaled)
                        
                        # Predictions
                        y_pred_svm = clf.predict(X_test_scaled)
                        svm_acc = accuracy_score(y_test, y_pred_svm)
                        
                        view_results = {
                            'svm_accuracy': svm_acc,
                            'y_pred_svm': y_pred_svm,
                            'classifier': clf,
                            'scaler': scaler,
                            'pca': pca
                        }
                        
                        # Apply Markov smoothing if available (with temporal sorting)
                        if markov_model and args.use_markov:
                            if frame_info_test is not None and len(frame_info_test) > 0:
                                # Sort by frame number for temporal order
                                sort_indices = np.argsort([f[0] for f in frame_info_test])
                                y_test_sorted = np.array(y_test)[sort_indices]
                                y_pred_svm_sorted = y_pred_svm[sort_indices]
                                
                                y_pred_markov_sorted = markov_model.smooth_predictions(
                                    list(y_pred_svm_sorted), method='viterbi'
                                )
                                y_pred_markov_sorted = np.array(y_pred_markov_sorted)
                                
                                # Unsort back to original order
                                unsort_indices = np.argsort(sort_indices)
                                y_pred_markov = y_pred_markov_sorted[unsort_indices]
                            else:
                                y_pred_markov = markov_model.smooth_predictions(
                                    list(y_pred_svm), method='viterbi'
                                )
                                y_pred_markov = np.array(y_pred_markov)
                            
                            markov_acc = accuracy_score(y_test, y_pred_markov)
                            view_results['markov_accuracy'] = markov_acc
                            view_results['y_pred_markov'] = y_pred_markov
                            view_results['markov_model'] = markov_model
                            view_results['y_pred'] = y_pred_markov
                        else:
                            view_results['y_pred'] = y_pred_svm
                        
                        elapsed = 0.0  # No training time
                        print(f"  ✓ Depth model loaded successfully")
                        
                    except Exception as e:
                        print(f"  Warning: Could not load depth model: {e}")
                        print(f"  Training new model...")
                        start_time = time.time()
                        view_results = train_hog_pca_svm_markov(
                            X_train, y_train, X_test, y_test,
                            frame_info_train, frame_info_test,
                            n_components=args.pca_components,
                            use_markov=args.use_markov
                        )
                        elapsed = time.time() - start_time
                else:
                    # Train new model
                    start_time = time.time()
                    view_results = train_hog_pca_svm_markov(
                        X_train, y_train, X_test, y_test,
                        frame_info_train, frame_info_test,
                        n_components=args.pca_components,
                        use_markov=args.use_markov
                    )
                    elapsed = time.time() - start_time
                
                # Save model if requested
                if args.save_models:
                    try:
                        model_data = {
                            'classifier': view_results['classifier'],
                            'scaler': view_results['scaler'],
                            'pca': view_results['pca'],
                            'markov_model': view_results.get('markov_model'),
                            'view': view,
                            'n_components': args.pca_components,
                            'type': 'depth'
                        }
                        with open(model_file, 'wb') as f:
                            pickle.dump(model_data, f)
                        print(f"  ✓ Depth model saved to {model_file}")
                    except Exception as e:
                        print(f"  Warning: Could not save depth model: {e}")
                
                all_results[f'{view}_depth_hog'] = {
                    'method': 'HOG (RGB+Depth)',
                    'view': view,
                    'svm_accuracy': view_results['svm_accuracy'],
                    'markov_accuracy': view_results.get('markov_accuracy'),
                    'elapsed_time': elapsed,
                    'y_pred': view_results['y_pred'],
                    'y_pred_svm': view_results.get('y_pred_svm'),
                    'frame_info_test': frame_info_test  # Store for frame alignment
                }
                
                all_predictions[f'{view}_depth_hog'] = view_results['y_pred']
                if 'y_pred_svm' in view_results:
                    all_predictions[f'{view}_depth_hog_svm'] = view_results['y_pred_svm']
                
                # Save progress after each depth view
                try:
                    progress_file = os.path.join(output_dir, 'progress_results.pkl')
                    with open(progress_file, 'wb') as f:
                        pickle.dump({
                            'all_results': all_results,
                            'all_predictions': all_predictions,
                            'all_test_labels': all_test_labels
                        }, f)
                    print(f"    Progress saved to {progress_file}")
                except Exception as e:
                    print(f"    Warning: Could not save progress: {e}")
                
                print(f"\n  ✓ {view} view with depth complete ({elapsed:.1f}s)")
                print(f"    SVM accuracy: {view_results['svm_accuracy']:.4f}")
                if view_results.get('markov_accuracy'):
                    print(f"    Markov accuracy: {view_results['markov_accuracy']:.4f}")
            
            # Ensemble depth views - use SVM predictions
            depth_predictions_svm_dict = {v: all_predictions[f'{v}_depth_hog_svm'] 
                                          for v in views if f'{v}_depth_hog_svm' in all_predictions}
            depth_frame_info_dict = {v: all_results[f'{v}_depth_hog'].get('frame_info_test')
                                    for v in views if f'{v}_depth_hog' in all_results}
            depth_predictions_markov = [all_predictions[f'{v}_depth_hog'] for v in views
                                      if f'{v}_depth_hog' in all_predictions]
            
            if len(depth_predictions_svm_dict) > 1:
                print(f"\n{'='*70}")
                print("ENSEMBLING DEPTH HOG VIEWS")
                print(f"{'='*70}")
                
                # Find frames that exist in ALL views (intersection, not union)
                common_frames = None
                for view in views:
                    if view in depth_frame_info_dict and depth_frame_info_dict[view]:
                        view_frames = set([f[0] for f in depth_frame_info_dict[view]])
                        if common_frames is None:
                            common_frames = view_frames
                        else:
                            common_frames = common_frames.intersection(view_frames)
                
                if common_frames and len(common_frames) > 0:
                    # Sort common frames
                    common_frames = sorted(list(common_frames))
                    print(f"  Found {len(common_frames)} frames common to all depth views")
                    
                    # Align predictions to common frames only
                    aligned_svm_lists = []
                    for view in views:
                        if view in depth_predictions_svm_dict and view in depth_frame_info_dict:
                            preds = depth_predictions_svm_dict[view]
                            frame_info = depth_frame_info_dict[view]
                            # Create frame number to prediction mapping
                            frame_to_pred = {f[0]: preds[i] for i, f in enumerate(frame_info)}
                            # Get predictions for common frames only
                            aligned = [frame_to_pred.get(f) for f in common_frames]
                            # Filter out None (shouldn't happen if common_frames is correct, but just in case)
                            aligned = [p for p in aligned if p is not None]
                            if len(aligned) > 0:
                                aligned_svm_lists.append(aligned)
                    
                    if len(aligned_svm_lists) > 1:
                        # All should have same length now (common frames)
                        min_len = min(len(p) for p in aligned_svm_lists)
                        aligned_svm_lists = [p[:min_len] for p in aligned_svm_lists]
                        
                        # Ensemble SVM predictions
                        ensemble_pred_svm = majority_vote_ensemble(aligned_svm_lists)
                        
                        # Align test labels to match ensemble predictions
                        # Since we've aligned predictions to common frames, just truncate labels to match
                        # Use the minimum of min_len and all_test_labels length to avoid index errors
                        if all_test_labels is not None and len(all_test_labels) > 0:
                            actual_len = min(min_len, len(all_test_labels))
                            test_labels_aligned = all_test_labels[:actual_len]
                            if actual_len < min_len:
                                print(f"  Warning: Truncating ensemble predictions from {min_len} to {actual_len} to match labels")
                                ensemble_pred_svm = ensemble_pred_svm[:actual_len]
                                if ensemble_pred_markov is not None:
                                    ensemble_pred_markov = ensemble_pred_markov[:actual_len]
                                min_len = actual_len
                        else:
                            test_labels_aligned = None
                            print(f"  Warning: all_test_labels is None or empty (length={len(all_test_labels) if all_test_labels is not None else 0})")
                    else:
                        # Fallback to simple truncation
                        min_len = min(len(p) for p in depth_predictions_svm_dict.values())
                        depth_predictions_svm = [p[:min_len] for p in depth_predictions_svm_dict.values()]
                        ensemble_pred_svm = majority_vote_ensemble(depth_predictions_svm)
                        test_labels_aligned = all_test_labels[:min_len] if all_test_labels is not None and len(all_test_labels) >= min_len else None
                else:
                    # No common frames - fallback to simple truncation
                    print(f"  Warning: No common frames found, using simple truncation")
                    min_len = min(len(p) for p in depth_predictions_svm_dict.values())
                    depth_predictions_svm = [p[:min_len] for p in depth_predictions_svm_dict.values()]
                    ensemble_pred_svm = majority_vote_ensemble(depth_predictions_svm)
                    test_labels_aligned = all_test_labels[:min_len] if all_test_labels is not None and len(all_test_labels) >= min_len else None
                
                # Also ensemble Markov predictions for comparison
                ensemble_pred_markov = None
                if len(depth_predictions_markov) > 1:
                    # Align Markov predictions the same way
                    if common_frames and len(common_frames) > 0:
                        aligned_markov_lists = []
                        for view in views:
                            if view in all_predictions and f'{view}_depth_hog' in all_predictions:
                                preds = all_predictions[f'{view}_depth_hog']
                                if view in depth_frame_info_dict and depth_frame_info_dict[view]:
                                    frame_info = depth_frame_info_dict[view]
                                    frame_to_pred = {f[0]: preds[i] for i, f in enumerate(frame_info)}
                                    aligned = [frame_to_pred.get(f) for f in common_frames[:min_len]]
                                    aligned = [p for p in aligned if p is not None]
                                    if len(aligned) > 0:
                                        aligned_markov_lists.append(aligned)
                        
                        if len(aligned_markov_lists) > 1:
                            aligned_markov_lists = [p[:min_len] for p in aligned_markov_lists]
                            ensemble_pred_markov = majority_vote_ensemble(aligned_markov_lists)
                    else:
                        depth_predictions_markov = [p[:min_len] for p in depth_predictions_markov]
                        ensemble_pred_markov = majority_vote_ensemble(depth_predictions_markov)
                
                # Compute accuracies
                ensemble_acc_svm = None
                ensemble_acc_markov = None
                
                if test_labels_aligned is not None:
                    ensemble_acc_svm = accuracy_score(test_labels_aligned, ensemble_pred_svm)
                    if ensemble_pred_markov is not None:
                        ensemble_acc_markov = accuracy_score(test_labels_aligned, ensemble_pred_markov)
                else:
                    print("  Warning: No test labels available for ensemble evaluation")
                
                # Use SVM ensemble as primary (better performance)
                all_results['ensemble_depth_hog'] = {
                    'method': 'Ensemble (RGB+Depth HOG)',
                    'view': 'all',
                    'svm_accuracy': ensemble_acc_svm,
                    'markov_accuracy': ensemble_acc_markov,
                    'elapsed_time': None,
                    'y_pred': ensemble_pred_svm  # Use SVM ensemble, not Markov
                }
                
                if ensemble_acc_svm is not None:
                    print(f"  Ensemble SVM accuracy: {ensemble_acc_svm:.4f} (on {min_len} samples)")
                    if ensemble_acc_markov is not None:
                        print(f"  Ensemble Markov accuracy: {ensemble_acc_markov:.4f} (for comparison)")
                else:
                    print(f"  Ensemble predictions generated for {min_len} samples")
        else:
            print(f"\nWarning: Depth directory not found: {depth_dir}")
    
    # Create comparison table
    print(f"\n{'='*70}")
    print("RESULTS COMPARISON TABLE")
    print(f"{'='*70}")
    
    summary_data = []
    for key, result in all_results.items():
        row = {
            'Method': result['method'],
            'View': result['view'],
            'SVM Accuracy': f"{result['svm_accuracy']:.4f}" if result['svm_accuracy'] else 'N/A',
            'Markov Accuracy': f"{result['markov_accuracy']:.4f}" if result['markov_accuracy'] else 'N/A',
            'Time (s)': f"{result['elapsed_time']:.1f}" if result['elapsed_time'] else 'N/A'
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Save results
    summary_file = os.path.join(output_dir, 'comparison_table.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✓ Comparison table saved to: {summary_file}")
    
    # Save detailed predictions
    if all_test_labels is not None:
        # Find the minimum length to align all predictions
        min_len = min(len(all_test_labels), *[len(p) for p in all_predictions.values() if len(p) > 0])
        
        preds_df = pd.DataFrame({
            'true_label': all_test_labels[:min_len],
        })
        for key, preds in all_predictions.items():
            if len(preds) > 0:
                preds_df[key] = preds[:min_len]
        
        preds_file = os.path.join(output_dir, 'all_predictions.csv')
        preds_df.to_csv(preds_file, index=False)
        print(f"✓ Detailed predictions saved to: {preds_file} (aligned to {min_len} samples)")
    
    print(f"\n✓ All results saved to: {output_dir}/")


if __name__ == "__main__":
    main()

