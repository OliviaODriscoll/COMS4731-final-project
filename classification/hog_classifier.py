"""
HOG-based Ballet Movement Classifier

Extracts Histogram of Oriented Gradients (HOG) features from frames
and uses them to classify ballet movements.
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA, IncrementalPCA
import os
from pathlib import Path
from skimage.feature import hog
from skimage import exposure
import pickle
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing


# Synchronization offsets (from dataset frame 0)
SYNC_OFFSETS = {
    'arabesque_left': 833,
    'barre_right': 1071,
    'barre_tripod': 1218
}

# Video paths
VIDEO_PATHS = {
    'arabesque_left': 'Raw Data/arabesque_left.mov',
    'barre_right': 'Raw Data/barre_right.mov',
    'barre_tripod': 'Raw Data/barre_tripod.mov'
}


def calculate_hog_feature_dimension(image_size, orientations=9, pixels_per_cell=(8, 8), 
                                    cells_per_block=(2, 2)):
    """
    Calculate the expected HOG feature dimension for given parameters.
    
    Args:
        image_size: (width, height) of the image
        orientations: Number of orientation bins
        pixels_per_cell: Size of cells for HOG computation
        cells_per_block: Number of cells per block
        
    Returns:
        Expected feature dimension
    """
    width, height = image_size
    cells_x = width // pixels_per_cell[0]
    cells_y = height // pixels_per_cell[1]
    blocks_x = cells_x - cells_per_block[0] + 1
    blocks_y = cells_y - cells_per_block[1] + 1
    feature_dim = blocks_x * blocks_y * orientations * cells_per_block[0] * cells_per_block[1]
    return feature_dim


def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), 
                         cells_per_block=(2, 2), target_size=(640, 480), 
                         expected_dim=None, validate_dim=True):
    """
    Extract HOG features from an image.
    
    Args:
        image: Input image (grayscale or color)
        orientations: Number of orientation bins
        pixels_per_cell: Size of cells for HOG computation
        cells_per_block: Number of cells per block
        target_size: Target size (width, height) to resize image to for consistent feature dimensions
        expected_dim: Expected feature dimension (for validation)
        validate_dim: Whether to validate the feature dimension
        
    Returns:
        HOG feature vector
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize to target size for consistent feature dimensions across different camera views
    if target_size is not None:
        # Ensure exact size match (important for consistent HOG dimensions)
        gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
        # Double-check the size is correct
        if gray.shape[1] != target_size[0] or gray.shape[0] != target_size[1]:
            gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_LINEAR)
    
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
    
    # Validate dimension if expected_dim is provided
    if validate_dim and expected_dim is not None:
        if len(features) != expected_dim:
            # Calculate what dimension we should have
            actual_dim = calculate_hog_feature_dimension(
                (gray.shape[1], gray.shape[0]), 
                orientations, 
                pixels_per_cell, 
                cells_per_block
            )
            if len(features) != actual_dim:
                raise ValueError(
                    f"HOG feature dimension mismatch: got {len(features)}, "
                    f"expected {expected_dim}, calculated {actual_dim}. "
                    f"Image size: {gray.shape[1]}x{gray.shape[0]}"
                )
            # If calculated dimension matches but not expected, pad/truncate
            if len(features) < expected_dim:
                features = np.pad(features, (0, expected_dim - len(features)), mode='constant')
            else:
                features = features[:expected_dim]
    
    return features


def load_frame_from_video(video_path, frame_number):
    """Load a specific frame from a video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame {frame_number} from {video_path}")
    
    return frame


def process_single_frame(args_tuple):
    """
    Process a single frame to extract HOG features from all views.
    This function is designed to be used with multiprocessing.
    
    Args:
        args_tuple: Tuple of (frame_num, base_dir, video_paths, sync_offsets, 
                             camera_views, all_frames_dict, orientations, 
                             pixels_per_cell, cells_per_block, target_size, 
                             use_all_views, expected_feature_dim_per_view)
    
    Returns:
        Tuple of (frame_num, features_dict, label, success)
    """
    (frame_num, base_dir, video_paths, sync_offsets, camera_views, 
     all_frames_dict, orientations, pixels_per_cell, cells_per_block, 
     target_size, use_all_views, expected_feature_dim_per_view) = args_tuple
    
    try:
        frame_features_per_view = {}
        frame_label = None
        
        for view_name, camera_name in camera_views.items():
            if view_name not in all_frames_dict:
                continue
            
            # Get label for this frame from dict
            if frame_num not in all_frames_dict[view_name]:
                continue
            
            # Get label (should be same across all views for same frame)
            if frame_label is None:
                frame_label = all_frames_dict[view_name][frame_num]
            
            # Calculate raw video frame number using synchronization offsets
            raw_frame_num = frame_num + sync_offsets[camera_name]
            
            # Load frame from video
            video_path = os.path.join(base_dir, video_paths[camera_name])
            if not os.path.exists(video_path):
                continue
            
            try:
                frame = load_frame_from_video(video_path, raw_frame_num)
                
                # Extract HOG features with dimension validation
                hog_features = extract_hog_features(
                    frame,
                    orientations=orientations,
                    pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block,
                    target_size=target_size,
                    expected_dim=expected_feature_dim_per_view,
                    validate_dim=True
                )
                
                # Double-check dimension
                if expected_feature_dim_per_view is not None:
                    if len(hog_features) != expected_feature_dim_per_view:
                        # Fix dimension if mismatch
                        if len(hog_features) < expected_feature_dim_per_view:
                            hog_features = np.pad(hog_features, (0, expected_feature_dim_per_view - len(hog_features)), mode='constant')
                        else:
                            hog_features = hog_features[:expected_feature_dim_per_view]
                
                frame_features_per_view[view_name] = hog_features
                
            except Exception as e:
                continue
        
        # Combine features from all views
        if use_all_views and len(frame_features_per_view) == 3:
            # Validate all views have same dimension before concatenating
            dims = [len(frame_features_per_view[v]) for v in ['left', 'middle', 'right']]
            if len(set(dims)) > 1:
                # Dimension mismatch - use the expected dimension
                if expected_feature_dim_per_view is not None:
                    for view_name in ['left', 'middle', 'right']:
                        feat = frame_features_per_view[view_name]
                        if len(feat) != expected_feature_dim_per_view:
                            if len(feat) < expected_feature_dim_per_view:
                                frame_features_per_view[view_name] = np.pad(feat, (0, expected_feature_dim_per_view - len(feat)), mode='constant')
                            else:
                                frame_features_per_view[view_name] = feat[:expected_feature_dim_per_view]
            
            # Concatenate features from all three views
            combined_features = np.concatenate([
                frame_features_per_view['left'],
                frame_features_per_view['middle'],
                frame_features_per_view['right']
            ])
            return (frame_num, {'combined': combined_features}, frame_label, True)
        elif len(frame_features_per_view) > 0:
            return (frame_num, frame_features_per_view, frame_label, True)
        else:
            return (frame_num, None, frame_label, False)
            
    except Exception as e:
        return (frame_num, None, None, False)


def extract_hog_features_from_dataset(base_dir, dataset_dir, split='train',
                                     orientations=9, pixels_per_cell=(8, 8), 
                                     cells_per_block=(2, 2), use_all_views=True,
                                     target_size=(640, 480), num_workers=None):
    """
    Extract HOG features from the remapped dataset structure.
    
    Args:
        base_dir: Base directory of the project
        dataset_dir: Path to dataset_midas_3d directory
        split: Which split to use ('train', 'val', 'test')
        orientations: HOG orientations parameter
        pixels_per_cell: HOG pixels_per_cell parameter
        cells_per_block: HOG cells_per_block parameter
        use_all_views: If True, concatenate features from all 3 views
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels
        frame_info: List of (frame_num, camera_name) tuples
    """
    # Load frames.csv from each camera view
    camera_views = {
        'left': 'arabesque_left',
        'middle': 'barre_tripod',
        'right': 'barre_right'
    }
    
    all_frames_data = {}
    all_frames_dict = {}  # Convert to dict for multiprocessing
    for view_name, camera_name in camera_views.items():
        frames_csv = os.path.join(dataset_dir, view_name, split, 'frames.csv')
        if os.path.exists(frames_csv):
            df = pd.read_csv(frames_csv)
            all_frames_data[view_name] = df
            # Convert to dict for multiprocessing (frame -> label mapping)
            all_frames_dict[view_name] = dict(zip(df['frame'], df['label']))
            print(f"Loaded {len(df)} frames from {view_name}/{split}/frames.csv")
        else:
            print(f"Warning: {frames_csv} not found, skipping {view_name}")
    
    if len(all_frames_data) == 0:
        raise ValueError(f"No frames.csv files found in {dataset_dir}")
    
    # Get unique frame numbers (dataset frame numbers)
    all_frame_numbers = set()
    for df in all_frames_data.values():
        all_frame_numbers.update(df['frame'].unique())
    frame_numbers = sorted(list(all_frame_numbers))
    
    print(f"\nProcessing {len(frame_numbers)} unique dataset frames from {split} split")
    
    # Calculate expected feature dimension per view (for validation)
    if target_size is not None:
        expected_feature_dim_per_view = calculate_hog_feature_dimension(
            target_size, orientations, pixels_per_cell, cells_per_block
        )
        print(f"Expected HOG feature dimension per view: {expected_feature_dim_per_view}")
        if use_all_views:
            expected_combined_dim = expected_feature_dim_per_view * 3
            print(f"Expected combined feature dimension (3 views): {expected_combined_dim}")
    else:
        expected_feature_dim_per_view = None
        print("Warning: target_size is None, cannot validate feature dimensions")
    
    # Extract features using multiprocessing
    features_list = []
    labels_list = []
    frame_info_list = []
    
    print(f"\nExtracting HOG features using multiprocessing...")
    
    # Prepare arguments for each frame (use dict instead of DataFrame)
    frame_args = []
    for frame_num in frame_numbers:
        frame_args.append((
            frame_num, base_dir, VIDEO_PATHS, SYNC_OFFSETS, camera_views,
            all_frames_dict, orientations, pixels_per_cell, cells_per_block,
            target_size, use_all_views, expected_feature_dim_per_view
        ))
    
    # Use multiprocessing to process frames in parallel
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), len(frame_numbers), 8)  # Limit to 8 workers
    else:
        num_workers = min(num_workers, len(frame_numbers))
    print(f"Using {num_workers} worker processes...")
    
    results_dict = {}  # Store results by frame_num to maintain order
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_frame = {
            executor.submit(process_single_frame, args): args[0] 
            for args in frame_args
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_frame), total=len(frame_args), desc="Extracting HOG"):
            frame_num = future_to_frame[future]
            try:
                result = future.result()
                frame_num_result, features_dict, label, success = result
                if success and features_dict is not None:
                    results_dict[frame_num_result] = (features_dict, label)
            except Exception as e:
                print(f"\nError processing frame {frame_num}: {e}")
                continue
    
    # Reconstruct features_list in frame order
    for frame_num in frame_numbers:
        if frame_num in results_dict:
            features_dict, label = results_dict[frame_num]
            
            if use_all_views and 'combined' in features_dict:
                features_list.append(features_dict['combined'])
                labels_list.append(label)
                frame_info_list.append((frame_num, 'combined'))
            else:
                # Use features from available views
                for view_name, features in features_dict.items():
                    features_list.append(features)
                    labels_list.append(label)
                    frame_info_list.append((frame_num, view_name))
    
    # Save features for recovery before conversion (in case of errors)
    recovery_file = os.path.join(base_dir, 'hog_features_recovery.pkl')
    try:
        import pickle
        with open(recovery_file, 'wb') as f:
            pickle.dump({
                'features_list': features_list,
                'labels_list': labels_list,
                'frame_info_list': frame_info_list
            }, f)
        print(f"\nSaved features for recovery to {recovery_file}")
    except Exception as e:
        print(f"Warning: Could not save recovery file: {e}")
    
    # Check feature dimensions and fix any mismatches
    if len(features_list) > 0:
        # Get expected feature dimension
        if use_all_views and expected_feature_dim_per_view is not None:
            expected_dim = expected_feature_dim_per_view * 3
        elif expected_feature_dim_per_view is not None:
            expected_dim = expected_feature_dim_per_view
        else:
            # Fall back to first feature's dimension
            expected_dim = len(features_list[0])
        
        # Check for dimension mismatches
        dim_mismatches = []
        for i, feat in enumerate(features_list):
            feat_dim = len(feat) if isinstance(feat, (list, np.ndarray)) else 0
            if feat_dim != expected_dim:
                dim_mismatches.append((i, feat_dim, expected_dim))
        
        if dim_mismatches:
            print(f"\n⚠️  Warning: Found {len(dim_mismatches)} features with dimension mismatches!")
            print(f"   Expected dimension: {expected_dim}")
            print(f"   First few mismatches: {dim_mismatches[:5]}")
            print("   Fixing by padding/truncating to expected dimension...")
            
            # Fix dimension mismatches
            for i, actual_dim, exp_dim in dim_mismatches:
                feat = features_list[i]
                if not isinstance(feat, np.ndarray):
                    feat = np.array(feat)
                
                if actual_dim < exp_dim:
                    # Pad with zeros
                    features_list[i] = np.pad(feat, (0, exp_dim - actual_dim), mode='constant')
                else:
                    # Truncate
                    features_list[i] = feat[:exp_dim]
            
            print(f"   ✓ Fixed all dimension mismatches")
        else:
            print(f"\n✓ All features have consistent dimension: {expected_dim}")
    
    X = np.array(features_list)
    
    # Filter out unlabeled samples
    valid_mask = np.array([label is not None and label != 'unlabeled' and label != 'no_label' 
                          for label in labels_list])
    X = X[valid_mask]
    y = np.array([labels_list[i] for i in range(len(labels_list)) if valid_mask[i]])
    frame_info_list = [frame_info_list[i] for i in range(len(frame_info_list)) if valid_mask[i]]
    
    print(f"\nExtracted features from {len(X)} labeled samples")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Classes: {np.unique(y)}")
    
    return X, y, frame_info_list


def extract_hog_features_from_videos(base_dir, frame_numbers, label_csv_path=None, 
                                     orientations=9, pixels_per_cell=(8, 8), 
                                     cells_per_block=(2, 2), use_all_views=True):
    """
    Extract HOG features from frames across all camera views.
    
    Args:
        base_dir: Base directory of the project
        frame_numbers: List of dataset frame numbers to process
        label_csv_path: Path to CSV file with labels (frame, label columns)
        orientations: HOG orientations parameter
        pixels_per_cell: HOG pixels_per_cell parameter
        cells_per_block: HOG cells_per_block parameter
        use_all_views: If True, concatenate features from all 3 views
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (if label_csv_path provided)
        frame_info: List of (frame_num, camera_name) tuples
    """
    # Load labels if provided
    labels_dict = {}
    if label_csv_path and os.path.exists(label_csv_path):
        df = pd.read_csv(label_csv_path)
        labels_dict = dict(zip(df['frame'], df['label']))
        print(f"Loaded {len(labels_dict)} labeled frames from {label_csv_path}")
    
    # Extract features
    features_list = []
    labels_list = []
    frame_info_list = []
    
    print(f"\nExtracting HOG features from {len(frame_numbers)} frames...")
    
    for frame_num in tqdm(frame_numbers):
        frame_features_per_view = {}
        
        for camera_name, offset in SYNC_OFFSETS.items():
            # Calculate raw video frame number
            raw_frame_num = frame_num + offset
            
            # Load frame
            video_path = os.path.join(base_dir, VIDEO_PATHS[camera_name])
            if not os.path.exists(video_path):
                print(f"Warning: Video not found: {video_path}, skipping...")
                continue
            
            try:
                frame = load_frame_from_video(video_path, raw_frame_num)
                
                # Extract HOG features
                hog_features = extract_hog_features(
                    frame,
                    orientations=orientations,
                    pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block,
                    target_size=target_size
                )
                
                frame_features_per_view[camera_name] = hog_features
                
            except Exception as e:
                print(f"Error processing {camera_name} frame {raw_frame_num}: {e}")
                continue
        
        # Combine features from all views
        if use_all_views and len(frame_features_per_view) == 3:
            # Concatenate features from all three views
            combined_features = np.concatenate([
                frame_features_per_view['arabesque_left'],
                frame_features_per_view['barre_right'],
                frame_features_per_view['barre_tripod']
            ])
            features_list.append(combined_features)
            
            # Get label for this frame
            if frame_num in labels_dict:
                labels_list.append(labels_dict[frame_num])
            else:
                labels_list.append(None)
            # Frame info for combined view
            frame_info_list.append((frame_num, 'combined'))
        elif len(frame_features_per_view) > 0:
            # Use features from available views
            for camera_name, features in frame_features_per_view.items():
                features_list.append(features)
                if frame_num in labels_dict:
                    labels_list.append(labels_dict[frame_num])
                else:
                    labels_list.append(None)
                frame_info_list.append((frame_num, camera_name))
    
    X = np.array(features_list)
    
    # Filter out samples without labels if labels were provided
    if label_csv_path:
        valid_mask = np.array([label is not None for label in labels_list])
        X = X[valid_mask]
        y = np.array([labels_list[i] for i in range(len(labels_list)) if valid_mask[i]])
        frame_info_list = [frame_info_list[i] for i in range(len(frame_info_list)) if valid_mask[i]]
        
        print(f"\nExtracted features from {len(X)} labeled samples")
        print(f"Feature dimension: {X.shape[1]}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"Classes: {np.unique(y)}")
        
        return X, y, frame_info_list
    else:
        return X, None, frame_info_list


def load_saved_features(file_path):
    """
    Load saved features from either .npz or .pkl format.
    
    Args:
        file_path: Path to saved features file
        
    Returns:
        X, y, frame_info or None if loading fails
    """
    if not os.path.exists(file_path):
        return None, None, None
    
    try:
        if file_path.endswith('.npz'):
            # Load from npz format (saved with --save-features)
            saved_data = np.load(file_path, allow_pickle=True)
            X = saved_data['X']
            y = saved_data['y']
            frame_info = saved_data['frame_info'].item() if saved_data['frame_info'].item() else saved_data['frame_info']
            return X, y, frame_info
        elif file_path.endswith('.pkl'):
            # Load from pickle format (recovery file)
            with open(file_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            # Handle recovery.pkl format (has lists)
            if 'features_list' in saved_data:
                features_list = saved_data['features_list']
                labels_list = saved_data['labels_list']
                frame_info_list = saved_data['frame_info_list']
                
                # Ensure features are numpy arrays
                features_list = [np.array(feat) if not isinstance(feat, np.ndarray) else feat 
                                for feat in features_list]
                
                # Check and fix feature dimensions (same logic as in extract_hog_features_from_dataset)
                if len(features_list) > 0:
                    # Get expected feature dimension from first feature
                    expected_dim = len(features_list[0])
                    
                    # Check for dimension mismatches
                    dim_mismatches = []
                    for i, feat in enumerate(features_list):
                        if len(feat) != expected_dim:
                            dim_mismatches.append((i, len(feat), expected_dim))
                    
                    if dim_mismatches:
                        print(f"  Found {len(dim_mismatches)} features with dimension mismatches")
                        print(f"  Expected dimension: {expected_dim}")
                        print(f"  Fixing by padding/truncating to expected dimension...")
                        
                        # Fix dimension mismatches
                        for i, actual_dim, exp_dim in dim_mismatches:
                            if actual_dim < exp_dim:
                                # Pad with zeros
                                features_list[i] = np.pad(features_list[i], (0, exp_dim - actual_dim), mode='constant')
                            else:
                                # Truncate
                                features_list[i] = features_list[i][:exp_dim]
                
                # Convert to numpy arrays (now all features should have same dimension)
                X = np.array(features_list)
                
                # Filter out unlabeled samples
                valid_mask = np.array([label is not None and label != 'unlabeled' and label != 'no_label' 
                                      for label in labels_list])
                X = X[valid_mask]
                y = np.array([labels_list[i] for i in range(len(labels_list)) if valid_mask[i]])
                frame_info = [frame_info_list[i] for i in range(len(frame_info_list)) if valid_mask[i]]
                
                return X, y, frame_info
            else:
                # Try as regular pickle format
                if 'X' in saved_data and 'y' in saved_data:
                    return saved_data['X'], saved_data['y'], saved_data.get('frame_info', None)
    except Exception as e:
        print(f"Error loading features from {file_path}: {e}")
        return None, None, None
    
    return None, None, None


def train_classifier(X_train, y_train, X_test, y_test, classifier_type='svm', 
                     use_pca=False, n_components=None):
    """
    Train a classifier on HOG features.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        classifier_type: 'svm' or 'rf' (Random Forest)
        use_pca: Whether to apply PCA dimensionality reduction
        n_components: Number of PCA components (None for auto)
        
    Returns:
        Trained classifier, scaler, pca (if used), and predictions
    """
    print(f"\nTraining {classifier_type.upper()} classifier...")
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    # Warn about high-dimensional features
    if X_train.shape[1] > 10000 and not use_pca:
        print(f"\n⚠️  WARNING: Very high-dimensional features ({X_train.shape[1]} dimensions)")
        print("   Consider using --use-pca to reduce dimensionality for faster training")
        print("   PCA can significantly speed up SVM training with minimal accuracy loss")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA if requested
    pca = None
    if use_pca:
        print(f"\nApplying PCA dimensionality reduction...")
        print(f"  Original dimension: {X_train_scaled.shape[1]}")
        print(f"  Training samples: {X_train_scaled.shape[0]}")
        
        if n_components is None:
            # For very high-dimensional data, use a reasonable default instead of all components
            # Computing 95% variance on 167K features would take ~9 minutes
            # Using fixed components is much faster
            if X_train_scaled.shape[1] > 50000:
                # For very high-dim data, use a fixed reasonable number (faster than 95% variance)
                # 1000 components is usually sufficient and takes ~5 minutes instead of ~9
                n_components = min(1000, X_train_scaled.shape[0] - 1)
                print(f"  Using {n_components} components (auto, optimized for high-dimensional data)")
                print(f"  Note: Computing 95% variance would take ~9 min, using {n_components} components (~5 min)")
            else:
                # For smaller dimensions, use 95% variance (fast enough)
                n_components = 0.95  # sklearn accepts float for variance ratio
                print(f"  Using PCA to explain 95% variance (auto)")
        else:
            print(f"  Using {n_components} components (specified)")
        
        # Use IncrementalPCA for very large datasets (much faster)
        # Regular PCA on 167K features can take 10+ minutes
        # IncrementalPCA uses batches and is much faster
        if X_train_scaled.shape[1] > 50000 and isinstance(n_components, int):
            print(f"  Using IncrementalPCA (faster for large datasets)...")
            print(f"  This will be much faster than regular PCA (~1-2 min vs ~10 min)")
            # Use IncrementalPCA with batches
            batch_size = min(100, X_train_scaled.shape[0])
            pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_test_scaled = pca.transform(X_test_scaled)
        else:
            print(f"  Fitting PCA (this may take a while for high-dimensional data)...")
            pca = PCA(n_components=n_components)
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_test_scaled = pca.transform(X_test_scaled)
        
        print(f"✓ Applied PCA: {X_train_scaled.shape[1]} components")
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        print(f"  Reduced from {X_train.shape[1]} → {X_train_scaled.shape[1]} dimensions")
    
    # Train classifier
    if classifier_type == 'svm':
        # Optimize SVM for high-dimensional data:
        # - Use linear kernel for faster training (or RBF with optimizations)
        # - Increase cache_size for better performance
        # - Only use probability=True if needed (adds significant overhead)
        # - For very high-dimensional data, linear kernel often works well
        use_linear_kernel = X_train_scaled.shape[1] > 10000  # Use linear for very high-dim features
        
        if use_linear_kernel:
            print("Using linear kernel (better for high-dimensional features)")
            clf = LinearSVC(C=1.0, max_iter=10000, random_state=42, dual=False)
        else:
            print("Using RBF kernel with optimizations")
            clf = SVC(
                kernel='rbf', 
                C=1.0, 
                gamma='scale', 
                probability=False,  # Set to False for faster training (can enable if needed)
                cache_size=1000,  # Increase cache size (MB) for better performance
                max_iter=-1  # No limit on iterations
            )
    elif classifier_type == 'rf':
        clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    clf.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = clf.predict(X_train_scaled)
    y_test_pred = clf.predict(X_test_scaled)
    
    # Evaluate
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\nTraining accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    
    print("\nTest set classification report:")
    print(classification_report(y_test, y_test_pred))
    
    print("\nTest set confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    
    return clf, scaler, pca, y_test_pred


def main():
    parser = argparse.ArgumentParser(description="Extract HOG features and train classifier on remapped dataset")
    parser.add_argument("--dataset-dir", type=str, default="dataset_midas_3d",
                       help="Path to dataset_midas_3d directory")
    parser.add_argument("--split", type=str, default="train", choices=['train', 'val', 'test'],
                       help="Which dataset split to use for training")
    parser.add_argument("--use-val-for-test", action="store_true",
                       help="Use val split for testing instead of splitting train")
    parser.add_argument("--classifier", type=str, default="svm", choices=['svm', 'rf'],
                       help="Classifier type")
    parser.add_argument("--use-pca", action="store_true",
                       help="Apply PCA dimensionality reduction")
    parser.add_argument("--pca-components", type=int, default=None,
                       help="Number of PCA components (None = auto)")
    parser.add_argument("--orientations", type=int, default=9,
                       help="HOG orientations parameter")
    parser.add_argument("--pixels-per-cell", type=int, nargs=2, default=[8, 8],
                       help="HOG pixels_per_cell parameter")
    parser.add_argument("--cells-per-block", type=int, nargs=2, default=[2, 2],
                       help="HOG cells_per_block parameter")
    parser.add_argument("--target-size", type=int, nargs=2, default=[640, 480],
                       help="Target image size (width, height) for resizing before HOG extraction")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set size fraction (if not using val split)")
    parser.add_argument("--output-dir", type=str, default="hog_classifier_results",
                       help="Output directory for results")
    parser.add_argument("--save-model", action="store_true",
                       help="Save trained model")
    parser.add_argument("--save-features", type=str, default=None,
                       help="Save extracted features to this file (for resuming)")
    parser.add_argument("--load-features", type=str, default=None,
                       help="Load extracted features from this file (skip extraction)")
    parser.add_argument("--force-regenerate", action="store_true",
                       help="Force regeneration of features (skip loading saved features)")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of worker processes (default: min(CPU_count, 8))")
    parser.add_argument("--use-markov", action="store_true",
                       help="Apply Markov model smoothing to predictions")
    parser.add_argument("--markov-method", type=str, default="viterbi", choices=['viterbi', 'greedy'],
                       help="Markov smoothing method: 'viterbi' (optimal) or 'greedy' (faster)")
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, args.dataset_dir)
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return
    
    print(f"Using dataset: {dataset_dir}")
    print(f"Training on {args.split} split (all frames)")
    
    # Try to load saved features (automatic detection)
    X_train = None
    y_train = None
    frame_info_train = None
    
    # If force-regenerate is set, skip loading and optionally delete old files
    if args.force_regenerate:
        print("\n⚠️  Force regeneration enabled - skipping saved features")
        recovery_file = os.path.join(base_dir, 'hog_features_recovery.pkl')
        auto_saved_file = os.path.join(base_dir, f'hog_features_{args.split}.npz')
        
        if os.path.exists(recovery_file):
            print(f"  Deleting old recovery file: {recovery_file}")
            try:
                os.remove(recovery_file)
            except Exception as e:
                print(f"  Warning: Could not delete {recovery_file}: {e}")
        
        if os.path.exists(auto_saved_file):
            print(f"  Deleting old saved features: {auto_saved_file}")
            try:
                os.remove(auto_saved_file)
            except Exception as e:
                print(f"  Warning: Could not delete {auto_saved_file}: {e}")
    
    # Priority: 1) explicit --load-features, 2) auto-saved features, 3) recovery.pkl
    feature_files_to_try = []
    if not args.force_regenerate:
        if args.load_features:
            feature_files_to_try.append(args.load_features)
        
        # Check for auto-saved features file (from previous run)
        auto_saved_file = os.path.join(base_dir, f'hog_features_{args.split}.npz')
        if os.path.exists(auto_saved_file):
            feature_files_to_try.append(auto_saved_file)
        
        # Check for recovery file
        recovery_file = os.path.join(base_dir, 'hog_features_recovery.pkl')
        if os.path.exists(recovery_file):
            feature_files_to_try.append(recovery_file)
    
    # Try loading from any available saved features
    for feature_file in feature_files_to_try:
        print(f"\nTrying to load saved features from {feature_file}...")
        X_train, y_train, frame_info_train = load_saved_features(feature_file)
        if X_train is not None and len(X_train) > 0:
            print(f"✓ Successfully loaded {len(X_train)} samples with {X_train.shape[1]} features")
            break
    
    # Extract HOG features from training split (if not loaded)
    if X_train is None or len(X_train) == 0:
        X_train, y_train, frame_info_train = extract_hog_features_from_dataset(
        base_dir,
        dataset_dir,
        split=args.split,
        orientations=args.orientations,
        pixels_per_cell=tuple(args.pixels_per_cell),
        cells_per_block=tuple(args.cells_per_block),
        use_all_views=True,
        target_size=tuple(args.target_size),
        num_workers=args.num_workers
    )
        
        # Save features for future use
        # Save to explicit path if provided, otherwise save to default location
        if args.save_features:
            save_path = args.save_features
        else:
            # Auto-save to default location for easy resuming
            save_path = os.path.join(base_dir, f'hog_features_{args.split}.npz')
        
        print(f"\nSaving extracted features to {save_path}...")
        try:
            np.savez(save_path, 
                    X=X_train, 
                    y=y_train, 
                    frame_info=frame_info_train)
            print(f"✓ Features saved! You can use --load-features {save_path} to skip extraction next time")
        except Exception as e:
            print(f"Warning: Could not save features: {e}")
    
    if X_train is None or len(X_train) == 0:
        print("Error: No features extracted from training set!")
        return
    
    # Get test set
    if args.use_val_for_test:
        print(f"\nUsing val split for testing...")
        X_test, y_test, frame_info_test = extract_hog_features_from_dataset(
            base_dir,
            dataset_dir,
            split='val',
            orientations=args.orientations,
            pixels_per_cell=tuple(args.pixels_per_cell),
            cells_per_block=tuple(args.cells_per_block),
            use_all_views=True,
            target_size=tuple(args.target_size),
            num_workers=args.num_workers
        )
        
        if X_test is None or len(X_test) == 0:
            print("Error: No features extracted from val set!")
            return
    else:
        # Split train into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=args.test_size, random_state=42, stratify=y_train
        )
        frame_info_test = None
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train classifier
    clf, scaler, pca, y_pred = train_classifier(
        X_train, y_train, X_test, y_test,
        classifier_type=args.classifier,
        use_pca=args.use_pca,
        n_components=args.pca_components
    )
    
    # Save results directory setup (needed for Markov model saving)
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply Markov model smoothing if requested
    markov_model = None
    y_pred_smoothed = None
    y_pred_original = y_pred.copy()  # Keep original SVM predictions
    
    if args.use_markov:
        print(f"\n{'='*70}")
        print("MARKOV MODEL SMOOTHING")
        print(f"{'='*70}")
        
        try:
            from markov_smoothing import fit_markov_from_training_data
            
            # Get frame_info for training data
            if frame_info_train is None:
                # Try to reconstruct from saved features or create approximate
                print("Warning: frame_info_train not available, creating approximate frame numbers...")
                frame_info_train = [(i, 'combined') for i in range(len(X_train))]
            
            # Fit Markov model on TRAINING DATA LABELS (ground truth)
            print("\nFitting Markov model on training data labels...")
            markov_model = fit_markov_from_training_data(frame_info_train, list(y_train))
            
            # Get frame_info for test data for smoothing
            if frame_info_test is not None:
                frame_info_test_sorted = sorted(frame_info_test, key=lambda x: x[0])
                test_frame_nums = [f[0] for f in frame_info_test_sorted]
            else:
                test_frame_nums = list(range(len(X_test)))
            
            # Smooth test predictions
            print(f"\nApplying Markov smoothing to test predictions (method: {args.markov_method})...")
            y_pred_smoothed = markov_model.smooth_predictions(list(y_pred), method=args.markov_method)
            y_pred_smoothed = np.array(y_pred_smoothed)
            
            # Compare accuracies
            from sklearn.metrics import accuracy_score
            svm_acc = accuracy_score(y_test, y_pred_original)
            markov_acc = accuracy_score(y_test, y_pred_smoothed)
            
            print(f"\n{'='*70}")
            print("RESULTS COMPARISON")
            print(f"{'='*70}")
            print(f"SVM accuracy (frame-level):     {svm_acc:.4f}")
            print(f"Markov-smoothed accuracy:       {markov_acc:.4f}")
            print(f"Improvement:                    {markov_acc - svm_acc:+.4f} ({((markov_acc - svm_acc) / svm_acc * 100):+.2f}%)")
            print(f"{'='*70}")
            
            # Use smoothed predictions for final results
            y_pred = y_pred_smoothed
            
            # Save Markov model
            if args.save_model:
                markov_model_path = os.path.join(output_dir, 'markov_model.pkl')
                markov_model.save(markov_model_path)
                print(f"\nMarkov model saved to: {markov_model_path}")
                
        except ImportError:
            print("Error: markov_smoothing module not found. Install or check import.")
        except Exception as e:
            print(f"Error applying Markov smoothing: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing with SVM predictions only...")
    
    # Save results (output_dir already created above)
    
    if args.save_model:
        model_file = os.path.join(output_dir, f'hog_{args.classifier}_model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump({
                'classifier': clf,
                'scaler': scaler,
                'pca': pca,
                'orientations': args.orientations,
                'pixels_per_cell': args.pixels_per_cell,
                'cells_per_block': args.cells_per_block
            }, f)
        print(f"\nModel saved to: {model_file}")
    
    # Save predictions
    if frame_info_test is not None:
        frame_info = frame_info_test
    else:
        # Create frame_info for test set (approximate)
        frame_info = [(i, 'combined') for i in range(len(X_test))]
    
    results_df = pd.DataFrame({
        'frame': [frame_info[i][0] for i in range(len(X_test))],
        'camera': [frame_info[i][1] for i in range(len(X_test))],
        'true_label': y_test,
        'predicted_label': y_pred
    })
    
    # Add smoothed predictions if Markov was used
    if y_pred_smoothed is not None:
        results_df['predicted_label_svm'] = y_pred_original  # Keep original SVM predictions
        results_df['predicted_label_markov'] = y_pred_smoothed
        results_df['predicted_label'] = y_pred_smoothed  # Use smoothed as final
    
    results_file = os.path.join(output_dir, 'hog_predictions.csv')
    results_df.to_csv(results_file, index=False)
    print(f"Predictions saved to: {results_file}")
    
    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()

