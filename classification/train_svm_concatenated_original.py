#!/usr/bin/env python3
"""
Train SVM using concatenated multi-view features (early fusion).
Uses ORIGINAL temporal split (not resplit).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import time


def load_skeleton(skeleton_path):
    """Load a single skeleton and flatten to 1D."""
    skeleton = np.load(skeleton_path)
    features = skeleton.flatten()
    features = np.nan_to_num(features, nan=0.0)
    return features


def load_concatenated_skeletons_original(base_dir, split, skeleton_type='2d', cameras=['left', 'right', 'middle']):
    """
    Load skeletons from all cameras and concatenate them into single feature vectors.
    Uses ORIGINAL split directories (not resplit).
    
    Args:
        base_dir: Base dataset directory
        split: 'train', 'val', or 'test'
        skeleton_type: '2d' or '3d'
        cameras: List of camera names to concatenate
    
    Returns:
        X: Feature array (N x concatenated_features)
        y: Labels (N,)
        frame_nums: Frame numbers (N,)
    """
    # Load metadata from each camera
    camera_data = {}
    
    for camera in cameras:
        camera_dir = Path(base_dir) / camera
        split_dir = camera_dir / split
        frames_csv = split_dir / "frames.csv"
        
        if not frames_csv.exists():
            print(f"Warning: {frames_csv} not found for {camera}")
            continue
        
        frames_df = pd.read_csv(frames_csv)
        
        # Remove unlabeled
        frames_df = frames_df[frames_df['label'] != 'unlabeled']
        
        # Create dict keyed by frame number for easy lookup
        frame_dict = {}
        for _, row in frames_df.iterrows():
            # Path is relative to split_dir
            skeleton_path = split_dir / row[f'skeleton_{skeleton_type}']
            frame_dict[row['frame']] = {
                'skeleton_path': skeleton_path,
                'label': row['label'],
                'time': row['time_seconds']
            }
        
        camera_data[camera] = frame_dict
        print(f"  {camera}: {len(frame_dict)} labeled frames")
    
    # Find common frames across all cameras
    if len(camera_data) == 0:
        return None, None, None
    
    # Get intersection of frame numbers
    common_frames = set(camera_data[cameras[0]].keys())
    for camera in cameras[1:]:
        if camera in camera_data:
            common_frames = common_frames.intersection(set(camera_data[camera].keys()))
    
    common_frames = sorted(list(common_frames))
    
    print(f"\n{split.upper()} set:")
    print(f"  Common frames across all cameras: {len(common_frames)}")
    
    # Build concatenated feature vectors
    X_list = []
    y_list = []
    frame_nums = []
    
    for frame_num in common_frames:
        # Concatenate features from all cameras
        concat_features = []
        label = None
        
        for camera in cameras:
            if camera not in camera_data:
                continue
            
            skeleton_path = camera_data[camera][frame_num]['skeleton_path']
            
            if not skeleton_path.exists():
                break  # Skip this frame if any camera is missing
            
            features = load_skeleton(skeleton_path)
            concat_features.append(features)
            
            if label is None:
                label = camera_data[camera][frame_num]['label']
        
        # Only add if we got features from all cameras
        if len(concat_features) == len(cameras) and label is not None:
            X_list.append(np.concatenate(concat_features))
            y_list.append(label)
            frame_nums.append(frame_num)
    
    if len(X_list) == 0:
        return None, None, None
    
    X = np.array(X_list)
    y = np.array(y_list)
    frame_nums = np.array(frame_nums)
    
    print(f"  Valid samples (all cameras): {len(X)}")
    print(f"  Feature dimension: {X.shape[1]}")
    
    # Show label distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"  Label distribution:")
    for label, count in sorted(zip(unique_labels, counts), key=lambda x: -x[1])[:5]:
        print(f"    {label}: {count}")
    
    return X, y, frame_nums


def temporal_smoothing(predictions, window_size=20):
    """Apply temporal smoothing using majority voting."""
    if len(predictions) < window_size:
        return predictions
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(predictions)):
        start = max(0, i - half_window)
        end = min(len(predictions), i + half_window + 1)
        window = predictions[start:end]
        
        if len(window) > 0:
            counter = Counter(window)
            smoothed_pred = counter.most_common(1)[0][0]
        else:
            smoothed_pred = predictions[i]
        
        smoothed.append(smoothed_pred)
    
    return np.array(smoothed)


def train_and_evaluate_concatenated_svm(X_train, y_train, X_val, y_val, X_test, y_test,
                                        frames_test, skeleton_type='2D'):
    """Train SVM on concatenated multi-view features."""
    print(f"\n{'='*70}")
    print(f"TRAINING CONCATENATED {skeleton_type} SVM")
    print(f"{'='*70}")
    
    # Standardize
    print("Standardizing concatenated features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    print(f"Training SVM (RBF kernel)...")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {X_train.shape[1]} (concatenated from 3 cameras)")
    
    start_time = time.time()
    
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    train_time = time.time() - start_time
    print(f"  Training time: {train_time:.2f}s")
    
    # Evaluate on validation
    print("\nValidation set:")
    y_val_pred = svm.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"  Accuracy: {val_acc:.4f}")
    
    # Evaluate on test (no smoothing)
    print("\nTest set (no temporal smoothing):")
    y_test_pred = svm.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"  Accuracy: {test_acc:.4f}")
    
    # Apply temporal smoothing
    print(f"\nApplying 20-frame temporal smoothing...")
    
    sort_idx = np.argsort(frames_test)
    y_test_sorted = y_test[sort_idx]
    y_test_pred_sorted = y_test_pred[sort_idx]
    
    y_test_smoothed = temporal_smoothing(y_test_pred_sorted, window_size=20)
    
    test_acc_smoothed = accuracy_score(y_test_sorted, y_test_smoothed)
    
    print(f"\nTest set (WITH 20-frame temporal smoothing):")
    print(f"  Accuracy: {test_acc_smoothed:.4f}")
    print(f"  Improvement: {test_acc_smoothed - test_acc:.4f}")
    
    # Classification report
    print(f"\n{'='*70}")
    print(f"CLASSIFICATION REPORT ({skeleton_type})")
    print(f"{'='*70}")
    print(classification_report(y_test_sorted, y_test_smoothed, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_sorted, y_test_smoothed)
    labels = sorted(list(set(y_test_sorted)))
    
    return {
        'model': svm,
        'scaler': scaler,
        'val_accuracy': val_acc,
        'test_accuracy': test_acc,
        'test_accuracy_smoothed': test_acc_smoothed,
        'confusion_matrix': cm,
        'labels': labels,
        'train_time': train_time
    }


def plot_comparison(results_2d, results_3d):
    """Plot confusion matrices for 2D and 3D concatenated models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 2D
    sns.heatmap(results_2d['confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=results_2d['labels'],
                yticklabels=results_2d['labels'],
                ax=axes[0])
    axes[0].set_title(f"2D Concatenated (3 views)\nAccuracy: {results_2d['test_accuracy_smoothed']:.3f}", 
                     fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 3D
    sns.heatmap(results_3d['confusion_matrix'], 
                annot=True, fmt='d', cmap='Greens',
                xticklabels=results_3d['labels'],
                yticklabels=results_3d['labels'],
                ax=axes[1])
    axes[1].set_title(f"3D Concatenated (3 views + MiDaS)\nAccuracy: {results_3d['test_accuracy_smoothed']:.3f}", 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.suptitle('SVM Concatenated Multi-View (Original Temporal Split)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = 'svm_concatenated_original_split.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_file}")


def main():
    print("="*70)
    print("SVM WITH CONCATENATED MULTI-VIEW FEATURES")
    print("USING ORIGINAL TEMPORAL SPLIT (before 1:59 = train/val, after = test)")
    print("="*70)
    
    base_dir = "/DATA/summer_students/process_OO/COMS4731/COMS4731-final-project/dataset_midas_3d"
    cameras = ['left', 'right', 'middle']
    
    all_results = {}
    
    for skeleton_type in ['2d', '3d']:
        print(f"\n\n{'#'*70}")
        print(f"# {skeleton_type.upper()} SKELETONS")
        print(f"{'#'*70}")
        
        # Load concatenated data from ORIGINAL splits
        print(f"\nLoading concatenated {skeleton_type.upper()} features from original splits...")
        
        X_train, y_train, frames_train = load_concatenated_skeletons_original(
            base_dir, 'train', skeleton_type, cameras
        )
        
        X_val, y_val, frames_val = load_concatenated_skeletons_original(
            base_dir, 'val', skeleton_type, cameras
        )
        
        X_test, y_test, frames_test = load_concatenated_skeletons_original(
            base_dir, 'test', skeleton_type, cameras
        )
        
        if X_train is None:
            print(f"No data found for {skeleton_type}")
            continue
        
        # Train and evaluate
        results = train_and_evaluate_concatenated_svm(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            frames_test,
            skeleton_type=skeleton_type.upper()
        )
        
        all_results[skeleton_type] = results
    
    # Plot comparison
    if '2d' in all_results and '3d' in all_results:
        plot_comparison(all_results['2d'], all_results['3d'])
    
    # Final summary
    print(f"\n\n{'='*70}")
    print(f"FINAL COMPARISON: CONCATENATED MULTI-VIEW SVM")
    print(f"(ORIGINAL TEMPORAL SPLIT)")
    print(f"{'='*70}")
    
    if '2d' in all_results:
        print(f"\n2D Concatenated (3 views × 66 features = 198 total):")
        print(f"  Val accuracy:  {all_results['2d']['val_accuracy']:.4f}")
        print(f"  Test accuracy (20-frame smooth): {all_results['2d']['test_accuracy_smoothed']:.4f}")
        print(f"  Training time: {all_results['2d']['train_time']:.2f}s")
    
    if '3d' in all_results:
        print(f"\n3D Concatenated (3 views × 99 features = 297 total):")
        print(f"  Val accuracy:  {all_results['3d']['val_accuracy']:.4f}")
        print(f"  Test accuracy (20-frame smooth): {all_results['3d']['test_accuracy_smoothed']:.4f}")
        print(f"  Training time: {all_results['3d']['train_time']:.2f}s")
    
    if '2d' in all_results and '3d' in all_results:
        improvement = all_results['3d']['test_accuracy_smoothed'] - all_results['2d']['test_accuracy_smoothed']
        print(f"\n3D vs 2D improvement: {improvement:+.4f}")
        
        if improvement > 0:
            print(f"✓ MiDaS depth information HELPS with concatenated features!")
        elif improvement < 0:
            print(f"⚠ 2D performs better (depth may add noise)")
        else:
            print(f"= Both perform equally")
    
    print(f"\n✓ All done!")


if __name__ == "__main__":
    main()


