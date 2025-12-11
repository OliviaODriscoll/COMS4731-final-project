#!/usr/bin/env python3
"""
Train SVM on each camera angle (left, right, middle) and use majority voting.
Ensemble learning with multi-view fusion for ballet pose classification.
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


def load_skeletons_from_split(dataset_dir, split, skeleton_type='2d', use_resplit=True):
    """
    Load skeletons from a split directory or resplit CSV.
    
    Args:
        dataset_dir: Path to camera dataset directory
        split: 'train', 'val', or 'test'
        skeleton_type: '2d' or '3d'
        use_resplit: If True, load from resplit/ CSV pointers; if False, use original splits
    """
    dataset_dir = Path(dataset_dir)
    
    if use_resplit:
        # Load from resplit CSV pointers
        resplit_dir = dataset_dir / "resplit"
        frames_csv = resplit_dir / f"{split}_frames.csv"
        
        if not frames_csv.exists():
            print(f"Warning: Resplit not found at {frames_csv}")
            print(f"Falling back to original split. Run: python3 resplit_dataset.py --camera {dataset_dir.name}")
            use_resplit = False
    
    if not use_resplit:
        # Load from original split directory
        split_dir = dataset_dir / split
        frames_csv = split_dir / "frames.csv"
        
        if not frames_csv.exists():
            return None, None, None
    
    frames_df = pd.read_csv(frames_csv)
    
    X_list = []
    y_list = []
    frame_nums = []
    
    for _, row in frames_df.iterrows():
        if use_resplit:
            # Path is relative to dataset_dir (e.g., "train/2d_skeletons/skeleton_2d_00750.npy")
            skeleton_rel_path = row[f'skeleton_{skeleton_type}']
            skeleton_path = dataset_dir / skeleton_rel_path
        else:
            # Path is relative to split_dir
            skeleton_path = split_dir / row[f'skeleton_{skeleton_type}']
        
        if not skeleton_path.exists():
            continue
        
        skeleton = np.load(skeleton_path)
        features = skeleton.flatten()
        features = np.nan_to_num(features, nan=0.0)
        
        X_list.append(features)
        y_list.append(row['label'])
        frame_nums.append(row['frame'])
    
    if len(X_list) == 0:
        return None, None, None
    
    X = np.array(X_list)
    y = np.array(y_list)
    frame_nums = np.array(frame_nums)
    
    # Remove unlabeled
    labeled_mask = y != 'unlabeled'
    return X[labeled_mask], y[labeled_mask], frame_nums[labeled_mask]


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


def majority_vote(predictions_list):
    """
    Apply majority voting across multiple predictions.
    Handles predictions of different lengths by taking minimum length.
    
    Args:
        predictions_list: List of prediction arrays from different models
    
    Returns:
        voted_predictions: Array of majority-voted predictions
    """
    # Use minimum length across all predictions
    n_samples = min(len(preds) for preds in predictions_list)
    voted = []
    
    for i in range(n_samples):
        # Get prediction from each model for this sample
        votes = [preds[i] for preds in predictions_list]
        
        # Majority vote
        counter = Counter(votes)
        voted_pred = counter.most_common(1)[0][0]
        voted.append(voted_pred)
    
    return np.array(voted)


def train_camera_svm(camera_name, base_dir, skeleton_type='3d', use_resplit=True):
    """Train SVM for a single camera."""
    camera_dir = Path(base_dir) / camera_name
    
    if not camera_dir.exists():
        print(f"Warning: {camera_dir} not found")
        return None
    
    print(f"\n{'='*70}")
    print(f"TRAINING {camera_name.upper()} CAMERA - {skeleton_type.upper()}")
    if use_resplit:
        print(f"Using RESPLIT data (shuffled)")
    else:
        print(f"Using ORIGINAL split (temporal)")
    print(f"{'='*70}")
    
    # Load data
    print(f"Loading data...")
    X_train, y_train, _ = load_skeletons_from_split(camera_dir, 'train', skeleton_type, use_resplit)
    X_val, y_val, _ = load_skeletons_from_split(camera_dir, 'val', skeleton_type, use_resplit)
    X_test, y_test, frames_test = load_skeletons_from_split(camera_dir, 'test', skeleton_type, use_resplit)
    
    if X_train is None:
        print(f"No data found for {camera_name}")
        return None
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    print(f"Training SVM (RBF kernel)...")
    start_time = time.time()
    
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    train_time = time.time() - start_time
    print(f"  Training time: {train_time:.2f}s")
    
    # Evaluate
    y_test_pred = svm.predict(X_test_scaled)
    
    # Sort by frame number for temporal consistency
    sort_idx = np.argsort(frames_test)
    y_test_sorted = y_test[sort_idx]
    y_test_pred_sorted = y_test_pred[sort_idx]
    
    # Apply temporal smoothing
    y_test_smoothed = temporal_smoothing(y_test_pred_sorted, window_size=20)
    
    test_acc = accuracy_score(y_test_sorted, y_test_smoothed)
    print(f"  Test Accuracy (20-frame smoothing): {test_acc:.4f}")
    
    return {
        'camera': camera_name,
        'model': svm,
        'scaler': scaler,
        'accuracy': test_acc,
        'y_test': y_test_sorted,
        'y_pred': y_test_smoothed,
        'frames_test': frames_test[sort_idx]
    }


def ensemble_predict(models, base_dir, skeleton_type='3d', use_resplit=True):
    """
    Make ensemble predictions using majority voting across all cameras.
    
    Args:
        models: List of trained model dictionaries
        base_dir: Base dataset directory
        skeleton_type: '2d' or '3d'
        use_resplit: Whether to use resplit data
    
    Returns:
        Dictionary with ensemble results
    """
    print(f"\n{'='*70}")
    print(f"ENSEMBLE PREDICTION (MAJORITY VOTING)")
    print(f"{'='*70}")
    
    # We need to ensure all cameras have same test samples
    # For simplicity, we'll use the test set from each camera independently
    # then align by frame numbers
    
    all_predictions = []
    y_test_true = None
    all_labels = None
    
    for model_dict in models:
        camera = model_dict['camera']
        camera_dir = Path(base_dir) / camera
        
        # Load test data
        X_test, y_test, frames_test = load_skeletons_from_split(
            camera_dir, 'test', skeleton_type, use_resplit
        )
        
        # Standardize and predict
        X_test_scaled = model_dict['scaler'].transform(X_test)
        y_pred = model_dict['model'].predict(X_test_scaled)
        
        # Sort by frame number
        sort_idx = np.argsort(frames_test)
        y_test_sorted = y_test[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        
        # Apply temporal smoothing
        y_pred_smoothed = temporal_smoothing(y_pred_sorted, window_size=20)
        
        all_predictions.append(y_pred_smoothed)
        
        if y_test_true is None:
            y_test_true = y_test_sorted
        
        print(f"  {camera}: {len(y_pred_smoothed)} predictions")
    
    # Majority voting across cameras
    print(f"\nApplying majority voting across {len(all_predictions)} cameras...")
    
    # Use minimum length
    min_len = min(len(preds) for preds in all_predictions)
    print(f"  Using first {min_len} samples (minimum across cameras)")
    
    y_ensemble = majority_vote(all_predictions)
    
    # Trim ground truth to match ensemble length
    y_test_true_trimmed = y_test_true[:min_len]
    
    # Calculate ensemble accuracy
    ensemble_acc = accuracy_score(y_test_true_trimmed, y_ensemble)
    
    print(f"\nEnsemble Accuracy: {ensemble_acc:.4f}")
    
    # Detailed report
    print(f"\n{'='*70}")
    print(f"ENSEMBLE CLASSIFICATION REPORT")
    print(f"{'='*70}")
    print(classification_report(y_test_true_trimmed, y_ensemble, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_true_trimmed, y_ensemble)
    labels = sorted(list(set(y_test_true_trimmed)))
    
    return {
        'accuracy': ensemble_acc,
        'y_test': y_test_true_trimmed,
        'y_pred': y_ensemble,
        'confusion_matrix': cm,
        'labels': labels,
        'individual_predictions': all_predictions
    }


def plot_results(individual_results, ensemble_results, skeleton_type='3D'):
    """Plot comparison of individual cameras vs ensemble."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Individual camera confusion matrices
    for idx, result in enumerate(individual_results[:3]):
        row = idx // 2
        col = idx % 2
        
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        labels = sorted(list(set(result['y_test'])))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   ax=axes[row, col])
        axes[row, col].set_title(
            f"{result['camera'].upper()} Camera\nAccuracy: {result['accuracy']:.3f}",
            fontsize=12, fontweight='bold'
        )
        axes[row, col].set_xlabel('Predicted')
        axes[row, col].set_ylabel('True')
        axes[row, col].tick_params(axis='x', rotation=45)
    
    # Ensemble confusion matrix
    sns.heatmap(ensemble_results['confusion_matrix'], 
               annot=True, fmt='d', cmap='Greens',
               xticklabels=ensemble_results['labels'],
               yticklabels=ensemble_results['labels'],
               ax=axes[1, 1])
    axes[1, 1].set_title(
        f"ENSEMBLE (Majority Voting)\nAccuracy: {ensemble_results['accuracy']:.3f}",
        fontsize=12, fontweight='bold'
    )
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('True')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'{skeleton_type} Skeletons - Multi-View SVM Ensemble', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_file = f'svm_ensemble_{skeleton_type.lower()}_voting.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_file}")


def main():
    print("="*70)
    print("MULTI-VIEW SVM ENSEMBLE WITH MAJORITY VOTING")
    print("="*70)
    print("Training separate SVMs on each camera angle")
    print("Combining predictions using majority voting")
    print("Testing on BOTH 2D and 3D skeletons")
    print("="*70)
    
    base_dir = "/DATA/summer_students/process_OO/COMS4731/COMS4731-final-project/dataset_midas_3d"
    cameras = ['left', 'right', 'middle']
    
    # USE RESPLIT DATA (shuffled) instead of temporal split
    use_resplit = True
    
    print(f"\nData source: {'RESPLIT (shuffled)' if use_resplit else 'ORIGINAL (temporal)'}")
    print("="*70)
    
    # Train on BOTH 2D and 3D
    all_results = {}
    
    for skeleton_type in ['2d', '3d']:
        print(f"\n\n{'#'*70}")
        print(f"# PROCESSING {skeleton_type.upper()} SKELETONS")
        print(f"{'#'*70}")
        
        # Train individual SVMs for each camera
        individual_results = []
        
        for camera in cameras:
            result = train_camera_svm(camera, base_dir, skeleton_type, use_resplit)
            if result:
                individual_results.append(result)
        
        # Ensemble prediction
        ensemble_results = ensemble_predict(individual_results, base_dir, skeleton_type, use_resplit)
        
        # Summary comparison
        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY ({skeleton_type.upper()} SKELETONS)")
        print(f"{'='*70}")
        
        print(f"\nIndividual Cameras:")
        for result in individual_results:
            print(f"  {result['camera']:6s}: {result['accuracy']:.4f}")
        
        best_individual = max(individual_results, key=lambda x: x['accuracy'])
        print(f"\nBest Individual: {best_individual['camera']} = {best_individual['accuracy']:.4f}")
        print(f"Ensemble:        {ensemble_results['accuracy']:.4f}")
        
        improvement = ensemble_results['accuracy'] - best_individual['accuracy']
        print(f"\nEnsemble Improvement: {improvement:+.4f}")
        
        if improvement > 0:
            print(f"✓ Ensemble performs BETTER than best individual camera!")
        elif improvement < 0:
            print(f"⚠ Best individual camera performs better than ensemble")
        else:
            print(f"= Ensemble matches best individual camera")
        
        # Plot results
        plot_results(individual_results, ensemble_results, skeleton_type.upper())
        
        # Store results
        all_results[skeleton_type] = {
            'individual': individual_results,
            'ensemble': ensemble_results,
            'best_individual': best_individual
        }
    
    # Final comparison: 2D vs 3D
    print(f"\n\n{'='*70}")
    print(f"FINAL COMPARISON: 2D vs 3D SKELETONS")
    print(f"{'='*70}")
    
    print(f"\n2D Skeletons (MediaPipe only):")
    print(f"  Best individual: {all_results['2d']['best_individual']['accuracy']:.4f}")
    print(f"  Ensemble:        {all_results['2d']['ensemble']['accuracy']:.4f}")
    
    print(f"\n3D Skeletons (MediaPipe + MiDaS depth):")
    print(f"  Best individual: {all_results['3d']['best_individual']['accuracy']:.4f}")
    print(f"  Ensemble:        {all_results['3d']['ensemble']['accuracy']:.4f}")
    
    improvement_3d_vs_2d = all_results['3d']['ensemble']['accuracy'] - all_results['2d']['ensemble']['accuracy']
    print(f"\n3D vs 2D Ensemble Improvement: {improvement_3d_vs_2d:+.4f}")
    
    if improvement_3d_vs_2d > 0:
        print(f"✓ MiDaS depth information HELPS classification!")
    elif improvement_3d_vs_2d < 0:
        print(f"⚠ 2D performs better (depth may be adding noise)")
    else:
        print(f"= Both perform equally")
    
    print(f"\n✓ All done!")


if __name__ == "__main__":
    main()

