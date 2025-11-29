"""
Triangulate 3D Skeleton from Multiple 2D Views

Uses MediaPipe 2D skeletons from multiple camera views and camera calibration
to reconstruct 3D skeleton coordinates via triangulation.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse
from typing import List, Tuple, Optional, Dict
import json


class MultiViewTriangulator:
    """
    Triangulates 3D points from 2D correspondences across multiple camera views.
    """
    
    def __init__(self, calibration_files: Dict[str, str], camera_poses: Optional[Dict[str, np.ndarray]] = None):
        """
        Initialize triangulator with camera calibrations.
        
        Args:
            calibration_files: Dictionary mapping camera names to calibration .npz file paths
                              e.g., {'left': 'calibration/calibration_left.npz', ...}
            camera_poses: Optional dictionary of camera poses (rotation and translation)
                         If None, will assume middle camera is at origin and estimate others
        """
        self.calibration_files = calibration_files
        self.camera_names = list(calibration_files.keys())
        self.num_cameras = len(calibration_files)
        
        # Load camera intrinsics
        self.camera_matrices = {}
        self.dist_coeffs = {}
        self.image_sizes = {}
        
        for cam_name, calib_path in calibration_files.items():
            calib_data = np.load(calib_path)
            self.camera_matrices[cam_name] = calib_data['camera_matrix']
            self.dist_coeffs[cam_name] = calib_data['dist_coeffs']
            self.image_sizes[cam_name] = tuple(calib_data['image_size'])
            print(f"Loaded calibration for {cam_name}: {self.image_sizes[cam_name]}")
        
        # Set up camera poses (extrinsics)
        if camera_poses is None:
            # Default: assume middle camera is at origin, others are offset
            # This is a simplified setup - you may need to calibrate camera-to-camera poses
            self.camera_poses = self._estimate_default_poses()
        else:
            self.camera_poses = camera_poses
        
        # Compute projection matrices P = K * [R|t] for each camera
        self.projection_matrices = {}
        for cam_name in self.camera_names:
            K = self.camera_matrices[cam_name]
            R, t = self.camera_poses[cam_name]
            # Convert rotation vector to rotation matrix if needed
            if R.shape == (3,):
                R, _ = cv2.Rodrigues(R)
            elif R.shape == (3, 1):
                R, _ = cv2.Rodrigues(R.ravel())
            
            # Create [R|t] matrix
            Rt = np.hstack([R, t.reshape(3, 1)])
            # Projection matrix P = K * [R|t]
            P = K @ Rt
            self.projection_matrices[cam_name] = P
            print(f"Projection matrix for {cam_name} computed")
    
    def _estimate_default_poses(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate default camera poses. Assumes middle camera at origin.
        For proper calibration, you should use stereo calibration or structure-from-motion.
        
        Returns:
            Dictionary of (rotation_matrix, translation_vector) for each camera
        """
        poses = {}
        
        # Assume middle camera is at origin (identity)
        if 'middle' in self.camera_names:
            poses['middle'] = (np.eye(3), np.zeros(3))
        
        # Estimate relative poses for other cameras
        # These are rough estimates - for accurate results, use proper stereo calibration
        if 'left' in self.camera_names:
            # Left camera: rotated and translated relative to middle
            # Rough estimate: assume left camera is to the left of middle
            R_left = np.eye(3)  # Identity for now
            t_left = np.array([-0.5, 0.0, 0.0])  # 0.5m to the left (adjust based on setup)
            poses['left'] = (R_left, t_left)
        
        if 'right' in self.camera_names:
            # Right camera: rotated and translated relative to middle
            R_right = np.eye(3)
            t_right = np.array([0.5, 0.0, 0.0])  # 0.5m to the right (adjust based on setup)
            poses['right'] = (R_right, t_right)
        
        return poses
    
    def triangulate_point(self, points_2d: Dict[str, np.ndarray], min_views: int = 2) -> Optional[np.ndarray]:
        """
        Triangulate a single 3D point from 2D correspondences.
        
        Args:
            points_2d: Dictionary mapping camera names to 2D points (x, y)
            min_views: Minimum number of views required (default: 2)
            
        Returns:
            3D point as numpy array [X, Y, Z] or None if insufficient views
        """
        # Filter out None/invalid points
        valid_points = {cam: pt for cam, pt in points_2d.items() 
                       if pt is not None and not np.isnan(pt).any()}
        
        if len(valid_points) < min_views:
            return None
        
        # Collect projection matrices and points
        proj_matrices = []
        points = []
        
        for cam_name, point_2d in valid_points.items():
            if cam_name in self.projection_matrices:
                proj_matrices.append(self.projection_matrices[cam_name])
                points.append(point_2d)
        
        if len(proj_matrices) < 2:
            return None
        
        # Use OpenCV's triangulation
        points = np.array(points, dtype=np.float32)
        proj_matrices = np.array(proj_matrices, dtype=np.float32)
        
        # OpenCV triangulatePoints expects points in shape (2, N) and projection matrices (3, 4)
        # For multiple views, we'll use the first two views
        if len(proj_matrices) == 2:
            point_3d = cv2.triangulatePoints(
                proj_matrices[0], proj_matrices[1],
                points[0].reshape(2, 1), points[1].reshape(2, 1)
            )
        else:
            # For 3+ views, use first two or implement multi-view triangulation
            # For now, use first two views
            point_3d = cv2.triangulatePoints(
                proj_matrices[0], proj_matrices[1],
                points[0].reshape(2, 1), points[1].reshape(2, 1)
            )
        
        # Convert from homogeneous coordinates
        if point_3d[3] != 0:
            point_3d = point_3d[:3] / point_3d[3]
        else:
            return None
        
        return point_3d.ravel()
    
    def triangulate_skeleton(self, skeletons_2d: Dict[str, np.ndarray], min_views: int = 2) -> np.ndarray:
        """
        Triangulate full skeleton from 2D skeletons across views.
        
        Args:
            skeletons_2d: Dictionary mapping camera names to 2D skeleton arrays (33, 2)
            min_views: Minimum number of views required per joint
            
        Returns:
            3D skeleton array of shape (33, 3) with NaN for invalid points
        """
        num_joints = 33  # MediaPipe has 33 landmarks
        skeleton_3d = np.full((num_joints, 3), np.nan)
        
        for joint_idx in range(num_joints):
            # Collect 2D points for this joint from all cameras
            points_2d = {}
            for cam_name, skeleton in skeletons_2d.items():
                if skeleton is not None and not np.isnan(skeleton[joint_idx]).any():
                    points_2d[cam_name] = skeleton[joint_idx]
            
            # Triangulate
            point_3d = self.triangulate_point(points_2d, min_views=min_views)
            if point_3d is not None:
                skeleton_3d[joint_idx] = point_3d
        
        return skeleton_3d


def load_2d_skeleton(frame_path: Path) -> Optional[np.ndarray]:
    """
    Load 2D skeleton from .npy file.
    
    Args:
        frame_path: Path to skeleton_2d_coords_frame_XXXXX.npy file
        
    Returns:
        2D skeleton array of shape (33, 2) or None if file doesn't exist
    """
    if not frame_path.exists():
        return None
    
    skeleton = np.load(frame_path)
    return skeleton


def process_multi_view_triangulation(
    skeleton_dirs: Dict[str, str],
    calibration_files: Dict[str, str],
    output_dir: str,
    frame_range: Optional[Tuple[int, int]] = None,
    min_views: int = 2,
    camera_poses_file: Optional[str] = None
):
    """
    Process 2D skeletons from multiple views to create 3D skeletons via triangulation.
    
    Args:
        skeleton_dirs: Dictionary mapping camera names to directories with 2D skeleton files
        calibration_files: Dictionary mapping camera names to calibration .npz files
        output_dir: Directory to save 3D skeleton results
        frame_range: Optional (start_frame, end_frame) tuple to process specific range
        min_views: Minimum number of views required for triangulation
        camera_poses_file: Optional JSON file with camera poses (if pre-calibrated)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load camera poses if provided
    camera_poses = None
    if camera_poses_file and os.path.exists(camera_poses_file):
        with open(camera_poses_file, 'r') as f:
            poses_data = json.load(f)
            camera_poses = {}
            for cam_name, pose_data in poses_data.items():
                R = np.array(pose_data['rotation'])
                t = np.array(pose_data['translation'])
                camera_poses[cam_name] = (R, t)
        print(f"Loaded camera poses from {camera_poses_file}")
    
    # Initialize triangulator
    triangulator = MultiViewTriangulator(calibration_files, camera_poses)
    
    # Find all skeleton files and determine frame range
    skeleton_files = {}
    frame_numbers = set()
    
    for cam_name, skeleton_dir in skeleton_dirs.items():
        skeleton_dir = Path(skeleton_dir)
        files = sorted(
            skeleton_dir.glob("skeleton_2d_coords_frame_*.npy"),
            key=lambda x: int(x.stem.split('_')[-1])
        )
        skeleton_files[cam_name] = {int(f.stem.split('_')[-1]): f for f in files}
        frame_numbers.update(skeleton_files[cam_name].keys())
    
    if not frame_numbers:
        raise ValueError("No skeleton files found in any directory")
    
    frame_numbers = sorted(frame_numbers)
    
    # Apply frame range if specified
    if frame_range:
        start_frame, end_frame = frame_range
        frame_numbers = [f for f in frame_numbers if start_frame <= f <= end_frame]
    
    print(f"\nProcessing {len(frame_numbers)} frames")
    print(f"Frame range: {min(frame_numbers)} to {max(frame_numbers)}")
    print(f"Cameras: {list(skeleton_dirs.keys())}")
    
    processed_count = 0
    skipped_count = 0
    
    for frame_num in frame_numbers:
        # Load 2D skeletons from all views
        skeletons_2d = {}
        for cam_name in skeleton_dirs.keys():
            if frame_num in skeleton_files[cam_name]:
                skeleton_path = skeleton_files[cam_name][frame_num]
                skeleton = load_2d_skeleton(skeleton_path)
                if skeleton is not None:
                    skeletons_2d[cam_name] = skeleton
        
        # Check if we have enough views
        if len(skeletons_2d) < min_views:
            skipped_count += 1
            continue
        
        # Triangulate to 3D
        skeleton_3d = triangulator.triangulate_skeleton(skeletons_2d, min_views=min_views)
        
        # Save 3D skeleton
        output_path = os.path.join(output_dir, f"skeleton_3d_coords_frame_{frame_num:05d}.npy")
        np.save(output_path, skeleton_3d)
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} frames...")
    
    print(f"\n✓ Triangulation complete!")
    print(f"  Processed: {processed_count} frames")
    print(f"  Skipped: {skipped_count} frames (insufficient views)")
    print(f"  Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Triangulate 3D skeletons from 2D MediaPipe skeletons across multiple camera views"
    )
    
    parser.add_argument("--skeleton-dirs", type=str, nargs='+', required=True,
                       help="Directories containing 2D skeleton files (space-separated, in order: left middle right)")
    parser.add_argument("--camera-names", type=str, nargs='+', default=None,
                       help="Camera names corresponding to skeleton-dirs (default: left middle right)")
    parser.add_argument("--calibration-files", type=str, nargs='+', required=True,
                       help="Calibration .npz files (space-separated, same order as skeleton-dirs)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for 3D skeleton files")
    parser.add_argument("--frame-start", type=int, default=None,
                       help="Starting frame number")
    parser.add_argument("--frame-end", type=int, default=None,
                       help="Ending frame number")
    parser.add_argument("--min-views", type=int, default=2,
                       help="Minimum number of views required for triangulation (default: 2)")
    parser.add_argument("--camera-poses", type=str, default=None,
                       help="JSON file with camera poses (rotation and translation matrices)")
    
    args = parser.parse_args()
    
    # Set up camera names
    if args.camera_names:
        camera_names = args.camera_names
    else:
        # Default names based on number of cameras
        if len(args.skeleton_dirs) == 3:
            camera_names = ['left', 'middle', 'right']
        else:
            camera_names = [f'camera_{i}' for i in range(len(args.skeleton_dirs))]
    
    if len(args.skeleton_dirs) != len(args.calibration_files):
        raise ValueError("Number of skeleton directories must match number of calibration files")
    
    if len(args.skeleton_dirs) != len(camera_names):
        raise ValueError("Number of skeleton directories must match number of camera names")
    
    # Create dictionaries
    skeleton_dirs = {name: dir_path for name, dir_path in zip(camera_names, args.skeleton_dirs)}
    calibration_files = {name: calib_path for name, calib_path in zip(camera_names, args.calibration_files)}
    
    # Frame range
    frame_range = None
    if args.frame_start is not None and args.frame_end is not None:
        frame_range = (args.frame_start, args.frame_end)
    
    print("="*60)
    print("Multi-View 3D Skeleton Triangulation")
    print("="*60)
    print(f"\nCamera setup:")
    for name, dir_path in skeleton_dirs.items():
        print(f"  {name}: {dir_path}")
        print(f"    Calibration: {calibration_files[name]}")
    
    process_multi_view_triangulation(
        skeleton_dirs=skeleton_dirs,
        calibration_files=calibration_files,
        output_dir=args.output,
        frame_range=frame_range,
        min_views=args.min_views,
        camera_poses_file=args.camera_poses
    )


if __name__ == "__main__":
    main()

