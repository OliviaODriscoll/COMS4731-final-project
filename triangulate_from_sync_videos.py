"""
Triangulate 3D skeletons from synchronized 2D MediaPipe skeletons.

Uses the 2D skeleton outputs from process_synchronized_videos.py
and camera calibration to create 3D skeletons.
"""

import argparse
import os
from pathlib import Path
import numpy as np
from triangulate_3d_skeleton import MultiViewTriangulator, process_multi_view_triangulation


def main():
    parser = argparse.ArgumentParser(
        description="Triangulate 3D skeletons from synchronized 2D skeletons"
    )
    
    parser.add_argument("--skeleton-dirs", type=str, nargs='+',
                       default=[
                           "/Users/olivia/Desktop/COMS4731-final-project/2d_skeletons/arabesque_left",
                           "/Users/olivia/Desktop/COMS4731-final-project/2d_skeletons/barre_right",
                           "/Users/olivia/Desktop/COMS4731-final-project/2d_skeletons/barre_tripod"
                       ],
                       help="Directories containing synchronized 2D skeleton files")
    parser.add_argument("--camera-names", type=str, nargs='+',
                       default=["left", "right", "middle"],
                       help="Camera names (default: left right middle)")
    parser.add_argument("--calibration-files", type=str, nargs='+',
                       default=[
                           "/Users/olivia/Desktop/COMS4731-final-project/calibration/calibration_left.npz",
                           "/Users/olivia/Desktop/COMS4731-final-project/calibration/calibration_right.npz",
                           "/Users/olivia/Desktop/COMS4731-final-project/calibration/calibration_middle.npz"
                       ],
                       help="Calibration .npz files (same order as skeleton-dirs)")
    parser.add_argument("--output", type=str,
                       default="/Users/olivia/Desktop/COMS4731-final-project/3d_skeletons_triangulated",
                       help="Output directory for 3D skeleton files")
    parser.add_argument("--camera-poses", type=str, default=None,
                       help="JSON file with camera poses (optional, will estimate if not provided)")
    parser.add_argument("--min-views", type=int, default=2,
                       help="Minimum number of views required for triangulation (default: 2)")
    parser.add_argument("--frame-start", type=int, default=None,
                       help="Starting frame number (default: all frames)")
    parser.add_argument("--frame-end", type=int, default=None,
                       help="Ending frame number (default: all frames)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.skeleton_dirs) != len(args.calibration_files):
        raise ValueError("Number of skeleton directories must match number of calibration files")
    
    if len(args.skeleton_dirs) != len(args.camera_names):
        raise ValueError("Number of skeleton directories must match number of camera names")
    
    if len(args.skeleton_dirs) < 2:
        raise ValueError("Need at least 2 camera views for triangulation")
    
    # Create dictionaries
    skeleton_dirs = {name: dir_path for name, dir_path in zip(args.camera_names, args.skeleton_dirs)}
    calibration_files = {name: calib_path for name, calib_path in zip(args.camera_names, args.calibration_files)}
    
    # Frame range
    frame_range = None
    if args.frame_start is not None and args.frame_end is not None:
        frame_range = (args.frame_start, args.frame_end)
    
    print("="*60)
    print("Triangulating 3D Skeletons from Synchronized 2D Views")
    print("="*60)
    print(f"\nCamera setup:")
    for name, dir_path in skeleton_dirs.items():
        print(f"  {name}:")
        print(f"    Skeletons: {dir_path}")
        print(f"    Calibration: {calibration_files[name]}")
    
    print(f"\nOutput: {args.output}")
    print(f"Minimum views: {args.min_views}")
    if frame_range:
        print(f"Frame range: {frame_range[0]} to {frame_range[1]}")
    print()
    
    # Process triangulation
    process_multi_view_triangulation(
        skeleton_dirs=skeleton_dirs,
        calibration_files=calibration_files,
        output_dir=args.output,
        frame_range=frame_range,
        min_views=args.min_views,
        camera_poses_file=args.camera_poses
    )
    
    print("\n" + "="*60)
    print("Triangulation complete!")
    print("="*60)
    print(f"\n3D skeletons saved to: {args.output}")
    print("\nYou can now use these 3D skeletons with your classifier:")
    print(f"  python train_ballet_classifier.py \\")
    print(f"      --skeleton-dir {args.output} \\")
    print(f"      --label-csv <path_to_labels.csv>")


if __name__ == "__main__":
    main()

