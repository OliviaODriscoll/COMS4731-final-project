"""
Standalone script for interactive homography computation using RANSAC.

Usage:
    python compute_homography_ransac.py <image1_path> <image2_path>

This script will:
1. Display both images for you to click and select corresponding points
2. Use RANSAC to compute a robust homography matrix
3. Visualize the correspondences (inliers in green, outliers in red)
4. Save the homography matrix and visualization
"""

import cv2
import numpy as np
import os
import sys
from depth_map_3cameras import MultiViewDepthReconstructor


def main():
    if len(sys.argv) < 3:
        print("Usage: python compute_homography_ransac.py <image1_path> <image2_path>")
        print("\nExample:")
        print("  python compute_homography_ransac.py image1.png image2.png")
        sys.exit(1)
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    
    # Check if files exist
    if not os.path.exists(img1_path):
        print(f"Error: Image 1 not found: {img1_path}")
        sys.exit(1)
    
    if not os.path.exists(img2_path):
        print(f"Error: Image 2 not found: {img2_path}")
        sys.exit(1)
    
    # Initialize reconstructor
    reconstructor = MultiViewDepthReconstructor()
    
    # Output directory
    output_dir = '/Users/olivia/Documents/COMS4731/COMS4731-final-project/homography_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run interactive homography computation
    homography, points1, points2, mask = reconstructor.select_points_for_homography(
        img1_path,
        img2_path,
        save_correspondences_path=os.path.join(output_dir, 'correspondences.png')
    )
    
    if homography is not None:
        # Save homography matrix
        homography_path = os.path.join(output_dir, 'homography_matrix.npy')
        np.save(homography_path, homography)
        print(f"\n✓ Homography matrix saved to: {homography_path}")
        
        # Save points and mask
        if points1 and points2:
            points_data = {
                'points1': np.array(points1),
                'points2': np.array(points2),
                'mask': mask.flatten() if mask is not None else None
            }
            points_path = os.path.join(output_dir, 'correspondence_points.npy')
            np.save(points_path, points_data, allow_pickle=True)
            print(f"✓ Correspondence points saved to: {points_path}")
        
        # Demonstrate warping
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is not None and img2 is not None:
            # Warp image 2 to align with image 1
            warped = reconstructor.apply_homography(
                img2, homography,
                output_size=(img1.shape[1], img1.shape[0])
            )
            
            warped_path = os.path.join(output_dir, 'warped_image.png')
            cv2.imwrite(warped_path, warped)
            print(f"✓ Warped image saved to: {warped_path}")
            
            # Create side-by-side comparison
            comparison = np.hstack([img1, warped])
            comparison_path = os.path.join(output_dir, 'comparison_original_warped.png')
            cv2.imwrite(comparison_path, comparison)
            print(f"✓ Comparison saved to: {comparison_path}")
            
            print(f"\n✓ All results saved to: {output_dir}")
            print("\nYou can now use the homography matrix for:")
            print("  - Image alignment")
            print("  - Stereo rectification")
            print("  - Depth estimation")
    else:
        print("\n✗ Failed to compute homography. Please try again with more points.")


if __name__ == "__main__":
    main()

