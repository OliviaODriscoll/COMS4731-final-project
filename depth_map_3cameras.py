import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MultiViewDepthReconstructor:
    def __init__(self):
        self.camera_matrices = {}  # Intrinsic parameters (K)
        self.distortion_coeffs = {}  # Distortion coefficients
        self.image_size = None
        self.stereo_pairs = {}  # Store stereo calibration results
        
    def calibrate_single_camera_from_images(self, calibration_image_paths, camera_name, 
                                            chessboard_size=(9, 6), square_size=1.0):
        """
        Calibrate a single camera using checkerboard pattern from image files
        
        Args:
            calibration_image_paths: List of paths to calibration images (or single path)
            camera_name: Name identifier for the camera
            chessboard_size: Tuple (cols, rows) of inner corners
            square_size: Size of chessboard squares in real world units (e.g., mm)
        """
        print(f"\n=== Calibrating {camera_name} ===")
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points (3D points in real world space)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size  # Scale by square size
        
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        # Handle both single image path and list of paths
        if isinstance(calibration_image_paths, str):
            calibration_image_paths = [calibration_image_paths]
        
        print(f"Processing {len(calibration_image_paths)} calibration image(s)...")
        for img_path in calibration_image_paths:
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Warning: Could not read {img_path}, skipping...")
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            
            if ret_corners:
                objpoints.append(objp.copy())
                # Refine corner positions
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                
                # Draw corners
                cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret_corners)
                print(f"✓ Found pattern in {os.path.basename(img_path)}")
                
                # Store image size from first successful detection
                if self.image_size is None:
                    self.image_size = gray.shape[::-1]
            else:
                print(f"✗ Could not find pattern in {os.path.basename(img_path)}")
        
        print(f"\nFound {len(objpoints)} valid calibration images")
        
        if len(objpoints) < 1:
            print(f"Error: Need at least 1 calibration image with detected pattern, found {len(objpoints)}")
            return False
        
        # Perform camera calibration
        print("Performing camera calibration...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, self.image_size, None, None
        )
        
        if ret:
            self.camera_matrices[camera_name] = mtx
            self.distortion_coeffs[camera_name] = dist
            
            # Calculate reprojection error
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            
            print(f"✓ {camera_name} calibration successful!")
            print(f"  Mean reprojection error: {mean_error/len(objpoints):.3f} pixels")
            print(f"  Camera matrix:\n{mtx}")
            print(f"  Distortion coefficients: {dist.flatten()}")
            return True
        else:
            print(f"✗ Failed to calibrate {camera_name}")
            return False
    
    def stereo_calibrate_pair_from_images(self, calib_img1, calib_img2, cam1_name, cam2_name,
                                          chessboard_size=(9, 6), square_size=1.0):
        """
        Perform stereo calibration between two cameras from synchronized images
        
        Args:
            calib_img1, calib_img2: Paths to synchronized calibration images (or lists of paths)
            cam1_name, cam2_name: Camera names
            chessboard_size: Tuple (cols, rows) of inner corners
            square_size: Size of chessboard squares in real world units
        """
        print(f"\n=== Stereo Calibration: {cam1_name} <-> {cam2_name} ===")
        
        if cam1_name not in self.camera_matrices or cam2_name not in self.camera_matrices:
            print("Error: Cameras must be individually calibrated first!")
            return False
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        objpoints = []
        imgpoints1 = []
        imgpoints2 = []
        
        # Handle both single image path and list of paths
        if isinstance(calib_img1, str):
            calib_img1 = [calib_img1]
        if isinstance(calib_img2, str):
            calib_img2 = [calib_img2]
        
        # Ensure we have matching pairs
        num_pairs = min(len(calib_img1), len(calib_img2))
        if num_pairs == 0:
            print("Error: No calibration images provided")
            return False
        
        print(f"Processing {num_pairs} synchronized calibration image pair(s)...")
        for i in range(num_pairs):
            frame1 = cv2.imread(calib_img1[i])
            frame2 = cv2.imread(calib_img2[i])
            
            if frame1 is None or frame2 is None:
                print(f"Warning: Could not read image pair {i+1}, skipping...")
                continue
            
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            ret_corners1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)
            ret_corners2, corners2 = cv2.findChessboardCorners(gray2, chessboard_size, None)
            
            # Only use frames where both cameras see the pattern
            if ret_corners1 and ret_corners2:
                objpoints.append(objp.copy())
                
                corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
                
                imgpoints1.append(corners1)
                imgpoints2.append(corners2)
                print(f"✓ Found synchronized pattern pair {i+1} (Total: {len(objpoints)})")
            else:
                print(f"✗ Could not find synchronized pattern in pair {i+1}")
        
        print(f"\nFound {len(objpoints)} synchronized calibration pairs")
        
        if len(objpoints) < 1:
            print(f"Error: Need at least 1 synchronized pair, found {len(objpoints)}")
            return False
        
        # Get camera matrices and distortion coefficients
        K1 = self.camera_matrices[cam1_name]
        D1 = self.distortion_coeffs[cam1_name]
        K2 = self.camera_matrices[cam2_name]
        D2 = self.distortion_coeffs[cam2_name]
        
        # Perform stereo calibration
        print("Performing stereo calibration...")
        ret, K1_new, D1_new, K2_new, D2_new, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints1, imgpoints2,
            K1, D1, K2, D2, self.image_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            flags=cv2.CALIB_FIX_INTRINSIC  # Use pre-calibrated intrinsics
        )
        
        if ret:
            self.stereo_pairs[(cam1_name, cam2_name)] = {
                'R': R,  # Rotation matrix from cam1 to cam2
                'T': T,  # Translation vector from cam1 to cam2
                'E': E,  # Essential matrix
                'F': F   # Fundamental matrix
            }
            
            print(f"✓ Stereo calibration successful!")
            print(f"  Rotation matrix:\n{R}")
            print(f"  Translation vector:\n{T}")
            return True
        else:
            print(f"✗ Stereo calibration failed")
            return False
    
    def load_frame(self, video_path, frame_number):
        """Load a specific frame from a video"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        return ret, frame
    
    def undistort_frame(self, frame, camera_name):
        """Undistort a frame using camera calibration parameters"""
        if camera_name not in self.camera_matrices:
            return frame
        
        K = self.camera_matrices[camera_name]
        D = self.distortion_coeffs[camera_name]
        
        h, w = frame.shape[:2]
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, K, D, None, new_K)
        
        return undistorted
    
    def compute_stereo_disparity(self, img1, img2, cam1_name, cam2_name):
        """
        Compute disparity map using stereo block matching
        """
        if (cam1_name, cam2_name) not in self.stereo_pairs:
            print(f"Error: No stereo calibration for {cam1_name} and {cam2_name}")
            return None, None
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        print(f"  Image sizes: {gray1.shape} and {gray2.shape}")
        
        # Get stereo calibration parameters
        stereo_info = self.stereo_pairs[(cam1_name, cam2_name)]
        K1 = self.camera_matrices[cam1_name]
        K2 = self.camera_matrices[cam2_name]
        R = stereo_info['R']
        T = stereo_info['T']
        
        print(f"  Camera matrices shapes: {K1.shape}, {K2.shape}")
        print(f"  Translation: {T.flatten()}")
        
        # Rectify stereo pair
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, self.distortion_coeffs[cam1_name],
            K2, self.distortion_coeffs[cam2_name],
            self.image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.9
        )
        
        print(f"  Stereo rectification complete")
        print(f"  ROIs: {roi1}, {roi2}")
        
        # Compute rectification maps
        map1x, map1y = cv2.initUndistortRectifyMap(
            K1, self.distortion_coeffs[cam1_name], R1, P1, self.image_size, cv2.CV_32FC1
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            K2, self.distortion_coeffs[cam2_name], R2, P2, self.image_size, cv2.CV_32FC1
        )
        
        # Rectify images
        rectified1 = cv2.remap(gray1, map1x, map1y, cv2.INTER_LINEAR)
        rectified2 = cv2.remap(gray2, map2x, map2y, cv2.INTER_LINEAR)
        
        print(f"  Rectified image sizes: {rectified1.shape}, {rectified2.shape}")
        
        # Create stereo matcher with better parameters
        # numDisparities must be divisible by 16
        width = rectified1.shape[1]
        num_disparities = min(((width // 8) + 15) & -16, 256)  # Adaptive, max 256
        num_disparities = max(num_disparities, 16)  # At least 16
        
        print(f"  Using numDisparities={num_disparities}")
        
        stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=15)
        # Set additional parameters for better results
        stereo.setPreFilterType(1)
        stereo.setPreFilterSize(5)
        stereo.setPreFilterCap(31)
        stereo.setTextureThreshold(10)
        stereo.setUniquenessRatio(15)
        stereo.setSpeckleWindowSize(100)
        stereo.setSpeckleRange(32)
        stereo.setDisp12MaxDiff(1)
        
        disparity = stereo.compute(rectified1, rectified2).astype(np.float32)
        
        # Check disparity statistics
        valid_mask = disparity > 0
        num_valid = np.sum(valid_mask)
        total_pixels = disparity.size
        
        print(f"  Disparity map statistics:")
        print(f"    Shape: {disparity.shape}")
        print(f"    Valid pixels: {num_valid}/{total_pixels} ({100*num_valid/total_pixels:.1f}%)")
        if num_valid > 0:
            print(f"    Min disparity: {disparity[valid_mask].min():.2f}")
            print(f"    Max disparity: {disparity[valid_mask].max():.2f}")
            print(f"    Mean disparity: {disparity[valid_mask].mean():.2f}")
        else:
            print(f"    WARNING: No valid disparity values found!")
        
        # Convert from fixed-point to float (divide by 16)
        disparity_float = disparity / 16.0
        
        # Save rectified images for debugging
        output_dir = '/Users/olivia/Documents/COMS4731/COMS4731-final-project/depth_results'
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f'rectified_{cam1_name}.png'), rectified1)
        cv2.imwrite(os.path.join(output_dir, f'rectified_{cam2_name}.png'), rectified2)
        
        # Save disparity visualization
        if num_valid > 0:
            disp_vis = cv2.normalize(disparity_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            disp_colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(output_dir, f'disparity_{cam1_name}_{cam2_name}_raw.png'), disp_colored)
            print(f"  Saved disparity visualization")
        
        return disparity_float, Q
    
    def disparity_to_depth(self, disparity, Q):
        """
        Convert disparity map to depth map using Q matrix
        """
        if Q is None:
            print("  Error: Q matrix is None")
            return None
        
        if disparity is None:
            print("  Error: Disparity map is None")
            return None
        
        print(f"  Converting disparity to depth...")
        print(f"  Disparity shape: {disparity.shape}, dtype: {disparity.dtype}")
        print(f"  Q matrix:\n{Q}")
        
        # Check for valid disparity values
        valid_mask = disparity > 0
        num_valid = np.sum(valid_mask)
        
        if num_valid == 0:
            print("  WARNING: No valid disparity values to convert!")
            return np.zeros_like(disparity)
        
        print(f"  Valid disparity pixels: {num_valid}/{disparity.size} ({100*num_valid/disparity.size:.1f}%)")
        
        # Convert disparity to depth using Q matrix
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        depth = points_3d[:, :, 2]  # Z coordinate is depth
        
        print(f"  Depth statistics (before filtering):")
        print(f"    Min: {depth[valid_mask].min():.2f}")
        print(f"    Max: {depth[valid_mask].max():.2f}")
        print(f"    Mean: {depth[valid_mask].mean():.2f}")
        print(f"    Median: {np.median(depth[valid_mask]):.2f}")
        
        # Filter invalid depths - be more permissive
        depth[depth < 0] = 0
        # Use a more reasonable max depth - try 5000 instead of 10000
        depth[depth > 5000] = 0
        
        # Also filter based on disparity validity
        depth[~valid_mask] = 0
        
        final_valid = np.sum(depth > 0)
        print(f"  Final valid depth pixels: {final_valid}/{depth.size} ({100*final_valid/depth.size:.1f}%)")
        
        if final_valid == 0:
            print("  WARNING: All depths filtered out! Trying without max depth limit...")
            # Try without max depth limit
            depth = points_3d[:, :, 2].copy()
            depth[depth < 0] = 0
            depth[~valid_mask] = 0
            final_valid = np.sum(depth > 0)
            if final_valid > 0:
                print(f"  Found {final_valid} valid depths without max limit")
        
        return depth
    
    def create_multi_view_depth(self, frames, camera_names, frame_number):
        """
        Create depth map from 3 camera views
        
        Args:
            frames: Dict of camera_name -> frame image
            camera_names: List of 3 camera names
            frame_number: Frame number for saving outputs
        """
        print(f"\n=== Creating Depth Map from 3 Views (Frame {frame_number}) ===")
        
        if len(camera_names) < 2:
            print("Error: Need at least 2 cameras for depth estimation")
            return None
        
        # Undistort all frames
        undistorted_frames = {}
        for cam_name in camera_names:
            if cam_name in frames:
                undistorted_frames[cam_name] = self.undistort_frame(frames[cam_name], cam_name)
                print(f"Undistorted frame from {cam_name}")
        
        # Compute disparity maps for each camera pair
        disparity_maps = {}
        depth_maps = {}
        
        # Use first two cameras as primary stereo pair
        cam1 = camera_names[0]
        cam2 = camera_names[1]
        
        if cam1 in undistorted_frames and cam2 in undistorted_frames:
            print(f"\nComputing disparity between {cam1} and {cam2}...")
            disparity, Q = self.compute_stereo_disparity(
                undistorted_frames[cam1], 
                undistorted_frames[cam2],
                cam1, cam2
            )
            
            if disparity is not None:
                disparity_maps[f"{cam1}_{cam2}"] = disparity
                depth = self.disparity_to_depth(disparity, Q)
                if depth is not None:
                    depth_maps[f"{cam1}_{cam2}"] = depth
                    print(f"✓ Created depth map from {cam1} and {cam2}")
        
        # If we have a third camera, try to use it with the first camera
        if len(camera_names) >= 3:
            cam3 = camera_names[2]
            if cam1 in undistorted_frames and cam3 in undistorted_frames:
                if (cam1, cam3) in self.stereo_pairs:
                    print(f"\nComputing disparity between {cam1} and {cam3}...")
                    disparity, Q = self.compute_stereo_disparity(
                        undistorted_frames[cam1],
                        undistorted_frames[cam3],
                        cam1, cam3
                    )
                    
                    if disparity is not None:
                        disparity_maps[f"{cam1}_{cam3}"] = disparity
                        depth = self.disparity_to_depth(disparity, Q)
                        if depth is not None:
                            depth_maps[f"{cam1}_{cam3}"] = depth
                            print(f"✓ Created depth map from {cam1} and {cam3}")
        
        # Combine depth maps if multiple available
        if len(depth_maps) > 0:
            # Use the first depth map, or average if multiple
            combined_depth = None
            if len(depth_maps) == 1:
                combined_depth = list(depth_maps.values())[0]
            else:
                # Average valid depth values
                valid_depths = [d for d in depth_maps.values() if d is not None]
                if valid_depths:
                    combined_depth = np.mean(valid_depths, axis=0)
            
            return combined_depth, disparity_maps, undistorted_frames
        
        return None, None, undistorted_frames
    
    def visualize_depth_map(self, depth_map, original_frame, save_path=None):
        """
        Visualize depth map alongside original image
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Depth map
        depth_display = depth_map.copy()
        depth_display[depth_display == 0] = np.nan  # Hide invalid depths
        
        im = axes[1].imshow(depth_display, cmap='jet', vmin=0, vmax=np.nanpercentile(depth_display, 95))
        axes[1].set_title('Depth Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], label='Depth (mm)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Depth visualization saved to {save_path}")
        
        plt.show()
    
    def save_depth_map(self, depth_map, save_path):
        """Save depth map as image and numpy array"""
        # Normalize for visualization
        depth_normalized = depth_map.copy()
        depth_normalized[depth_normalized == 0] = 0
        if depth_normalized.max() > 0:
            depth_normalized = (depth_normalized / depth_normalized.max() * 255).astype(np.uint8)
        else:
            depth_normalized = depth_normalized.astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        cv2.imwrite(save_path, depth_colored)
        
        # Also save as numpy array
        np.save(save_path.replace('.png', '.npy'), depth_map)
        print(f"Depth map saved to {save_path}")
    
    def select_points_interactive(self, image, window_name="Select Points", max_points=None):
        """
        Interactively select points on an image by clicking
        
        Args:
            image: Image to select points on
            window_name: Name of the window
            max_points: Maximum number of points to select (None for unlimited)
        
        Returns:
            List of (x, y) point coordinates
        """
        points = []
        display_image = image.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if max_points is None or len(points) < max_points:
                    points.append((x, y))
                    cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
                    cv2.circle(display_image, (x, y), 10, (0, 255, 0), 2)
                    cv2.putText(display_image, str(len(points)), (x+10, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow(window_name, display_image)
                    print(f"Selected point {len(points)}: ({x}, {y})")
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)
        cv2.imshow(window_name, display_image)
        
        instructions = "Click to select points. Press 'q' or ESC to finish."
        print(instructions)
        print(f"Select at least 4 points for homography computation.")
        if max_points:
            print(f"Maximum {max_points} points.")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            if max_points and len(points) >= max_points:
                print(f"Maximum points reached ({max_points}). Press 'q' to finish.")
        
        cv2.destroyWindow(window_name)
        print(f"Selected {len(points)} points total.")
        return points
    
    def compute_homography_ransac(self, points1, points2, 
                                   ransac_threshold=3.0, 
                                   confidence=0.995,
                                   max_iters=2000):
        """
        Compute homography matrix using RANSAC
        
        Args:
            points1: List of (x, y) points in first image
            points2: List of (x, y) points in second image (corresponding points)
            ransac_threshold: Maximum reprojection error for inliers
            confidence: Confidence level (0-1)
            max_iters: Maximum RANSAC iterations
        
        Returns:
            homography: 3x3 homography matrix
            mask: Boolean mask indicating inliers
            inlier_count: Number of inliers
        """
        if len(points1) < 4 or len(points2) < 4:
            print("Error: Need at least 4 point correspondences for homography")
            return None, None, 0
        
        if len(points1) != len(points2):
            print("Error: Number of points must match between images")
            return None, None, 0
        
        pts1 = np.array(points1, dtype=np.float32)
        pts2 = np.array(points2, dtype=np.float32)
        
        # Compute homography using RANSAC
        homography, mask = cv2.findHomography(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            confidence=confidence,
            maxIters=max_iters
        )
        
        if homography is None:
            print("Failed to compute homography")
            return None, None, 0
        
        inlier_count = np.sum(mask)
        inlier_ratio = inlier_count / len(points1)
        
        print(f"Homography computed using RANSAC:")
        print(f"  Total correspondences: {len(points1)}")
        print(f"  Inliers: {inlier_count} ({inlier_ratio*100:.1f}%)")
        print(f"  Outliers: {len(points1) - inlier_count}")
        print(f"\nHomography matrix:\n{homography}")
        
        return homography, mask, inlier_count
    
    def visualize_correspondences(self, img1, img2, points1, points2, 
                                  mask=None, save_path=None):
        """
        Visualize point correspondences between two images
        
        Args:
            img1, img2: Images to visualize
            points1, points2: Corresponding point lists
            mask: Optional boolean mask to show inliers/outliers
            save_path: Optional path to save visualization
        """
        # Create side-by-side visualization
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Resize images to same height
        if h1 != h2:
            scale = min(h1, h2) / max(h1, h2)
            if h1 > h2:
                img1 = cv2.resize(img1, (int(w1 * scale), int(h1 * scale)))
            else:
                img2 = cv2.resize(img2, (int(w2 * scale), int(h2 * scale)))
        
        h, w = img1.shape[:2]
        combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
        combined[:, :w] = img1
        combined[:, w:] = img2
        
        # Draw points and correspondences
        for i, ((x1, y1), (x2, y2)) in enumerate(zip(points1, points2)):
            # Adjust x2 for right image
            x2_adjusted = x2 + w
            
            # Choose color based on mask (if provided)
            if mask is not None:
                color = (0, 255, 0) if mask[i] else (0, 0, 255)  # Green for inliers, red for outliers
            else:
                color = (0, 255, 255)  # Yellow for all points
            
            # Draw points
            cv2.circle(combined, (int(x1), int(y1)), 5, color, -1)
            cv2.circle(combined, (int(x2_adjusted), int(y2)), 5, color, -1)
            
            # Draw line connecting correspondences
            cv2.line(combined, (int(x1), int(y1)), (int(x2_adjusted), int(y2)), color, 2)
            
            # Draw point number
            cv2.putText(combined, str(i+1), (int(x1)+10, int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(combined, str(i+1), (int(x2_adjusted)+10, int(y2)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if save_path:
            cv2.imwrite(save_path, combined)
            print(f"Correspondence visualization saved to {save_path}")
        
        cv2.imshow("Point Correspondences (Left: Image 1, Right: Image 2)", combined)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return combined
    
    def apply_homography(self, image, homography, output_size=None):
        """
        Apply homography transformation to warp an image
        
        Args:
            image: Input image
            homography: 3x3 homography matrix
            output_size: (width, height) of output image
        
        Returns:
            Warped image
        """
        if output_size is None:
            h, w = image.shape[:2]
            output_size = (w, h)
        
        warped = cv2.warpPerspective(image, homography, output_size)
        return warped
    
    def compute_homography_from_sift(self, img1, img2, 
                                     max_matches=2000,  # Increased default
                                     ransac_threshold=3.0,
                                     save_visualization_path=None):
        """
        Compute homography using SIFT features with RANSAC
        
        Args:
            img1, img2: Input images (can be numpy arrays or paths)
            max_matches: Maximum number of matches to use
            ransac_threshold: RANSAC reprojection threshold
            save_visualization_path: Optional path to save feature matching visualization
        
        Returns:
            homography: 3x3 homography matrix
            matches_mask: Inlier mask from RANSAC
            num_matches: Number of feature matches found
        """
        print("\n=== Computing Homography from SIFT Features ===")
        
        # Load images if paths provided
        if isinstance(img1, str):
            img1 = cv2.imread(img1)
        if isinstance(img2, str):
            img2 = cv2.imread(img2)
        
        if img1 is None or img2 is None:
            print("Error: Could not load images")
            return None, None, 0
        
        print(f"Image 1: {img1.shape[1]}x{img1.shape[0]}")
        print(f"Image 2: {img2.shape[1]}x{img2.shape[0]}")
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # Create SIFT detector with many more features
        print("Detecting SIFT features...")
        # Increase significantly: use 0 for unlimited features, or a very high number
        sift = cv2.SIFT_create(
            nfeatures=0,  # 0 = unlimited features
            nOctaveLayers=5,  # More octave layers for more scale invariance
            contrastThreshold=0.03,  # Lower threshold to detect more features
            edgeThreshold=10,  # Lower edge threshold
            sigma=1.6
        )
        
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        print(f"Found {len(kp1)} keypoints in image 1")
        print(f"Found {len(kp2)} keypoints in image 2")
        
        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            print("Error: Could not find features in one or both images")
            return None, None, 0
        
        # Match features using FLANN or BFMatcher
        print("Matching features...")
        
        # Use FLANN matcher for faster matching with large feature sets
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test to filter good matches (relaxed threshold)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # More relaxed ratio (0.75 instead of 0.7) to get more matches
                if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                    good_matches.append(m)
        
        print(f"Found {len(good_matches)} good matches after ratio test")
        
        if len(good_matches) < 4:
            print(f"Error: Need at least 4 matches for homography, found {len(good_matches)}")
            return None, None, len(good_matches)
        
        # Limit to max_matches best matches (sorted by quality)
        if len(good_matches) > max_matches:
            good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_matches]
            print(f"Using top {max_matches} best matches (sorted by distance)")
        else:
            print(f"Using all {len(good_matches)} matches")
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        print(f"Computing homography using RANSAC with {len(pts1)} matches...")
        
        # Compute homography using RANSAC
        homography, matches_mask = cv2.findHomography(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            confidence=0.995,
            maxIters=2000
        )
        
        if homography is None:
            print("Failed to compute homography")
            return None, None, len(good_matches)
        
        # Count inliers
        if matches_mask is not None:
            num_inliers = np.sum(matches_mask)
            inlier_ratio = num_inliers / len(good_matches)
            print(f"Homography computed successfully!")
            print(f"  Total matches: {len(good_matches)}")
            print(f"  Inliers: {num_inliers} ({inlier_ratio*100:.1f}%)")
            print(f"  Outliers: {len(good_matches) - num_inliers}")
            print(f"\nHomography matrix:\n{homography}")
        else:
            print("Homography computed but no mask returned")
            matches_mask = np.ones(len(good_matches), dtype=np.uint8)
        
        # Visualize matches
        if save_visualization_path:
            self._visualize_sift_matches(img1, img2, kp1, kp2, good_matches, 
                                       matches_mask, save_visualization_path)
        
        return homography, matches_mask, len(good_matches)
    
    def _visualize_sift_matches(self, img1, img2, kp1, kp2, matches, matches_mask, save_path):
        """Visualize SIFT feature matches"""
        # Draw matches
        draw_params = dict(
            matchColor=(0, 255, 0),  # Green for inliers
            singlePointColor=None,
            matchesMask=matches_mask.ravel().tolist(),
            flags=cv2.DrawMatchesFlags_DEFAULT
        )
        
        # Also draw outliers in red if mask available
        if matches_mask is not None:
            outlier_mask = (~matches_mask.astype(bool)).astype(np.uint8)
            if np.sum(outlier_mask) > 0:
                outlier_params = dict(
                    matchColor=(0, 0, 255),  # Red for outliers
                    singlePointColor=None,
                    matchesMask=outlier_mask.ravel().tolist(),
                    flags=cv2.DrawMatchesFlags_DEFAULT
                )
                img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
                img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, img_matches, **outlier_params)
            else:
                img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
        else:
            img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
        
        cv2.imwrite(save_path, img_matches)
        print(f"SIFT matches visualization saved to {save_path}")
    
    def compute_depth_from_homography(self, img1, img2, homography, 
                                     cam1_name, cam2_name):
        """
        Compute depth map using homography-warped images
        
        Args:
            img1, img2: Input images
            homography: Homography matrix from img1 to img2
            cam1_name, cam2_name: Camera names for reference
        
        Returns:
            depth_map: Computed depth map
        """
        print(f"\n=== Computing Depth from Homography ===")
        
        if homography is None:
            print("Error: Homography is None")
            return None
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # Warp image 2 to align with image 1
        print("Warping image 2 to align with image 1...")
        h, w = gray1.shape
        gray2_warped = cv2.warpPerspective(gray2, np.linalg.inv(homography), (w, h))
        
        # Now compute disparity on aligned images using SGBM (much better than StereoBM)
        print("Computing disparity using Semi-Global Block Matching (SGBM)...")
        
        # Calculate optimal number of disparities (must be divisible by 16)
        width = w
        num_disparities = ((width // 4) + 15) & -16  # More disparities for better results
        num_disparities = max(num_disparities, 64)  # At least 64
        num_disparities = min(num_disparities, 256)  # Max 256 for performance
        
        block_size = 5  # Smaller block size for finer details
        
        print(f"Using numDisparities={num_disparities}, blockSize={block_size}")
        
        # Create SGBM matcher (much better than StereoBM)
        # Using SGBM mode (not 3WAY) for better quality
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,  # Penalty parameters for SGBM
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            preFilterCap=63,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM  # Full SGBM for best quality (slower but better)
        )
        
        # Create right matcher for left-right consistency check
        try:
            stereo_right = cv2.StereoSGBM_create(
                minDisparity=-num_disparities,
                numDisparities=num_disparities,
                blockSize=block_size,
                P1=8 * 3 * block_size ** 2,
                P2=32 * 3 * block_size ** 2,
                disp12MaxDiff=1,
                preFilterCap=63,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                mode=cv2.STEREO_SGBM_MODE_SGBM
            )
            
            # Compute left-right disparity and filter inconsistent matches
            print("Computing left-right consistency check...")
            disparity_left = stereo.compute(gray1, gray2_warped)
            disparity_right = stereo_right.compute(gray2_warped, gray1)
            
            # Convert to float
            disparity_left_f = disparity_left.astype(np.float32) / 16.0
            disparity_right_f = disparity_right.astype(np.float32) / 16.0
            
            # Filter based on left-right consistency (vectorized for speed)
            w = disparity_left_f.shape[1]
            h = disparity_left_f.shape[0]
            
            # Create coordinate grids
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            
            # Calculate corresponding x coordinates in right image
            x_right_coords = (x_coords - disparity_left_f).astype(np.int32)
            
            # Create mask for valid coordinates
            valid_x_right = (x_right_coords >= 0) & (x_right_coords < w)
            valid_disparity = disparity_left_f > 0
            
            # Initialize consistent mask
            consistent_mask = np.zeros_like(disparity_left_f, dtype=bool)
            
            # Check consistency where valid
            valid_mask = valid_disparity & valid_x_right
            consistent_mask[valid_mask] = (
                np.abs(disparity_left_f[valid_mask] - 
                      disparity_right_f[y_coords[valid_mask], x_right_coords[valid_mask]]) <= 1.0
            )
            
            disparity = disparity_left_f.copy()
            disparity[~consistent_mask] = 0
            print(f"  Left-right consistency: {np.sum(consistent_mask)}/{np.sum(disparity_left_f > 0)} consistent pixels")
        except Exception as e:
            # Fallback to single-direction matching
            print(f"Using single-direction matching (left-right check not available: {e})")
            disparity = stereo.compute(gray1, gray2_warped).astype(np.float32) / 16.0
        
        # Refine disparity map using weighted median filter (removes noise)
        # This is optional - only if opencv-contrib is available
        try:
            print("Refining disparity map with weighted median filter...")
            disparity_refined = cv2.ximgproc.weightedMedianFilter(
                gray1.astype(np.uint8), 
                disparity.astype(np.float32), 
                5  # Filter size
            )
            disparity = disparity_refined
            print("  ✓ Applied weighted median filter")
        except (AttributeError, cv2.error):
            print("  ⚠ Weighted median filter not available (opencv-contrib required), using basic filtering")
            # Basic median filter as fallback
            disparity_uint8 = np.clip(disparity, 0, 255).astype(np.uint8)
            disparity_filtered = cv2.medianBlur(disparity_uint8, 5)
            disparity = disparity_filtered.astype(np.float32)
        
        # Additional post-processing: filter out small speckles
        print("Post-processing disparity map...")
        # Remove small isolated regions
        kernel = np.ones((3, 3), np.uint8)
        valid_mask_binary = (disparity > 0).astype(np.uint8) * 255
        valid_mask_cleaned = cv2.morphologyEx(valid_mask_binary, cv2.MORPH_CLOSE, kernel)
        valid_mask_cleaned = cv2.morphologyEx(valid_mask_cleaned, cv2.MORPH_OPEN, kernel)
        disparity[valid_mask_cleaned == 0] = 0
        
        # Statistics
        valid_mask = disparity > 0
        num_valid = np.sum(valid_mask)
        print(f"Disparity map statistics:")
        print(f"  Valid pixels: {num_valid}/{disparity.size} ({100*num_valid/disparity.size:.1f}%)")
        if num_valid > 0:
            print(f"  Min: {disparity[valid_mask].min():.2f}")
            print(f"  Max: {disparity[valid_mask].max():.2f}")
            print(f"  Mean: {disparity[valid_mask].mean():.2f}")
        
        # Estimate depth from disparity using calibrated camera parameters if available
        if cam1_name in self.camera_matrices and cam2_name in self.camera_matrices:
            # Use calibrated camera parameters
            K1 = self.camera_matrices[cam1_name]
            K2 = self.camera_matrices[cam2_name]
            
            # Estimate baseline from stereo calibration if available
            if (cam1_name, cam2_name) in self.stereo_pairs:
                T = self.stereo_pairs[(cam1_name, cam2_name)]['T']
                baseline = np.linalg.norm(T)  # Baseline in mm (if square_size is in mm)
            else:
                baseline = 100.0  # Fallback estimate
            
            # Get focal length from camera matrix
            focal_length = K1[0, 0]  # fx
            
            print(f"Using calibrated parameters: baseline={baseline:.2f}mm, focal_length={focal_length:.2f}px")
        else:
            # Fallback estimates
            baseline = 100.0
            focal_length = 1000.0
            print(f"Using estimated parameters: baseline={baseline:.2f}mm, focal_length={focal_length:.2f}px")
        
        depth_map = np.zeros_like(disparity)
        # Depth = (baseline * focal_length) / disparity
        # Avoid division by zero
        disparity_safe = np.maximum(disparity, 0.1)  # Minimum disparity of 0.1
        depth_map[valid_mask] = (baseline * focal_length) / disparity_safe[valid_mask]
        
        # Filter unreasonable depths with more intelligent bounds
        depth_map[depth_map < 0] = 0
        
        # Use percentile-based filtering instead of fixed threshold
        if num_valid > 100:
            valid_depths = depth_map[valid_mask]
            percentile_95 = np.percentile(valid_depths, 95)
            percentile_5 = np.percentile(valid_depths, 5)
            
            # Filter outliers: keep depths within 5th-95th percentile range
            reasonable_min = max(percentile_5 * 0.5, 100)  # At least 100mm
            reasonable_max = min(percentile_95 * 2.0, 20000)  # Max 20m
            
            depth_map[depth_map > reasonable_max] = 0
            depth_map[depth_map < reasonable_min] = 0
            
            print(f"Depth filtering: keeping values between {reasonable_min:.2f}mm and {reasonable_max:.2f}mm")
        else:
            # Fallback to fixed threshold if not enough samples
            depth_map[depth_map > 10000] = 0
        
        # Fill small holes using inpainting (for visualization)
        print("Filling small holes in depth map...")
        depth_mask = (depth_map > 0).astype(np.uint8) * 255
        
        # Find small holes (regions smaller than threshold)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            depth_mask, connectivity=8
        )
        
        # Fill small holes (less than 0.1% of image area)
        min_area = depth_map.size * 0.001
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < min_area:
                # Fill this small hole with nearest valid depth
                hole_mask = (labels == label_id).astype(np.uint8)
                # Use nearest neighbor interpolation from valid depths
                depth_filled = cv2.inpaint(
                    depth_map.astype(np.float32),
                    hole_mask,
                    3,
                    cv2.INPAINT_NS
                )
                depth_map = depth_filled
        
        final_valid = np.sum(depth_map > 0)
        print(f"Final depth map: {final_valid}/{depth_map.size} valid pixels ({100*final_valid/depth_map.size:.1f}%)")
        
        # Save intermediate results
        output_dir = '/Users/olivia/Documents/COMS4731/COMS4731-final-project/depth_results'
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f'warped_{cam2_name}.png'), gray2_warped)
        
        if num_valid > 0:
            disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            disp_colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(output_dir, f'disparity_homography_{cam1_name}_{cam2_name}.png'), disp_colored)
        
        return depth_map
    
    def select_points_for_homography(self, img1_path, img2_path, 
                                     save_correspondences_path=None):
        """
        Interactive workflow to select corresponding points and compute homography
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            save_correspondences_path: Optional path to save correspondence visualization
        
        Returns:
            homography: 3x3 homography matrix
            points1: Selected points from image 1
            points2: Selected points from image 2
            mask: Inlier mask from RANSAC
        """
        print("\n" + "="*60)
        print("Interactive Homography Computation")
        print("="*60)
        
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"Error: Could not load images")
            print(f"  Image 1: {img1_path}")
            print(f"  Image 2: {img2_path}")
            return None, None, None, None
        
        print(f"\nImage 1: {os.path.basename(img1_path)} ({img1.shape[1]}x{img1.shape[0]})")
        print(f"Image 2: {os.path.basename(img2_path)} ({img2.shape[1]}x{img2.shape[0]})")
        
        # Select points on first image
        print("\n--- Select points on Image 1 ---")
        points1 = self.select_points_interactive(img1, "Image 1 - Select Corresponding Points")
        
        if len(points1) < 4:
            print("Error: Need at least 4 points for homography")
            return None, None, None, None
        
        # Select corresponding points on second image
        print("\n--- Select corresponding points on Image 2 (in same order) ---")
        points2 = self.select_points_interactive(img2, "Image 2 - Select Corresponding Points", 
                                                 max_points=len(points1))
        
        if len(points2) != len(points1):
            print(f"Error: Number of points doesn't match ({len(points1)} vs {len(points2)})")
            return None, None, None, None
        
        # Visualize correspondences
        print("\n--- Visualizing correspondences ---")
        viz = self.visualize_correspondences(img1, img2, points1, points2, 
                                            save_path=save_correspondences_path)
        
        # Compute homography with RANSAC
        print("\n--- Computing homography with RANSAC ---")
        homography, mask, inlier_count = self.compute_homography_ransac(points1, points2)
        
        if homography is not None and mask is not None:
            # Visualize with inlier/outlier distinction
            print("\n--- Visualizing inliers (green) and outliers (red) ---")
            self.visualize_correspondences(img1, img2, points1, points2, mask=mask)
        
        return homography, points1, points2, mask


def main():
    """
    Main function to calibrate cameras and create depth map.
    
    Provide paths to:
    1. Calibration images (one per camera showing checkerboard)
    2. Matching frames to reconstruct (one per camera from the same scene)
    """
    # Initialize reconstructor
    reconstructor = MultiViewDepthReconstructor()
    
    # ============================================================
    # STEP 1: Provide paths to calibration images (one per camera)
    # ============================================================
    calibration_images = {
        'left': '/Users/olivia/Documents/COMS4731/COMS4731-final-project/frames/left_calibration/left_camera_calibration_frame_01400.png',    # TODO: Set path to left camera calibration image
        'right': '/Users/olivia/Documents/COMS4731/COMS4731-final-project/frames/right_calibration/right_camera_calibration_frame_00000.png',   # TODO: Set path to right camera calibration image  
        'tripod': '/Users/olivia/Documents/COMS4731/COMS4731-final-project/frames/tripod_calibration/tripod_camera_calibration_frame_00600.png'  # TODO: Set path to tripod camera calibration image
    }
    
    # ============================================================
    # STEP 2: Provide paths to matching frames to reconstruct
    # ============================================================
    reconstruction_frames = {
        'left': '/Users/olivia/Documents/COMS4731/COMS4731-final-project/frames/reconstruction_frames/arabesque_left_frame_26s.png',
        'right': '/Users/olivia/Documents/COMS4731/COMS4731-final-project/frames/reconstruction_frames/barre_right_frame_35s.png',
        'tripod': '/Users/olivia/Documents/COMS4731/COMS4731-final-project/frames/reconstruction_frames/barre_tripod_frame_39s.png'
    }
    
    # ============================================================
    # Configuration: Adjust based on your checkerboard
    # ============================================================
    chessboard_size = (9, 6)  # (cols, rows) - inner corners of checkerboard
    square_size = 1.0         # Size of each square in real world units (e.g., mm)
    
    camera_names = ['left', 'right', 'tripod']
    
    # Check if paths are provided
    missing_calib = [cam for cam, path in calibration_images.items() if path is None]
    missing_frames = [cam for cam, path in reconstruction_frames.items() if path is None]
    
    if missing_calib:
        print("ERROR: Please provide calibration image paths:")
        for cam in missing_calib:
            print(f"  calibration_images['{cam}'] = 'path/to/image'")
        return
    
    if missing_frames:
        print("ERROR: Please provide reconstruction frame paths:")
        for cam in missing_frames:
            print(f"  reconstruction_frames['{cam}'] = 'path/to/image'")
        return
    
    # Step 1: Calibrate each camera individually
    print("\n" + "="*60)
    print("STEP 1: Individual Camera Calibration")
    print("="*60)
    
    for cam_name in camera_names:
        if calibration_images[cam_name]:
            reconstructor.calibrate_single_camera_from_images(
                calibration_images[cam_name],
                cam_name,
                chessboard_size=chessboard_size,
                square_size=square_size
            )
    
    # Step 2: Stereo calibration between camera pairs
    print("\n" + "="*60)
    print("STEP 2: Stereo Calibration")
    print("="*60)
    
    # Calibrate left-right pair (using same calibration images - they should show checkerboard)
    reconstructor.stereo_calibrate_pair_from_images(
        calibration_images['left'],
        calibration_images['right'],
        'left', 'right',
        chessboard_size=chessboard_size,
        square_size=square_size
    )
    
    # Calibrate left-tripod pair
    reconstructor.stereo_calibrate_pair_from_images(
        calibration_images['left'],
        calibration_images['tripod'],
        'left', 'tripod',
        chessboard_size=chessboard_size,
        square_size=square_size
    )
    
    # Step 3: Load frames to reconstruct
    print("\n" + "="*60)
    print("STEP 3: Loading Frames to Reconstruct")
    print("="*60)
    
    frames = {}
    for cam_name in camera_names:
        if reconstruction_frames[cam_name]:
            frame = cv2.imread(reconstruction_frames[cam_name])
            if frame is not None:
                frames[cam_name] = frame
                print(f"✓ Loaded frame from {cam_name}: {frame.shape}")
            else:
                print(f"✗ Could not load frame from {cam_name}")
    
    # Step 4: Create depth map using SIFT + Homography + RANSAC
    print("\n" + "="*60)
    print("STEP 4: Creating 3D Depth Map using SIFT + Homography + RANSAC")
    print("="*60)
    
    # Undistort frames
    undistorted_frames = {}
    for cam_name in camera_names:
        if cam_name in frames:
            undistorted_frames[cam_name] = reconstructor.undistort_frame(frames[cam_name], cam_name)
    
    # Use first two cameras
    cam1 = camera_names[0]
    cam2 = camera_names[1]
    
    if cam1 in undistorted_frames and cam2 in undistorted_frames:
        img1 = undistorted_frames[cam1]
        img2 = undistorted_frames[cam2]
        
        # Compute homography from SIFT features with RANSAC
        output_dir = '/Users/olivia/Documents/COMS4731/COMS4731-final-project/depth_results'
        os.makedirs(output_dir, exist_ok=True)
        
        sift_vis_path = os.path.join(output_dir, f'sift_matches_{cam1}_{cam2}.png')
        homography, matches_mask, num_matches = reconstructor.compute_homography_from_sift(
            img1, img2,
            max_matches=2000,  # Increased from 500 to get many more matches
            ransac_threshold=3.0,
            save_visualization_path=sift_vis_path
        )
        
        if homography is not None:
            # Compute depth from homography-warped images
            depth_map = reconstructor.compute_depth_from_homography(
                img1, img2, homography, cam1, cam2
            )
            
            if depth_map is not None and np.sum(depth_map > 0) > 0:
                # Save results
                depth_path = os.path.join(output_dir, 'depth_map.png')
                reconstructor.save_depth_map(depth_map, depth_path)
                
                # Visualize depth map
                vis_path = os.path.join(output_dir, 'depth_visualization.png')
                reconstructor.visualize_depth_map(depth_map, img1, vis_path)
                
                valid_depths = depth_map[depth_map > 0]
                print(f"\n✓ Successfully created 3D depth map!")
                print(f"  Valid depth pixels: {len(valid_depths)}/{depth_map.size} ({100*len(valid_depths)/depth_map.size:.1f}%)")
                if len(valid_depths) > 0:
                    print(f"  Depth range: {valid_depths.min():.2f} - {valid_depths.max():.2f} mm")
                print(f"  Results saved to: {output_dir}")
            else:
                print("\n✗ Depth map is empty after homography-based computation")
                depth_map = None
        else:
            print("\n✗ Failed to compute homography from SIFT features")
            depth_map = None
    else:
        print("\n✗ Could not load frames for depth computation")
        depth_map = None
    
    # Initialize disparity_maps for fallback
    disparity_maps = {}
    
    # Fallback to original method if homography fails
    if depth_map is None or np.sum(depth_map > 0) == 0:
        print("\n" + "="*60)
        print("Falling back to traditional stereo method...")
        print("="*60)
        depth_map, disparity_maps, undistorted_frames = reconstructor.create_multi_view_depth(
            frames, camera_names, 0
        )
    
    if depth_map is not None and np.sum(depth_map > 0) > 0:
        # Save results
        output_dir = '/Users/olivia/Documents/COMS4731/COMS4731-final-project/depth_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save depth map (if not already saved)
        depth_path = os.path.join(output_dir, 'depth_map.png')
        if not os.path.exists(depth_path) or np.sum(depth_map > 0) > 0:
            reconstructor.save_depth_map(depth_map, depth_path)
        
        # Visualize depth map (if not already visualized)
        primary_cam = camera_names[0]
        if primary_cam in undistorted_frames:
            vis_path = os.path.join(output_dir, 'depth_visualization.png')
            if not os.path.exists(vis_path):
                reconstructor.visualize_depth_map(depth_map, undistorted_frames[primary_cam], vis_path)
        
        # Save disparity maps if available
        if disparity_maps:
            for pair_name, disparity in disparity_maps.items():
                # Normalize disparity for visualization
                disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                disp_colored = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
                disp_path = os.path.join(output_dir, f'disparity_{pair_name}.png')
                cv2.imwrite(disp_path, disp_colored)
                print(f"Saved disparity map: {disp_path}")
        
        valid_depths = depth_map[depth_map > 0]
        if len(valid_depths) > 0:
            print(f"\n✓ Successfully created 3D depth map!")
            print(f"  Depth range: {valid_depths.min():.2f} - {valid_depths.max():.2f}")
        print(f"  Results saved to: {output_dir}")
    else:
        print("\n✗ Failed to create depth map. Check calibration results.")


def homography_example():
    """
    Example function demonstrating how to use interactive homography computation
    """
    reconstructor = MultiViewDepthReconstructor()
    
    # Example: Compute homography between two images
    # Replace these paths with your actual image paths
    img1_path = '/path/to/first/image.png'
    img2_path = '/path/to/second/image.png'
    
    # Interactive workflow: select points and compute homography
    homography, points1, points2, mask = reconstructor.select_points_for_homography(
        img1_path, 
        img2_path,
        save_correspondences_path='correspondences_visualization.png'
    )
    
    if homography is not None:
        print("\n✓ Homography computed successfully!")
        print(f"  You can now use this homography for image warping or stereo calibration")
        
        # Example: Warp image 2 to align with image 1
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is not None and img2 is not None:
            warped = reconstructor.apply_homography(img2, homography, 
                                                     output_size=(img1.shape[1], img1.shape[0]))
            
            # Save warped image
            cv2.imwrite('warped_image.png', warped)
            print(f"  Warped image saved to 'warped_image.png'")
            
            # Visualize
            combined = np.hstack([img1, warped])
            cv2.imshow("Original (left) vs Warped (right)", combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("\n✗ Failed to compute homography")


if __name__ == "__main__":
    import sys
    
    # Check if user wants to run homography example instead
    if len(sys.argv) > 1 and sys.argv[1] == '--homography':
        homography_example()
    else:
        main()

