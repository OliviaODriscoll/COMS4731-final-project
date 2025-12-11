import cv2
import numpy as np
import os
from scipy.optimize import least_squares
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Skeleton3DReconstructor:
    def __init__(self):
        self.cameras = {}
        self.camera_matrices = {}
        self.distortion_coeffs = {}
        self.rotation_matrices = {}
        self.translation_vectors = {}
        
    def calibrate_cameras(self, calibration_videos):
        """
        Calibrate cameras using calibration videos
        """
        print("=== CAMERA CALIBRATION ===")
        
        # Chessboard parameters (adjust based on your calibration pattern)
        chessboard_size = (9, 6)  # Adjust to match your calibration pattern
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
        for camera_name, video_path in calibration_videos.items():
            print(f"Calibrating {camera_name}...")
            
            cap = cv2.VideoCapture(video_path)
            objpoints = []  # 3D points in real world space
            imgpoints = []  # 2D points in image plane
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Sample every 30th frame for calibration
                if frame_count % 30 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Find chessboard corners
                    ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
                    
                    if ret_corners:
                        objpoints.append(objp)
                        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                        imgpoints.append(corners2)
                        
                        # Draw and display corners
                        cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret_corners)
                        cv2.imshow(f'{camera_name} Calibration', frame)
                        cv2.waitKey(1)
                
                frame_count += 1
            
            cap.release()
            cv2.destroyAllWindows()
            
            if len(objpoints) > 10:  # Need sufficient calibration images
                # Calibrate camera
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, gray.shape[::-1], None, None
                )
                
                if ret:
                    self.camera_matrices[camera_name] = mtx
                    self.distortion_coeffs[camera_name] = dist
                    print(f"{camera_name} calibration successful!")
                    print(f"Camera matrix shape: {mtx.shape}")
                    print(f"Distortion coefficients shape: {dist.shape}")
                else:
                    print(f"Failed to calibrate {camera_name}")
            else:
                print(f"Insufficient calibration data for {camera_name}")
    
    def stereo_calibrate(self, camera_pairs):
        """
        Perform stereo calibration between camera pairs
        """
        print("\n=== STEREO CALIBRATION ===")
        
        for pair_name, (cam1, cam2) in camera_pairs.items():
            print(f"Stereo calibrating {cam1} and {cam2}...")
            
            # This would require synchronized calibration videos
            # For now, we'll use a simplified approach
            # In practice, you'd need to capture synchronized calibration frames
            
            # Placeholder for stereo calibration
            # In a real implementation, you'd use cv2.stereoCalibrate()
            print(f"Stereo calibration for {pair_name} - placeholder")
    
    def extract_2d_skeleton(self, frame, camera_name):
        """
        Extract 2D skeleton from a single camera view
        This uses your existing skeletonization pipeline
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Yellow detection (shirt)
        lower_yellow1 = np.array([22, 80, 100])
        upper_yellow1 = np.array([28, 255, 255])
        lower_yellow2 = np.array([20, 60, 80])
        upper_yellow2 = np.array([30, 255, 255])
        
        yellow_mask1 = cv2.inRange(hsv, lower_yellow1, upper_yellow1)
        yellow_mask2 = cv2.inRange(hsv, lower_yellow2, upper_yellow2)
        yellow_mask = cv2.bitwise_or(yellow_mask1, yellow_mask2)
        
        # White detection (pants)
        lower_white1 = np.array([0, 0, 120])
        upper_white1 = np.array([180, 80, 255])
        lower_white2 = np.array([0, 0, 100])
        upper_white2 = np.array([180, 100, 255])
        
        white_mask1 = cv2.inRange(hsv, lower_white1, upper_white1)
        white_mask2 = cv2.inRange(hsv, lower_white2, upper_white2)
        white_mask_hsv = cv2.bitwise_or(white_mask1, white_mask2)
        
        # Grayscale thresholding
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, white_gray = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        
        # Combine white detection methods
        white_mask = cv2.bitwise_or(white_mask_hsv, white_gray)
        
        # Skin detection
        lower_skin1 = np.array([0, 30, 50])
        upper_skin1 = np.array([20, 150, 255])
        lower_skin2 = np.array([160, 30, 50])
        upper_skin2 = np.array([180, 150, 255])
        
        skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        
        # Combine all masks
        person_mask = cv2.bitwise_or(yellow_mask, white_mask)
        person_mask = cv2.bitwise_or(person_mask, skin_mask)
        
        # Clean up mask
        kernel = np.ones((5,5), np.uint8)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply bilateral filter
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        people_enhanced = cv2.bitwise_and(bilateral, bilateral, mask=person_mask)
        
        # Threshold
        _, thresh = cv2.threshold(people_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Preprocess for skeletonization
        kernel_fill = np.ones((5,5), np.uint8)
        thresh_filled = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_fill)
        kernel_noise = np.ones((3,3), np.uint8)
        thresh_clean = cv2.morphologyEx(thresh_filled, cv2.MORPH_OPEN, kernel_noise)
        
        # Distance transform for better skeleton
        dist_transform = cv2.distanceTransform(thresh_clean, cv2.DIST_L2, 5)
        dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
        _, dist_binary = cv2.threshold(dist_transform, 0.3, 1.0, cv2.THRESH_BINARY)
        dist_binary = (dist_binary * 255).astype(np.uint8)
        
        # Skeletonize
        skeleton = skeletonize(dist_binary > 0)
        skeleton_uint8 = img_as_ubyte(skeleton)
        
        # Clean up skeleton
        kernel_clean = np.ones((2,2), np.uint8)
        skeleton_clean = cv2.morphologyEx(skeleton_uint8, cv2.MORPH_OPEN, kernel_clean)
        
        return skeleton_clean, person_mask
    
    def find_skeleton_keypoints(self, skeleton_image):
        """
        Find key skeleton points (joints, endpoints, branch points)
        """
        # Find contours
        contours, _ = cv2.findContours(skeleton_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        keypoints = []
        
        for contour in contours:
            # Find endpoints and branch points
            for i, point in enumerate(contour):
                x, y = point[0]
                
                # Count neighbors to determine if it's an endpoint or branch point
                neighbors = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < skeleton_image.shape[1] and 
                            0 <= ny < skeleton_image.shape[0] and 
                            skeleton_image[ny, nx] > 0):
                            neighbors += 1
                
                # Endpoint (1 neighbor) or branch point (3+ neighbors)
                if neighbors == 1 or neighbors >= 3:
                    keypoints.append((x, y))
        
        return keypoints
    
    def match_keypoints_stereo(self, keypoints1, keypoints2, camera1, camera2):
        """
        Match keypoints between two camera views using epipolar constraints
        """
        if camera1 not in self.camera_matrices or camera2 not in self.camera_matrices:
            print("Cameras not calibrated!")
            return []
        
        # Convert keypoints to numpy arrays
        kp1 = np.array(keypoints1, dtype=np.float32)
        kp2 = np.array(keypoints2, dtype=np.float32)
        
        if len(kp1) == 0 or len(kp2) == 0:
            return []
        
        # For now, use simple distance-based matching
        # In practice, you'd use epipolar constraints and feature descriptors
        matches = []
        max_distance = 50  # Maximum pixel distance for matching
        
        for i, pt1 in enumerate(kp1):
            best_match = None
            best_distance = float('inf')
            
            for j, pt2 in enumerate(kp2):
                # Simple distance-based matching
                distance = np.linalg.norm(pt1 - pt2)
                if distance < max_distance and distance < best_distance:
                    best_distance = distance
                    best_match = j
            
            if best_match is not None:
                matches.append((i, best_match))
        
        return matches
    
    def triangulate_3d_points(self, keypoints1, keypoints2, matches, camera1, camera2):
        """
        Triangulate 3D points from matched 2D keypoints
        """
        if len(matches) == 0:
            return []
        
        # Get camera matrices
        K1 = self.camera_matrices[camera1]
        K2 = self.camera_matrices[camera2]
        
        # For stereo setup, we need to compute the fundamental matrix
        # This is a simplified version - in practice you'd use proper stereo calibration
        
        # Convert keypoints to homogeneous coordinates
        points1 = []
        points2 = []
        
        for match in matches:
            idx1, idx2 = match
            pt1 = keypoints1[idx1]
            pt2 = keypoints2[idx2]
            
            points1.append([pt1[0], pt1[1], 1])
            points2.append([pt2[0], pt2[1], 1])
        
        points1 = np.array(points1).T
        points2 = np.array(points2).T
        
        # Triangulate points (simplified - assumes cameras are calibrated)
        # In practice, you'd use cv2.triangulatePoints() with proper projection matrices
        points_3d = []
        
        for i in range(points1.shape[1]):
            # Simple triangulation assuming parallel cameras
            # This is a placeholder - real implementation would be more complex
            x1, y1 = points1[0, i], points1[1, i]
            x2, y2 = points2[0, i], points2[1, i]
            
            # Simplified 3D reconstruction
            # In practice, you'd use proper stereo geometry
            disparity = abs(x1 - x2)
            if disparity > 0:
                z = 1000 / disparity  # Simplified depth calculation
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                points_3d.append([x, y, z])
        
        return points_3d
    
    def reconstruct_3d_skeleton(self, video_paths, frame_number):
        """
        Reconstruct 3D skeleton from multiple camera views
        """
        print(f"\n=== 3D SKELETON RECONSTRUCTION (Frame {frame_number}) ===")
        
        skeletons_2d = {}
        keypoints_2d = {}
        
        # Extract 2D skeletons from each camera
        for camera_name, video_path in video_paths.items():
            print(f"Processing {camera_name}...")
            
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                skeleton, mask = self.extract_2d_skeleton(frame, camera_name)
                skeletons_2d[camera_name] = skeleton
                keypoints_2d[camera_name] = self.find_skeleton_keypoints(skeleton)
                print(f"Found {len(keypoints_2d[camera_name])} keypoints in {camera_name}")
            else:
                print(f"Could not read frame {frame_number} from {camera_name}")
        
        # Match keypoints between camera pairs
        camera_names = list(video_paths.keys())
        all_3d_points = []
        
        for i in range(len(camera_names)):
            for j in range(i + 1, len(camera_names)):
                cam1, cam2 = camera_names[i], camera_names[j]
                print(f"Matching keypoints between {cam1} and {cam2}...")
                
                matches = self.match_keypoints_stereo(
                    keypoints_2d[cam1], keypoints_2d[cam2], cam1, cam2
                )
                print(f"Found {len(matches)} matches")
                
                if len(matches) > 0:
                    points_3d = self.triangulate_3d_points(
                        keypoints_2d[cam1], keypoints_2d[cam2], matches, cam1, cam2
                    )
                    all_3d_points.extend(points_3d)
                    print(f"Triangulated {len(points_3d)} 3D points")
        
        return all_3d_points, skeletons_2d, keypoints_2d
    
    def visualize_3d_skeleton(self, points_3d, save_path=None):
        """
        Visualize the 3D skeleton
        """
        if len(points_3d) == 0:
            print("No 3D points to visualize")
            return
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        points_3d = np.array(points_3d)
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  c='red', s=50, alpha=0.7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Skeleton Reconstruction')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"3D visualization saved to {save_path}")
        
        plt.show()
    
    def save_2d_visualizations(self, skeletons_2d, keypoints_2d, frame_number, output_dir):
        """
        Save 2D skeleton visualizations from all cameras
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for camera_name, skeleton in skeletons_2d.items():
            # Save skeleton
            skeleton_path = os.path.join(output_dir, f"{camera_name}_frame_{frame_number:05d}_skeleton.jpg")
            cv2.imwrite(skeleton_path, skeleton)
            
            # Save skeleton with keypoints
            skeleton_with_kp = skeleton.copy()
            keypoints = keypoints_2d[camera_name]
            
            for kp in keypoints:
                cv2.circle(skeleton_with_kp, (int(kp[0]), int(kp[1])), 3, 255, -1)
            
            kp_path = os.path.join(output_dir, f"{camera_name}_frame_{frame_number:05d}_keypoints.jpg")
            cv2.imwrite(kp_path, skeleton_with_kp)
            
            print(f"Saved {camera_name} visualizations")

def main():
    # Initialize reconstructor
    reconstructor = Skeleton3DReconstructor()
    
    # Define calibration videos
    calibration_videos = {
        'left': '/Users/olivia/Documents/COMS4731/COMS4731-final-project/Raw Data/left_camera_calibration.mov',
        'right': '/Users/olivia/Documents/COMS4731/COMS4731-final-project/Raw Data/right_camera_calibration.mov',
        'tripod': '/Users/olivia/Documents/COMS4731/COMS4731-final-project/Raw Data/tripod_camera_calibration.mov'
    }
    
    # Define main video paths
    video_paths = {
        'left': '/Users/olivia/Documents/COMS4731/COMS4731-final-project/Raw Data/barre_left.mov',
        'right': '/Users/olivia/Documents/COMS4731/COMS4731-final-project/Raw Data/barre_right.mov',
        'tripod': '/Users/olivia/Documents/COMS4731/COMS4731-final-project/Raw Data/barre_tripod.mov'
    }
    
    # Calibrate cameras
    reconstructor.calibrate_cameras(calibration_videos)
    
    # Reconstruct 3D skeleton for a specific frame
    frame_number = 2130  # Adjust as needed
    points_3d, skeletons_2d, keypoints_2d = reconstructor.reconstruct_3d_skeleton(
        video_paths, frame_number
    )
    
    # Save visualizations
    output_dir = '/Users/olivia/Documents/COMS4731/COMS4731-final-project/3d_visualizations'
    reconstructor.save_2d_visualizations(skeletons_2d, keypoints_2d, frame_number, output_dir)
    
    # Visualize 3D skeleton
    if len(points_3d) > 0:
        reconstructor.visualize_3d_skeleton(
            points_3d, 
            os.path.join(output_dir, f'3d_skeleton_frame_{frame_number:05d}.png')
        )
    else:
        print("No 3D points were reconstructed")

if __name__ == "__main__":
    main()

