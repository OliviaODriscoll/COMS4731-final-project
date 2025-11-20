import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Install with: pip install mediapipe")


class DepthBased3DSkeleton:
    def __init__(self, use_mediapipe=True):
        """
        Initialize 3D skeleton reconstructor using depth maps.
        
        Args:
            use_mediapipe: If True, use MediaPipe for pose estimation. 
                          If False, will need alternative pose estimation method.
        """
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE
        
        if self.use_mediapipe:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,  # 0, 1, or 2 (2 is most accurate)
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # MediaPipe Pose landmark connections (33 landmarks)
        # Key landmarks: 0=nose, 11=left_shoulder, 12=right_shoulder, 
        # 13=left_elbow, 14=right_elbow, 15=left_wrist, 16=right_wrist,
        # 23=left_hip, 24=right_hip, 25=left_knee, 26=right_knee,
        # 27=left_ankle, 28=right_ankle
        self.landmark_connections = [
            # Face
            (0, 1), (0, 2), (1, 3), (2, 4),
            # Upper body
            (11, 12),  # Shoulders
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 23), (12, 24),  # Torso
            # Lower body
            (23, 24),  # Hips
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28),  # Right leg
        ]
    
    def get_2d_pose(self, frame):
        """
        Extract 2D pose keypoints from frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            landmarks: List of (x, y) tuples or None if no pose detected
            visibility: List of visibility scores
        """
        if not self.use_mediapipe:
            raise NotImplementedError("MediaPipe is required for pose estimation")
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None, None
        
        landmarks_2d = []
        visibility = []
        
        h, w = frame.shape[:2]
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks_2d.append((x, y))
            visibility.append(landmark.visibility)
        
        return landmarks_2d, visibility
    
    def get_depth_at_point(self, depth_map, x, y, window_size=5):
        """
        Get depth value at a point, using a small window for robustness.
        
        Args:
            depth_map: Depth map array
            x, y: Pixel coordinates
            window_size: Size of window to average over
            
        Returns:
            depth: Depth value (or None if invalid)
        """
        h, w = depth_map.shape
        
        # Clamp coordinates
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        
        # Extract window around point
        half = window_size // 2
        x_min = max(0, x - half)
        x_max = min(w, x + half + 1)
        y_min = max(0, y - half)
        y_max = min(h, y + half + 1)
        
        window = depth_map[y_min:y_max, x_min:x_max]
        
        # Filter out invalid depths (zeros or very small values)
        valid_depths = window[window > 0.01 * window.max()]
        
        if len(valid_depths) == 0:
            return None
        
        # Return median depth (more robust than mean)
        return np.median(valid_depths)
    
    def landmarks_to_3d(self, landmarks_2d, depth_map, visibility=None, min_visibility=0.5):
        """
        Convert 2D landmarks to 3D using depth map.
        
        Args:
            landmarks_2d: List of (x, y) tuples
            depth_map: Depth map array
            visibility: List of visibility scores (optional)
            min_visibility: Minimum visibility to include landmark
            
        Returns:
            landmarks_3d: List of (x, y, z) tuples or None for invalid points
        """
        landmarks_3d = []
        
        for i, (x, y) in enumerate(landmarks_2d):
            # Check visibility if provided
            if visibility and visibility[i] < min_visibility:
                landmarks_3d.append(None)
                continue
            
            # Get depth at this point
            z = self.get_depth_at_point(depth_map, x, y)
            
            if z is None:
                landmarks_3d.append(None)
            else:
                # Note: MediaPipe uses normalized coordinates, but we've already converted
                # to pixel coordinates. Depth from MiDaS is in arbitrary units.
                # We'll use pixel coordinates for x, y and depth for z.
                landmarks_3d.append((x, y, z))
        
        return landmarks_3d
    
    def visualize_3d_skeleton(self, landmarks_3d, save_path=None, show=True):
        """
        Visualize 3D skeleton.
        
        Args:
            landmarks_3d: List of (x, y, z) tuples or None
            save_path: Path to save visualization
            show: Whether to display the plot
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract valid points
        valid_points = [(i, pt) for i, pt in enumerate(landmarks_3d) if pt is not None]
        
        if len(valid_points) == 0:
            print("No valid 3D points to visualize")
            return
        
        # Plot points
        x_coords = [pt[0] for _, pt in valid_points]
        y_coords = [pt[1] for _, pt in valid_points]
        z_coords = [pt[2] for _, pt in valid_points]
        
        ax.scatter(x_coords, y_coords, z_coords, c='red', s=50, alpha=0.8)
        
        # Draw connections
        for start_idx, end_idx in self.landmark_connections:
            if (start_idx < len(landmarks_3d) and end_idx < len(landmarks_3d) and
                landmarks_3d[start_idx] is not None and landmarks_3d[end_idx] is not None):
                x_line = [landmarks_3d[start_idx][0], landmarks_3d[end_idx][0]]
                y_line = [landmarks_3d[start_idx][1], landmarks_3d[end_idx][1]]
                z_line = [landmarks_3d[start_idx][2], landmarks_3d[end_idx][2]]
                ax.plot(x_line, y_line, z_line, 'b-', linewidth=2, alpha=0.6)
        
        # Label key points
        key_points = {0: 'Nose', 11: 'L.Shoulder', 12: 'R.Shoulder',
                     15: 'L.Wrist', 16: 'R.Wrist', 23: 'L.Hip', 24: 'R.Hip',
                     27: 'L.Ankle', 28: 'R.Ankle'}
        for idx, label in key_points.items():
            if idx < len(landmarks_3d) and landmarks_3d[idx] is not None:
                pt = landmarks_3d[idx]
                ax.text(pt[0], pt[1], pt[2], label, fontsize=8)
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_zlabel('Z (depth)')
        ax.set_title('3D Skeleton from Depth Map')
        
        # Set equal aspect ratio for better visualization
        max_range = np.array([max(x_coords) - min(x_coords),
                             max(y_coords) - min(y_coords),
                             max(z_coords) - min(z_coords)]).max() / 2.0
        mid_x = (max(x_coords) + min(x_coords)) * 0.5
        mid_y = (max(y_coords) + min(y_coords)) * 0.5
        mid_z = (max(z_coords) + min(z_coords)) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 3D visualization to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def draw_2d_pose_on_frame(self, frame, landmarks_2d, visibility=None, min_visibility=0.5):
        """
        Draw 2D pose on frame.
        
        Args:
            frame: Input frame
            landmarks_2d: List of (x, y) tuples
            visibility: List of visibility scores
            min_visibility: Minimum visibility to draw landmark
            
        Returns:
            frame_with_pose: Frame with pose drawn
        """
        frame_copy = frame.copy()
        
        if landmarks_2d:
            # Draw connections
            for start_idx, end_idx in self.landmark_connections:
                if (start_idx < len(landmarks_2d) and end_idx < len(landmarks_2d)):
                    # Check visibility if provided
                    if visibility:
                        if (visibility[start_idx] < min_visibility or 
                            visibility[end_idx] < min_visibility):
                            continue
                    
                    pt1 = landmarks_2d[start_idx]
                    pt2 = landmarks_2d[end_idx]
                    cv2.line(frame_copy, pt1, pt2, (0, 255, 0), 2)
            
            # Draw keypoints
            for i, (x, y) in enumerate(landmarks_2d):
                if visibility and visibility[i] < min_visibility:
                    continue
                cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
        
        return frame_copy
    
    def visualize_2d_skeleton(self, frame, landmarks_2d, visibility=None, 
                             save_path=None, show=True, min_visibility=0.5):
        """
        Visualize 2D skeleton on frame.
        
        Args:
            frame: Input frame
            landmarks_2d: List of (x, y) tuples
            visibility: List of visibility scores
            save_path: Path to save visualization
            show: Whether to display the frame
            min_visibility: Minimum visibility to draw landmark
            
        Returns:
            frame_with_pose: Frame with 2D skeleton drawn
        """
        frame_with_pose = self.draw_2d_pose_on_frame(frame, landmarks_2d, visibility, min_visibility)
        
        if save_path:
            cv2.imwrite(save_path, frame_with_pose)
            print(f"Saved 2D skeleton visualization to {save_path}")
        
        if show:
            # Resize if too large for display
            h, w = frame_with_pose.shape[:2]
            max_display_size = 1200
            if w > max_display_size or h > max_display_size:
                scale = max_display_size / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame_display = cv2.resize(frame_with_pose, (new_w, new_h))
            else:
                frame_display = frame_with_pose
            
            cv2.imshow('2D Skeleton', frame_display)
            cv2.waitKey(1)  # Non-blocking wait
        
        return frame_with_pose
    
    def process_frame(self, frame, depth_map, min_visibility=0.5):
        """
        Process a single frame to get 3D skeleton.
        
        Args:
            frame: Input frame (BGR)
            depth_map: Depth map array
            min_visibility: Minimum visibility for landmarks
            
        Returns:
            landmarks_2d: 2D landmarks
            landmarks_3d: 3D landmarks
            frame_with_pose: Frame with 2D pose drawn
            visibility: Visibility scores for landmarks
        """
        # Get 2D pose
        landmarks_2d, visibility = self.get_2d_pose(frame)
        
        if landmarks_2d is None:
            return None, None, frame, None
        
        # Convert to 3D
        landmarks_3d = self.landmarks_to_3d(landmarks_2d, depth_map, visibility, min_visibility)
        
        # Draw 2D pose on frame
        frame_with_pose = self.draw_2d_pose_on_frame(frame, landmarks_2d, visibility, min_visibility)
        
        return landmarks_2d, landmarks_3d, frame_with_pose, visibility


def process_video_with_depth(video_path, depth_dir, output_dir, 
                             start_frame=0, num_frames=None,
                             min_visibility=0.5, show_2d=True):
    """
    Process video frames with corresponding depth maps to create 3D skeletons.
    
    Args:
        video_path: Path to input video
        depth_dir: Directory containing depth_frame_*.npy files
        output_dir: Directory to save outputs
        start_frame: Starting frame number
        num_frames: Number of frames to process (None for all)
        min_visibility: Minimum visibility for landmarks
        show_2d: Whether to display 2D skeleton visualization during processing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize skeleton reconstructor
    skeleton_reconstructor = DepthBased3DSkeleton()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    # Set to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    processed_count = 0
    
    print(f"Processing video: {video_path}")
    print(f"Starting from frame {start_frame}")
    print(f"Depth maps from: {depth_dir}")
    print()
    
    while True:
        if num_frames and processed_count >= num_frames:
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Load corresponding depth map
        depth_file = os.path.join(depth_dir, f"depth_frame_{frame_count:05d}.npy")
        
        if not os.path.exists(depth_file):
            print(f"Warning: Depth file not found for frame {frame_count}: {depth_file}")
            frame_count += 1
            continue
        
        depth_map = np.load(depth_file)
        
        # Process frame
        landmarks_2d, landmarks_3d, frame_with_pose, visibility = skeleton_reconstructor.process_frame(
            frame, depth_map, min_visibility=min_visibility
        )
        
        if landmarks_3d is None:
            print(f"Frame {frame_count}: No pose detected")
            frame_count += 1
            continue
        
        # Save and optionally display 2D visualization
        pose_2d_path = os.path.join(output_dir, f"pose_2d_frame_{frame_count:05d}.png")
        skeleton_reconstructor.visualize_2d_skeleton(
            frame, landmarks_2d, visibility, 
            save_path=pose_2d_path, 
            show=show_2d,
            min_visibility=min_visibility
        )
        
        # Save 3D visualization
        pose_3d_path = os.path.join(output_dir, f"skeleton_3d_frame_{frame_count:05d}.png")
        skeleton_reconstructor.visualize_3d_skeleton(landmarks_3d, pose_3d_path, show=False)
        
        # Save 3D coordinates (convert to numpy array with NaN for invalid points)
        coords_path = os.path.join(output_dir, f"skeleton_3d_coords_frame_{frame_count:05d}.npy")
        coords_array = np.full((len(landmarks_3d), 3), np.nan)
        for i, pt in enumerate(landmarks_3d):
            if pt is not None:
                coords_array[i] = pt
        np.save(coords_path, coords_array)
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} frames...")
        
        frame_count += 1
    
    cap.release()
    if show_2d:
        cv2.destroyAllWindows()
    print(f"\n✓ Processing complete!")
    print(f"  Processed {processed_count} frames")
    print(f"  Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create 3D skeleton from video using depth maps")
    parser.add_argument("--video", type=str,
                       default="Raw Data/barre_tripod.mov",
                       help="Path to input video file")
    parser.add_argument("--depth-dir", type=str,
                       default="midas_depth_results",
                       help="Directory containing depth_frame_*.npy files")
    parser.add_argument("--output", type=str,
                       default="3d_skeleton_results",
                       help="Output directory for 3D skeleton visualizations")
    parser.add_argument("--start-frame", type=int, default=0,
                       help="Starting frame number")
    parser.add_argument("--num-frames", type=int, default=None,
                       help="Number of frames to process (default: all)")
    parser.add_argument("--min-visibility", type=float, default=0.5,
                       help="Minimum visibility for landmarks (0.0-1.0)")
    parser.add_argument("--no-show-2d", action="store_true",
                       help="Don't display 2D skeleton visualization during processing")
    
    args = parser.parse_args()
    
    if not MEDIAPIPE_AVAILABLE:
        print("Error: MediaPipe is required for pose estimation.")
        print("Install with: pip install mediapipe")
        return
    
    process_video_with_depth(
        video_path=args.video,
        depth_dir=args.depth_dir,
        output_dir=args.output,
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        min_visibility=args.min_visibility,
        show_2d=not args.no_show_2d
    )


if __name__ == "__main__":
    main()

