"""
MediaPipe 2D Skeleton Extraction

Extracts 2D pose landmarks from video using MediaPipe Pose.
Similar structure to depth_based_3d_skeleton.py but for 2D skeletons.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Install with: pip install mediapipe")


class MediaPipe2DSkeleton:
    def __init__(self, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize 2D skeleton extractor using MediaPipe.
        
        Args:
            model_complexity: 0, 1, or 2 (2 is most accurate, slower)
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required. Install with: pip install mediapipe")
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
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
    
    def get_2d_pose_normalized(self, frame):
        """
        Extract 2D pose keypoints in normalized coordinates (0-1 range).
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            landmarks: List of (x, y) tuples in normalized coordinates or None
            visibility: List of visibility scores
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None, None
        
        landmarks_2d = []
        visibility = []
        
        for landmark in results.pose_landmarks.landmark:
            landmarks_2d.append((landmark.x, landmark.y))
            visibility.append(landmark.visibility)
        
        return landmarks_2d, visibility
    
    def landmarks_to_array(self, landmarks_2d, visibility=None, min_visibility=0.5):
        """
        Convert 2D landmarks to numpy array format.
        
        Args:
            landmarks_2d: List of (x, y) tuples
            visibility: List of visibility scores (optional)
            min_visibility: Minimum visibility to include landmark
            
        Returns:
            coords_array: NumPy array of shape (num_landmarks, 2) with NaN for invalid points
        """
        coords_array = np.full((len(landmarks_2d), 2), np.nan)
        
        for i, (x, y) in enumerate(landmarks_2d):
            # Check visibility if provided
            if visibility and visibility[i] < min_visibility:
                continue
            
            coords_array[i] = [x, y]
        
        return coords_array
    
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
        
        if save_path:
            cv2.imwrite(save_path, frame_copy)
            print(f"Saved 2D skeleton visualization to {save_path}")
        
        if show:
            # Resize if too large for display
            h, w = frame_with_pose.shape[:2]
            max_display_size = 1200
            if w > max_display_size or h > max_display_size:
                scale = max_display_size / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame_display = cv2.resize(frame_copy, (new_w, new_h))
            else:
                frame_display = frame_copy
            
            cv2.imshow('2D Skeleton', frame_display)
            cv2.waitKey(1)  # Non-blocking wait
        
        return frame_copy
    
    def process_frame(self, frame, min_visibility=0.5, normalized=False):
        """
        Process a single frame to get 2D skeleton.
        
        Args:
            frame: Input frame (BGR)
            min_visibility: Minimum visibility for landmarks
            normalized: If True, return normalized coordinates (0-1), else pixel coordinates
            
        Returns:
            landmarks_2d: 2D landmarks (list of tuples)
            coords_array: NumPy array of shape (33, 2)
            frame_with_pose: Frame with 2D pose drawn
            visibility: Visibility scores for landmarks
        """
        # Get 2D pose
        if normalized:
            landmarks_2d, visibility = self.get_2d_pose_normalized(frame)
        else:
            landmarks_2d, visibility = self.get_2d_pose(frame)
        
        if landmarks_2d is None:
            return None, None, frame, None
        
        # Convert to array
        coords_array = self.landmarks_to_array(landmarks_2d, visibility, min_visibility)
        
        # Draw 2D pose on frame
        frame_with_pose = self.visualize_2d_skeleton(
            frame, landmarks_2d, visibility, 
            save_path=None, show=False, min_visibility=min_visibility
        )
        
        return landmarks_2d, coords_array, frame_with_pose, visibility


def process_video_2d_skeleton(video_path, output_dir, 
                              start_frame=0, num_frames=None,
                              min_visibility=0.5, show_2d=True,
                              model_complexity=2,
                              save_coords=True, save_visualizations=True,
                              normalized_coords=False):
    """
    Process video frames to extract 2D skeletons using MediaPipe.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save outputs
        start_frame: Starting frame number
        num_frames: Number of frames to process (None for all)
        min_visibility: Minimum visibility for landmarks
        show_2d: Whether to display 2D skeleton visualization during processing
        model_complexity: MediaPipe model complexity (0, 1, or 2)
        save_coords: Whether to save coordinate arrays
        save_visualizations: Whether to save visualization images
        normalized_coords: If True, save normalized coordinates (0-1), else pixel coordinates
    """
    if not MEDIAPIPE_AVAILABLE:
        print("Error: MediaPipe is required for pose estimation.")
        print("Install with: pip install mediapipe")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize skeleton extractor
    skeleton_extractor = MediaPipe2DSkeleton(
        model_complexity=model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # Set to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    processed_count = 0
    no_pose_count = 0
    
    print(f"\nProcessing video: {video_path}")
    print(f"Starting from frame {start_frame}")
    print(f"Output directory: {output_dir}")
    print()
    
    while True:
        if num_frames and processed_count >= num_frames:
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        landmarks_2d, coords_array, frame_with_pose, visibility = skeleton_extractor.process_frame(
            frame, min_visibility=min_visibility, normalized=normalized_coords
        )
        
        if landmarks_2d is None:
            no_pose_count += 1
            if no_pose_count % 10 == 0:
                print(f"Frame {frame_count}: No pose detected (total: {no_pose_count})")
            frame_count += 1
            continue
        
        # Save coordinates if requested
        if save_coords:
            coords_path = os.path.join(output_dir, f"skeleton_2d_coords_frame_{frame_count:05d}.npy")
            np.save(coords_path, coords_array)
        
        # Save and optionally display 2D visualization
        if save_visualizations:
            pose_2d_path = os.path.join(output_dir, f"pose_2d_frame_{frame_count:05d}.png")
            skeleton_extractor.visualize_2d_skeleton(
                frame, landmarks_2d, visibility, 
                save_path=pose_2d_path, 
                show=show_2d,
                min_visibility=min_visibility
            )
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} frames...")
        
        frame_count += 1
    
    cap.release()
    if show_2d:
        cv2.destroyAllWindows()
    
    print(f"\n✓ Processing complete!")
    print(f"  Processed {processed_count} frames with pose detected")
    print(f"  Frames without pose: {no_pose_count}")
    print(f"  Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract 2D skeletons from video using MediaPipe")
    parser.add_argument("--video", type=str,
                       help="Path to input video file")
    parser.add_argument("--output", type=str,
                       default="2d_skeleton_results",
                       help="Output directory for 2D skeleton data")
    parser.add_argument("--start-frame", type=int, default=0,
                       help="Starting frame number")
    parser.add_argument("--num-frames", type=int, default=None,
                       help="Number of frames to process (default: all)")
    parser.add_argument("--min-visibility", type=float, default=0.5,
                       help="Minimum visibility for landmarks (0.0-1.0)")
    parser.add_argument("--no-show-2d", action="store_true",
                       help="Don't display 2D skeleton visualization during processing")
    parser.add_argument("--model-complexity", type=int, default=2,
                       choices=[0, 1, 2],
                       help="MediaPipe model complexity (0=fastest, 2=most accurate)")
    parser.add_argument("--no-coords", action="store_true",
                       help="Don't save coordinate arrays")
    parser.add_argument("--no-visualizations", action="store_true",
                       help="Don't save visualization images")
    parser.add_argument("--normalized", action="store_true",
                       help="Save normalized coordinates (0-1) instead of pixel coordinates")
    
    args = parser.parse_args()
    
    if not MEDIAPIPE_AVAILABLE:
        print("Error: MediaPipe is required for pose estimation.")
        print("Install with: pip install mediapipe")
        return
    
    if not args.video:
        print("Error: --video argument is required")
        parser.print_help()
        return
    
    process_video_2d_skeleton(
        video_path=args.video,
        output_dir=args.output,
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        min_visibility=args.min_visibility,
        show_2d=not args.no_show_2d,
        model_complexity=args.model_complexity,
        save_coords=not args.no_coords,
        save_visualizations=not args.no_visualizations,
        normalized_coords=args.normalized
    )


if __name__ == "__main__":
    main()

