import cv2
import os

def extract_frame_at_timestamp(video_path, timestamp_seconds, output_path):
    """
    Extract a single frame from a video at a specific timestamp.
    
    Args:
        video_path (str): Path to the input video file
        timestamp_seconds (float): Timestamp in seconds
        output_path (str): Path where the frame will be saved
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video: {os.path.basename(video_path)}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Target timestamp: {timestamp_seconds} seconds")
    
    # Set video position to timestamp (in milliseconds)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
    
    # Read the frame
    ret, frame = cap.read()
    
    if ret:
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save frame
        cv2.imwrite(output_path, frame)
        print(f"  ✓ Frame extracted and saved to: {output_path}")
        
        # Get actual timestamp we landed on
        actual_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(f"  Actual timestamp: {actual_timestamp:.2f} seconds (frame {frame_number})")
        
        cap.release()
        return True
    else:
        print(f"  ✗ Failed to read frame at timestamp {timestamp_seconds} seconds")
        cap.release()
        return False


if __name__ == "__main__":
    base_dir = "/Users/olivia/Documents/COMS4731/COMS4731-final-project"
    raw_data_dir = os.path.join(base_dir, "Raw Data")
    output_dir = os.path.join(base_dir, "frames", "reconstruction_frames")
    
    # Extract frames at specified timestamps
    extractions = [
        {
            'video': 'arabesque_left.mov',
            'timestamp': 26,
            'output_name': 'arabesque_left_frame_26s.png'
        },
        {
            'video': 'barre_tripod.mov',
            'timestamp': 39,
            'output_name': 'barre_tripod_frame_39s.png'
        },
        {
            'video': 'barre_right.mov',
            'timestamp': 35,
            'output_name': 'barre_right_frame_35s.png'
        }
    ]
    
    print("=" * 60)
    print("Extracting frames at specific timestamps")
    print("=" * 60)
    
    for ext in extractions:
        video_path = os.path.join(raw_data_dir, ext['video'])
        output_path = os.path.join(output_dir, ext['output_name'])
        
        print(f"\n{ext['video']} at {ext['timestamp']} seconds:")
        extract_frame_at_timestamp(video_path, ext['timestamp'], output_path)
    
    print("\n" + "=" * 60)
    print("Extraction complete!")
    print(f"Frames saved to: {output_dir}")
    print("=" * 60)

