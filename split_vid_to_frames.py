import cv2
import os
import glob

def video_to_frames(video_path, output_dir, frame_interval=300):
    """
    Splits a video into individual frames and saves them as images.
    Uses every nth frame (default 10th) and includes video name and frame ID.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where frames will be saved.
        frame_interval (int): Extract every nth frame (default: 10).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frame_count = 0
    saved_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop when video ends

        # Only save every nth frame
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"{video_name}_frame_{frame_count:05d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_frames} frames from '{video_name}' to '{output_dir}'")

def process_all_videos(raw_data_dir, output_base_dir, frame_interval=300):
    """
    Process all video files in the raw data directory.
    
    Args:
        raw_data_dir (str): Directory containing video files.
        output_base_dir (str): Base directory where frame folders will be created.
        frame_interval (int): Extract every nth frame (default: 10).
    """
    # Find all video files
    video_extensions = ['*.mov', '*.mp4', '*.avi', '*.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(raw_data_dir, ext)))
    
    if not video_files:
        print(f"No video files found in {raw_data_dir}")
        return
    
    print(f"Found {len(video_files)} video files to process")
    
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(output_base_dir, video_name)
        
        try:
            video_to_frames(video_path, output_dir, frame_interval)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

# Process only barre_tripod.mov
if __name__ == "__main__":
    video_path = "/Users/olivia/Documents/COMS4731/COMS4731-final-project/Raw Data/tripod_camera_calibration.mov"
    output_dir = "/Users/olivia/Documents/COMS4731/COMS4731-final-project/frames/tripod_calibration/"
    
    video_to_frames(video_path, output_dir, frame_interval=200)
