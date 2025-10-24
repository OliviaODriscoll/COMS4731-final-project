import cv2
import os

def video_to_frames(video_path, output_dir):
    """c
    Splits a video into individual frames and saves them as images.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where frames will be saved.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop when video ends

        frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to '{output_dir}'")

# Example usage:
video_to_frames("/Users/olivia/Downloads/videoplayback.mp4", "/Users/olivia/Documents/COMS4731/COMS4731-final-project/frames")
