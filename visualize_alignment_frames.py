"""
Display raw video frames side-by-side to verify alignment.

Shows frames from all three videos at the same timepoint (accounting for offsets)
so you can visually compare and determine the exact offset mapping.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def get_video_fps(video_path: str) -> float:
    """Get FPS from video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 30.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0 or np.isnan(fps):
        return 30.0
    return fps


def seconds_to_frame(seconds: float, fps: float) -> int:
    """Convert seconds to frame number."""
    return int(seconds * fps)


def frame_to_seconds(frame: int, fps: float) -> float:
    """Convert frame number to seconds."""
    return frame / fps


def load_frame(video_path: str, frame_num: int):
    """Load a specific frame from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame
    return None


def resize_frame(frame, target_width=640, target_height=480):
    """Resize frame to target dimensions while maintaining aspect ratio, then pad if needed."""
    if frame is None:
        return None
    
    h, w = frame.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Create a black canvas of target size
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Center the resized frame on the canvas
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def visualize_alignment(video_paths: dict, reference_time: float = 26.0, 
                       offsets: dict = None):
    """
    Display frames side-by-side for alignment verification.
    
    Args:
        video_paths: Dictionary mapping video names to paths
        reference_time: Reference time in arabesque_left (seconds)
        offsets: Dictionary mapping video names to time offsets (seconds)
    """
    if offsets is None:
        # Current assumed offsets
        offsets = {
            "arabesque_left": 0.0,
            "barre_right": 8.0,
            "barre_tripod": 14.0
        }
    
    # Get FPS for each video
    fps_dict = {}
    for name, path in video_paths.items():
        fps_dict[name] = get_video_fps(path)
        print(f"{name}: FPS = {fps_dict[name]:.2f}")
    
    # Current time for each video (independent control)
    current_times = {}
    for name in video_paths.keys():
        if name == "arabesque_left":
            current_times[name] = reference_time
        else:
            current_times[name] = reference_time + offsets.get(name, 0.0)
    
    # Track which video is currently selected for individual scrubbing
    selected_video = "arabesque_left"
    video_list = ["arabesque_left", "barre_right", "barre_tripod"]
    
    print("\nControls:")
    print("  'a' / 'd' - Move selected video backward/forward 0.1 seconds")
    print("  's' / 'w' - Move selected video backward/forward 1.0 seconds")
    print("  'q' / 'e' - Switch selected video (left/right)")
    print("  'r' - Reset all to reference time (26.0s)")
    print("  'p' - Print current times and calculated offsets")
    print("  'ESC' - Quit and show final offsets")
    print(f"\nCurrently selected: {selected_video} (highlighted in green)")
    print("\nPress any key to start...")
    
    window_name = "Frame Alignment Verification"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 600)
    
    while True:
        # Load frames from all videos
        frames = {}
        frame_nums = {}
        
        for name, path in video_paths.items():
            time = current_times[name]
            fps = fps_dict[name]
            frame_num = seconds_to_frame(time, fps)
            frame_nums[name] = frame_num
            
            frame = load_frame(path, frame_num)
            if frame is not None:
                frames[name] = resize_frame(frame, target_width=640, target_height=480)
            else:
                frames[name] = None
        
        # Create side-by-side display
        display_frames = []
        labels = []
        
        for name in ["arabesque_left", "barre_right", "barre_tripod"]:
            if name in frames and frames[name] is not None:
                frame = frames[name].copy()
                # Add border to highlight selected video
                if name == selected_video:
                    cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 255, 0), 5)
                
                # Add text overlay
                time = current_times[name]
                frame_num = frame_nums[name]
                # Calculate offset relative to arabesque_left
                if name == "arabesque_left":
                    offset = 0.0
                else:
                    offset = current_times[name] - current_times["arabesque_left"]
                
                label = f"{name}\nTime: {time:.2f}s\nFrame: {frame_num}\nOffset: {offset:+.2f}s"
                labels.append(label)
                
                # Add text to frame
                color = (0, 255, 0) if name == selected_video else (255, 255, 255)
                cv2.putText(frame, f"{name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Time: {time:.2f}s", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Frame: {frame_num}", (10, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Offset: {offset:+.2f}s", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if name == selected_video:
                    cv2.putText(frame, "SELECTED", (10, 135), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                display_frames.append(frame)
            else:
                # Create blank frame if video not found
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, f"{name} - Not found", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                display_frames.append(blank)
                labels.append(f"{name}\nNot found")
        
        # Combine frames horizontally
        combined = np.hstack(display_frames)
        
        # Add instruction text at bottom
        instruction = f"Selected: {selected_video} | a/d: ±0.1s | s/w: ±1.0s | q/e: switch video | r: reset | p: print | ESC: quit"
        cv2.putText(combined, instruction, (10, combined.shape[0] - 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Show current times and calculated offsets
        barre_right_offset = current_times["barre_right"] - current_times["arabesque_left"]
        barre_tripod_offset = current_times["barre_tripod"] - current_times["arabesque_left"]
        offset_text = f"Times: left={current_times['arabesque_left']:.2f}s, right={current_times['barre_right']:.2f}s, tripod={current_times['barre_tripod']:.2f}s"
        cv2.putText(combined, offset_text, (10, combined.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        offset_calc = f"Offsets: barre_right={barre_right_offset:+.2f}s, barre_tripod={barre_tripod_offset:+.2f}s"
        cv2.putText(combined, offset_calc, (10, combined.shape[0] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow(window_name, combined)
        
        # Handle keyboard input
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27:  # ESC to quit
            break
        elif key == ord('a'):  # Move selected video back 0.1s
            current_times[selected_video] -= 0.1
            if current_times[selected_video] < 0:
                current_times[selected_video] = 0.0
        elif key == ord('d'):  # Move selected video forward 0.1s
            current_times[selected_video] += 0.1
        elif key == ord('s'):  # Move selected video back 1.0s
            current_times[selected_video] -= 1.0
            if current_times[selected_video] < 0:
                current_times[selected_video] = 0.0
        elif key == ord('w'):  # Move selected video forward 1.0s
            current_times[selected_video] += 1.0
        elif key == ord('q'):  # Switch to previous video
            current_idx = video_list.index(selected_video)
            selected_video = video_list[(current_idx - 1) % len(video_list)]
            print(f"Selected: {selected_video}")
        elif key == ord('e'):  # Switch to next video
            current_idx = video_list.index(selected_video)
            selected_video = video_list[(current_idx + 1) % len(video_list)]
            print(f"Selected: {selected_video}")
        elif key == ord('r'):  # Reset all to reference time
            reference_time = 26.0
            current_times["arabesque_left"] = reference_time
            current_times["barre_right"] = reference_time + 8.0
            current_times["barre_tripod"] = reference_time + 14.0
            print("Reset to reference times")
        elif key == ord('p'):  # Print current times and offsets
            barre_right_offset = current_times["barre_right"] - current_times["arabesque_left"]
            barre_tripod_offset = current_times["barre_tripod"] - current_times["arabesque_left"]
            print("\n" + "="*60)
            print("Current times:")
            print(f"  arabesque_left: {current_times['arabesque_left']:.2f}s")
            print(f"  barre_right: {current_times['barre_right']:.2f}s")
            print(f"  barre_tripod: {current_times['barre_tripod']:.2f}s")
            print("\nCalculated offsets (relative to arabesque_left):")
            print(f"  barre_right: {barre_right_offset:+.2f}s")
            print(f"  barre_tripod: {barre_tripod_offset:+.2f}s")
            print("="*60)
    
    cv2.destroyAllWindows()
    
    # Calculate final offsets relative to arabesque_left
    final_offsets = {}
    for name in video_paths.keys():
        if name == "arabesque_left":
            final_offsets[name] = 0.0
        else:
            final_offsets[name] = current_times[name] - current_times["arabesque_left"]
    
    print("\n" + "="*60)
    print("Final alignment:")
    print("="*60)
    print(f"arabesque_left time: {current_times['arabesque_left']:.2f}s (reference)")
    print(f"barre_right time: {current_times['barre_right']:.2f}s")
    print(f"barre_tripod time: {current_times['barre_tripod']:.2f}s")
    print("\nFinal offsets (relative to arabesque_left):")
    print(f"  barre_right: {final_offsets.get('barre_right', 0.0):+.2f} seconds")
    print(f"  barre_tripod: {final_offsets.get('barre_tripod', 0.0):+.2f} seconds")
    print("="*60)
    print("\nUse these offset values to update the alignment code.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize frames side-by-side to verify alignment"
    )
    
    parser.add_argument("--video-dir", type=str,
                       default="/Users/olivia/Documents/COMS4731/COMS4731-final-project/Raw Data",
                       help="Directory containing video files")
    parser.add_argument("--reference-time", type=float, default=26.0,
                       help="Reference time in arabesque_left video (seconds)")
    
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    
    # Find video files
    video_paths = {}
    for video_name in ["arabesque_left", "barre_right", "barre_tripod"]:
        for ext in ['.mov', '.MOV', '.mp4', '.MP4']:
            candidate = video_dir / f"{video_name}{ext}"
            if candidate.exists():
                video_paths[video_name] = str(candidate)
                break
    
    if len(video_paths) < 3:
        print(f"Error: Found only {len(video_paths)} videos")
        print(f"Looking for: arabesque_left, barre_right, barre_tripod")
        return
    
    print("="*60)
    print("Frame Alignment Visualization")
    print("="*60)
    print(f"\nVideos found:")
    for name, path in video_paths.items():
        print(f"  {name}: {path}")
    
    visualize_alignment(video_paths, reference_time=args.reference_time)


if __name__ == "__main__":
    main()

