import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path
import os
import argparse


class MiDaSDepthEstimator:
    def __init__(self, model_path=None, model_type="DPT_Large"):
        """
        Initialize MiDaS depth estimator.
        
        Args:
            model_path: Path to local model file (e.g., dpt_swin2_large_384.pt)
                       If None, will download from torch.hub
            model_type: Type of model to use. Options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Track which model type we're actually using
        actual_model_type = model_type
        
        # Load MiDaS model
        if model_path and os.path.exists(model_path):
            print(f"Loading model from local file: {model_path}")
            try:
                # Try loading as torchscript model first
                self.model = torch.jit.load(model_path, map_location=self.device)
                self.model.eval()
                # For torchscript models, assume DPT transform
                actual_model_type = "DPT_Large"
            except Exception as e:
                # If that fails, try loading as regular model
                print(f"  Torchscript load failed ({e}), attempting to load as regular PyTorch model...")
                self.model = torch.hub.load("intel-isl/MiDaS", model_type)
                # Try to load state dict if it's a checkpoint
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    elif isinstance(checkpoint, dict):
                        self.model.load_state_dict(checkpoint)
                    else:
                        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                except Exception as e2:
                    print(f"  Could not load weights from file ({e2}), using default model")
                self.model.to(self.device)
                self.model.eval()
        else:
            print(f"Loading {model_type} model from torch.hub...")
            self.model = torch.hub.load("intel-isl/MiDaS", model_type)
            self.model.to(self.device)
            self.model.eval()
        
        # MiDaS transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        
        if actual_model_type == "DPT_Large" or actual_model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
    
    def process_frame(self, frame):
        """
        Process a single frame to get depth map.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            depth_map: Depth map as numpy array
        """
        # Convert BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        input_batch = self.transform(img).to(self.device)
        
        # Predict depth
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy
        depth_map = prediction.cpu().numpy()
        
        return depth_map
    
    def normalize_depth(self, depth_map, min_percentile=5, max_percentile=95):
        """
        Normalize depth map to 0-255 range for visualization.
        Clips far and close objects to focus on middle ground.
        
        Args:
            depth_map: Raw depth map
            min_percentile: Percentile for minimum depth (clips close objects)
            max_percentile: Percentile for maximum depth (clips far objects)
            
        Returns:
            depth_colored: Colored depth map (BGR)
            depth_normalized: Normalized depth map (uint8)
        """
        # Calculate percentile-based clipping thresholds
        min_depth = np.percentile(depth_map, min_percentile)
        max_depth = np.percentile(depth_map, max_percentile)
        
        # Clip depth map to remove far and close objects
        depth_clipped = np.clip(depth_map, min_depth, max_depth)
        
        # Normalize the clipped middle ground to 0-255
        depth_normalized = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)
        
        # Apply colormap for better visualization
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        
        return depth_colored, depth_normalized


def process_video(video_path, output_dir, model_path=None, model_type="DPT_Large", 
                  save_frames=True, save_video=True, frame_interval=1,
                  min_percentile=5, max_percentile=95):
    """
    Process a video file to generate depth maps using MiDaS.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save output files
        model_path: Path to local MiDaS model file (optional)
        model_type: Type of MiDaS model to use
        save_frames: Whether to save individual depth map frames
        save_video: Whether to save depth visualization video
        frame_interval: Process every nth frame (1 = all frames)
        min_percentile: Percentile for minimum depth clipping (clips close objects)
        max_percentile: Percentile for maximum depth clipping (clips far objects)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize depth estimator
    estimator = MiDaSDepthEstimator(model_path=model_path, model_type=model_type)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Processing every {frame_interval} frame(s)")
    
    # Setup video writer if saving video
    video_writer = None
    if save_video:
        video_name = Path(video_path).stem
        output_video_path = os.path.join(output_dir, f"{video_name}_depth.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps / frame_interval, (width, height))
        print(f"  Saving depth video to: {output_video_path}")
    
    # Process frames
    frame_count = 0
    processed_count = 0
    
    print(f"\nProcessing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every nth frame
        if frame_count % frame_interval == 0:
            # Get depth map
            depth_map = estimator.process_frame(frame)
            
            # Normalize and colorize (with clipping to focus on middle ground)
            depth_colored, depth_normalized = estimator.normalize_depth(
                depth_map, min_percentile=min_percentile, max_percentile=max_percentile
            )
            
            # Save individual frame if requested
            if save_frames:
                frame_filename = os.path.join(output_dir, f"depth_frame_{frame_count:05d}.png")
                cv2.imwrite(frame_filename, depth_colored)
                
                # Also save raw depth map as numpy array
                depth_npy_path = os.path.join(output_dir, f"depth_frame_{frame_count:05d}.npy")
                np.save(depth_npy_path, depth_map)
            
            # Write to video if saving video
            if save_video:
                video_writer.write(depth_colored)
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"  Processed {processed_count} frames...")
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    
    print(f"\n✓ Processing complete!")
    print(f"  Processed {processed_count} frames")
    print(f"  Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="MiDaS Depth Estimation on Video")
    parser.add_argument("--video", type=str, 
                       default="Raw Data/barre_tripod.mov",
                       help="Path to input video file")
    parser.add_argument("--output", type=str,
                       default="midas_depth_results",
                       help="Output directory for depth maps")
    parser.add_argument("--model", type=str,
                       default="dpt_swin2_large_384.pt",
                       help="Path to local MiDaS model file (optional)")
    parser.add_argument("--model-type", type=str,
                       default="DPT_Large",
                       choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
                       help="Type of MiDaS model to use")
    parser.add_argument("--no-frames", action="store_true",
                       help="Don't save individual depth map frames")
    parser.add_argument("--no-video", action="store_true",
                       help="Don't save depth visualization video")
    parser.add_argument("--frame-interval", type=int, default=1,
                       help="Process every nth frame (default: 1 = all frames)")
    parser.add_argument("--min-percentile", type=float, default=5.0,
                       help="Minimum depth percentile for clipping close objects (default: 5.0)")
    parser.add_argument("--max-percentile", type=float, default=95.0,
                       help="Maximum depth percentile for clipping far objects (default: 95.0)")
    
    args = parser.parse_args()
    
    # Check if model file exists
    model_path = args.model if os.path.exists(args.model) else None
    
    # Process video
    process_video(
        video_path=args.video,
        output_dir=args.output,
        model_path=model_path,
        model_type=args.model_type,
        save_frames=not args.no_frames,
        save_video=not args.no_video,
        frame_interval=args.frame_interval,
        min_percentile=args.min_percentile,
        max_percentile=args.max_percentile
    )


if __name__ == "__main__":
    main()

