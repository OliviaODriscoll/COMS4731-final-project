import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import time


class MiDASDepthEstimator:
    def __init__(self, model_type='DPT_Large', device=None):
        """
        Initialize MiDAS depth estimation model
        
        Args:
            model_type: Type of MiDAS model to use
                - 'DPT_Large': Large DPT model (best quality, slower)
                - 'DPT_Hybrid': Hybrid DPT model (balanced)
                - 'MiDaS_small': Small model (fastest, lower quality)
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        print(f"\n=== Initializing MiDAS Depth Estimator ===")
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load MiDAS model from torch hub
        print(f"Loading MiDAS model: {model_type}...")
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", model_type)
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying alternative loading method...")
            # Alternative: try loading from transformers
            try:
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation
                self.processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
                self.model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large")
                self.model.to(self.device)
                self.model.eval()
                self.use_transformers = True
                print(f"✓ Model loaded via transformers")
            except Exception as e2:
                print(f"Error with transformers: {e2}")
                raise RuntimeError("Could not load MiDAS model. Please install: pip install torch torchvision")
        
        # Image transformation pipeline
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        
        print(f"✓ MiDAS Depth Estimator initialized")
    
    def estimate_depth(self, image):
        """
        Estimate depth map from a single image
        
        Args:
            image: Input image (numpy array, BGR format from OpenCV)
        
        Returns:
            depth_map: Depth map as numpy array (normalized, higher values = closer)
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Apply transformation
        input_tensor = self.transform(image_rgb).to(self.device)
        
        # Predict depth
        with torch.no_grad():
            prediction = self.model(input_tensor)
            
            # Post-process prediction
            if isinstance(prediction, dict):
                prediction = prediction['predicted_depth']
            
            # Convert to numpy
            depth = prediction.cpu().numpy()
            
            # Resize to original image size if needed
            if depth.shape != image.shape[:2]:
                depth = cv2.resize(depth[0], (image.shape[1], image.shape[0]), 
                                 interpolation=cv2.INTER_CUBIC)
            else:
                depth = depth[0]
        
        return depth
    
    def process_video(self, video_path, output_dir, frame_interval=30, 
                     max_frames=None, start_frame=0):
        """
        Process a video and estimate depth for frames
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save depth maps
            frame_interval: Process every Nth frame (default: 30)
            max_frames: Maximum number of frames to process (None for all)
            start_frame: Frame number to start from
        
        Returns:
            List of saved depth map paths
        """
        print(f"\n=== Processing Video: {os.path.basename(video_path)} ===")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Processing every {frame_interval} frame(s)")
        if max_frames:
            print(f"  Max frames to process: {max_frames}")
        
        # Set starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        saved_paths = []
        frame_count = start_frame
        processed_count = 0
        
        print(f"\nProcessing frames...")
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame if it matches interval
            if frame_count % frame_interval == 0:
                print(f"  Processing frame {frame_count}...", end=" ", flush=True)
                frame_start = time.time()
                
                # Estimate depth
                depth_map = self.estimate_depth(frame)
                
                # Save depth map
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                depth_path = os.path.join(output_dir, f"{video_name}_depth_frame_{frame_count:05d}.npy")
                np.save(depth_path, depth_map)
                
                # Save visualization
                vis_path = os.path.join(output_dir, f"{video_name}_depth_frame_{frame_count:05d}.png")
                self.save_depth_visualization(depth_map, frame, vis_path)
                
                saved_paths.append(depth_path)
                processed_count += 1
                
                elapsed = time.time() - frame_start
                print(f"✓ ({elapsed:.2f}s)")
                
                # Check max frames limit
                if max_frames and processed_count >= max_frames:
                    print(f"\nReached max frames limit ({max_frames})")
                    break
            
            frame_count += 1
        
        cap.release()
        
        total_time = time.time() - start_time
        print(f"\n✓ Processing complete!")
        print(f"  Processed {processed_count} frames in {total_time:.2f}s")
        print(f"  Average: {total_time/processed_count:.2f}s per frame")
        print(f"  Results saved to: {output_dir}")
        
        return saved_paths
    
    def save_depth_visualization(self, depth_map, original_image, save_path):
        """
        Save depth map visualization alongside original image
        
        Args:
            depth_map: Depth map (numpy array)
            original_image: Original image (BGR format)
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Depth map (normalized)
        depth_normalized = depth_map.copy()
        # Invert so closer objects are brighter (MiDAS outputs inverse depth)
        depth_normalized = 1.0 / (depth_normalized + 1e-6)
        depth_normalized = (depth_normalized - depth_normalized.min()) / (depth_normalized.max() - depth_normalized.min() + 1e-6)
        
        im1 = axes[1].imshow(depth_normalized, cmap='jet')
        axes[1].set_title('Depth Map (Normalized)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], label='Depth (normalized)')
        
        # Depth map (raw)
        im2 = axes[2].imshow(depth_map, cmap='viridis')
        axes[2].set_title('Depth Map (Raw)')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], label='Depth (raw)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def process_single_frame(self, image_path, output_path=None):
        """
        Process a single image and estimate depth
        
        Args:
            image_path: Path to input image
            output_path: Path to save depth map (optional)
        
        Returns:
            depth_map: Depth map as numpy array
        """
        print(f"\n=== Processing Single Image ===")
        print(f"Image: {os.path.basename(image_path)}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"Cannot load image: {image_path}")
        
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Estimate depth
        print("Estimating depth...")
        start_time = time.time()
        depth_map = self.estimate_depth(image)
        elapsed = time.time() - start_time
        
        print(f"✓ Depth estimation complete ({elapsed:.2f}s)")
        print(f"  Depth map shape: {depth_map.shape}")
        print(f"  Depth range: {depth_map.min():.4f} - {depth_map.max():.4f}")
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            np.save(output_path, depth_map)
            
            # Save visualization
            vis_path = output_path.replace('.npy', '_visualization.png')
            self.save_depth_visualization(depth_map, image, vis_path)
            print(f"  Saved to: {output_path}")
            print(f"  Visualization: {vis_path}")
        
        return depth_map


def main():
    """
    Main function to run MiDAS depth estimation on a video
    """
    # ============================================================
    # Configuration
    # ============================================================
    
    # Video to process
    video_path = '/Users/olivia/Documents/COMS4731/COMS4731-final-project/Raw Data/barre_tripod.mov'
    
    # Output directory
    output_dir = '/Users/olivia/Documents/COMS4731/COMS4731-final-project/midas_depth_results'
    
    # Processing parameters
    frame_interval = 30  # Process every Nth frame (30 = ~1 frame per second at 30fps)
    max_frames = 10  # Limit number of frames to process (None for all)
    start_frame = 0  # Frame to start from
    
    # Model selection
    model_type = 'DPT_Large'  # Options: 'DPT_Large', 'DPT_Hybrid', 'MiDaS_small'
    
    # ============================================================
    # Run depth estimation
    # ============================================================
    
    print("="*60)
    print("MiDAS Depth Estimation")
    print("="*60)
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        print("\nAvailable videos in Raw Data:")
        raw_data_dir = '/Users/olivia/Documents/COMS4731/COMS4731-final-project/Raw Data'
        if os.path.exists(raw_data_dir):
            for f in os.listdir(raw_data_dir):
                if f.endswith(('.mov', '.mp4', '.avi')):
                    print(f"  - {f}")
        return
    
    # Initialize depth estimator
    try:
        estimator = MiDASDepthEstimator(model_type=model_type)
    except Exception as e:
        print(f"ERROR: Failed to initialize MiDAS: {e}")
        print("\nPlease install required packages:")
        print("  pip install torch torchvision")
        return
    
    # Process video
    try:
        saved_paths = estimator.process_video(
            video_path=video_path,
            output_dir=output_dir,
            frame_interval=frame_interval,
            max_frames=max_frames,
            start_frame=start_frame
        )
        
        print(f"\n✓ Successfully processed {len(saved_paths)} frames")
        print(f"  Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"ERROR: Failed to process video: {e}")
        import traceback
        traceback.print_exc()


def process_single_image_example():
    """
    Example: Process a single image instead of a video
    """
    # Initialize estimator
    estimator = MiDASDepthEstimator(model_type='DPT_Large')
    
    # Process single image
    image_path = '/Users/olivia/Documents/COMS4731/COMS4731-final-project/frames/reconstruction_frames/barre_tripod_frame_39s.png'
    output_path = '/Users/olivia/Documents/COMS4731/COMS4731-final-project/midas_depth_results/single_frame_depth.npy'
    
    if os.path.exists(image_path):
        depth_map = estimator.process_single_frame(image_path, output_path)
        print(f"\n✓ Depth estimation complete!")
    else:
        print(f"Image not found: {image_path}")


if __name__ == "__main__":
    import sys
    
    # Check if user wants to process a single image
    if len(sys.argv) > 1 and sys.argv[1] == '--single':
        process_single_image_example()
    else:
        main()

