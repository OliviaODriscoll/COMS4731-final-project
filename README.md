# 3D Skeleton Reconstruction from Multi-View Video

A computer vision project for reconstructing 3D human skeletons from multi-view video data, with applications in dance and movement analysis.

## Overview

This project implements a complete pipeline for:
- **2D skeleton extraction** from single-view images using morphological operations
- **Depth estimation** using MiDaS (DPT-Swin2-Large) models
- **3D skeleton reconstruction** from multiple camera viewpoints
- **Pose classification** using HOG features and SVM ensembles
- **Temporal smoothing** using Markov models

## Features

### Core Functionality
- **Multi-camera calibration** and stereo vision depth estimation
- **Skeleton extraction** using color segmentation and morphological operations
- **3D reconstruction** from multiple viewpoints with triangulation
- **Depth-based 3D skeleton** using MediaPipe pose estimation
- **Pose classification** with HOG features and ensemble SVM models
- **Temporal smoothing** with Markov chain models

### Key Scripts

#### Core Scripts (`core_scripts/`)
- `skeletonize.py` - Extracts 2D skeletons from images using color segmentation and morphological operations
- `3d_skeleton_reconstruction.py` - Reconstructs 3D skeletons from multiple camera views using triangulation
- `midas_depth_estimation.py` - MiDaS depth estimation using DPT-Swin2-Large model
- `depth_based_3d_skeleton.py` - 3D skeleton reconstruction using depth maps and MediaPipe

#### Classification (`classification/`)
- `hog_classifier.py` - HOG feature extraction and classification
- `train_svm_concatenated_original.py` - SVM training with concatenated features
- `train_svm_ensemble_voting.py` - Ensemble SVM with voting
- `train_hog_ensemble_comparison.py` - HOG ensemble comparison and evaluation
- `markov_smoothing.py` - Markov chain-based temporal smoothing

#### Utilities (`utilities/`)
- `split_vid_to_frames.py` - Extract frames from video files
- `extract_frames_at_timestamp.py` - Extract specific frames at given timestamps
- `compute_homography_ransac.py` - Homography computation for multi-view alignment
- `depth_map_3cameras.py` - Multi-view depth reconstruction utilities

## Installation

### Prerequisites
- Python 3.10.8
- CUDA 11.7 (for GPU acceleration)
- PyTorch 1.13.0

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd COMS4731-final-project
   ```

2. **Create conda environment**
   ```bash
   conda env create -f environment.yaml
   conda activate midas-py310
   ```

3. **Download MiDaS model** (if not already present)
   - The model file `dpt_swin2_large_384.pt` should be in the project root
   - If missing, the script will attempt to download it automatically

4. **Install additional dependencies** (if needed)
   ```bash
   pip install mediapipe scikit-image scipy pandas scikit-learn matplotlib seaborn tqdm
   ```

## Usage

### Extract Frames from Video
```bash
python utilities/split_vid_to_frames.py --video_path path/to/video.mov --output_dir frames/
```

### Extract Frame at Specific Timestamp
```bash
python utilities/extract_frames_at_timestamp.py --video_path path/to/video.mov --timestamp 26.0 --output_path frame.png
```

### Skeleton Extraction
```bash
python core_scripts/skeletonize.py
```
Note: Update the frame path in the script before running.

### Depth Estimation
```bash
python core_scripts/midas_depth_estimation.py --input_path path/to/image.png --output_path depth_map.png
```

### 3D Skeleton Reconstruction
```bash
python core_scripts/3d_skeleton_reconstruction.py
```
Configure camera calibration paths and video paths in the script.

### Depth-Based 3D Skeleton
```bash
python core_scripts/depth_based_3d_skeleton.py --input_dir frames/ --output_dir results/
```

### Train Classification Models
```bash
# Train HOG ensemble
python classification/train_hog_ensemble_comparison.py --data_dir data/ --output_dir models/

# Train SVM ensemble
python classification/train_svm_ensemble_voting.py --data_dir data/ --output_dir models/
```

## Project Structure

```
COMS4731-final-project/
├── README.md                          # This file
├── environment.yaml                   # Conda environment configuration
├── .gitignore                         # Git ignore rules
├── file_mapping.txt                   # Video file mapping
│
├── core_scripts/                      # Core skeleton reconstruction scripts
│   ├── __init__.py
│   ├── skeletonize.py                 # 2D skeleton extraction
│   ├── 3d_skeleton_reconstruction.py  # Multi-view 3D reconstruction
│   ├── depth_based_3d_skeleton.py     # Depth-based 3D skeleton
│   └── midas_depth_estimation.py      # MiDaS depth estimation
│
├── classification/                    # Classification and ML scripts
│   ├── __init__.py
│   ├── hog_classifier.py              # HOG feature extraction
│   ├── train_hog_ensemble_comparison.py
│   ├── train_svm_concatenated_original.py
│   ├── train_svm_ensemble_voting.py
│   └── markov_smoothing.py            # Temporal smoothing
│
├── utilities/                         # Utility scripts
│   ├── __init__.py
│   ├── split_vid_to_frames.py         # Video frame extraction
│   ├── extract_frames_at_timestamp.py
│   ├── compute_homography_ransac.py   # Multi-view alignment
│   └── depth_map_3cameras.py          # Multi-view depth utilities
│
├── dpt_swin2_large_384.pt            # MiDaS depth model
│
└── archive/                           # Archived data and results
    ├── Raw Data/                      # Original video files
    ├── frames/                        # Extracted frames
    ├── depth_results/                 # Depth estimation results
    ├── visualizations/                # Processing visualizations
    └── reconstruction_frames/         # Sample reconstruction frames
```

## Dependencies

### Core Libraries
- **OpenCV** (4.6.0.66) - Computer vision operations
- **NumPy** (1.23.4) - Numerical computations
- **PyTorch** (1.13.0) - Deep learning framework
- **scikit-image** - Image processing and morphology
- **scikit-learn** - Machine learning models
- **MediaPipe** - Pose estimation (optional)

### Additional Libraries
- **pandas** - Data manipulation
- **matplotlib** - Visualization
- **scipy** - Scientific computing
- **timm** (0.6.12) - PyTorch image models
- **einops** (0.6.0) - Tensor operations

## Data Organization

- **Raw videos**: Original `.mov` files (see `archive/Raw Data/`)
- **Frames**: Extracted frames organized by video name
- **Calibration**: Camera calibration videos and images
- **Results**: Depth maps, skeleton visualizations, and 3D reconstructions

## Notes

- The project uses a multi-camera setup (left, right, tripod/center views)
- Calibration videos are required for camera parameter estimation
- Large video files and intermediate results are stored in `archive/`
- Model weights (`dpt_swin2_large_384.pt`) are required for depth estimation


## Authors

Olivia O'Driscoll, Jungyun Kim, Trinity Suma
