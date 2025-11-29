## Ballet Multi-View Skeleton Dataset

This `dataset/` directory contains train/val/test splits of synchronized multi-view ballet recordings, with 2D and 3D skeletons and per-frame labels. It is designed to be plug-and-play with the training and visualization scripts in this repo.

### Structure

- **`train/`, `val/`, `test/`**
  - **`2d_skeletons/`**
    - **`left/`, `middle/`, `right/`**: 2D MediaPipe pose landmarks for each camera view  
      - Files: `skeleton_2d_coords_frame_00000.npy`, `skeleton_2d_coords_frame_00001.npy`, ...  
      - Each `.npy` is a NumPy array of shape `(num_joints, 2 or 3)`:
        - 2D: `(x, y)` in pixels or normalized coordinates (handled by the visualization code)
        - 3D: if present, the third coordinate is depth but for 2D we usually ignore it
  - **`3d_skeletons/`**
    - Triangulated 3D pose for each synchronized frame across the three views  
      - Files: `skeleton_3d_coords_frame_00000.npy`, ...  
      - Each `.npy` is a NumPy array of shape `(num_joints, 3)` in a common 3D coordinate system.
  - **`labels.csv`**
    - Timestamp-based labels for this split, derived from `arabesque_left.csv`  
    - Columns:
      - `step`: ballet movement label (e.g., `arabesque`, `passe`, `tendu`, `sous sous`, ...)
      - `start`: start time in `MM:SS.s` (minutes:seconds) in the reference video
      - `end`: end time in `MM:SS.s` in the reference video
    - Used mainly for diagnostics; the training code typically uses per-frame labels (see below).
  - **`frame_labels.csv`**
    - Per-frame labels for every dataset frame in this split. This is the main label file to use.
    - Columns:
      - `frame`: dataset frame index (starts at 0 in each split)
      - `label`: label for this frame (`no_label` if between annotated segments)
      - `reference_timestamp`: timestamp in the reference video corresponding to this frame

### Frame Numbering and Mapping

- The dataset is aligned to the **`arabesque_left`** video as the reference.
- Alignment point (where all three cameras are synchronized):
  - `arabesque_left` original frame **833** is defined to be **dataset frame 0**.
- Mapping between dataset frames and original `arabesque_left` video frames:
  - **original_frame = dataset_frame + 833**
  - The corresponding video time is:
    - `time_seconds = original_frame / fps` (fps ≈ 29.97 for `arabesque_left`)
  - This mapping is already reflected in `frame_labels.csv` via the `reference_timestamp` column.

  ## Multi-View Video Offsets

- At the synchronization point (dataset frame 0), the three raw videos align at:
  - `arabesque_left` frame **833**
  - `barre_right` frame **1071**
  - `barre_tripod` (middle) frame **1218**
- For any dataset frame `f_dataset`, the corresponding raw video frames are:
  - **arabesque_left**: `frame_arabesque_left = f_dataset + 833`
  - **barre_right**: `frame_barre_right = f_dataset + 1071`
  - **barre_tripod**: `frame_barre_tripod = f_dataset + 1218`
- To convert a raw frame index to time in its video:
  - `time_seconds = frame_index / fps_video`

### Using the Dataset in Code

#### 1. Training a Classifier

Use `train_on_dataset.py`, which wraps the `BalletSequenceDataset` and model definitions in `ballet_classifier.py`.

Example (3D skeletons, LSTM):

```bash
cd /Users/olivia/Desktop/COMS4731-final-project

python3 train_on_dataset.py \
  --dataset-dir dataset \
  --use-3d \
  --model-type lstm \
  --sequence-length 30 \
  --batch-size 32 \
  --num-epochs 50 \
  --augment \
  --learning-rate 0.001 \
  --hidden-size 128 \
  --num-layers 2 \
  --output-dir classifier_output \
  --model-name ballet_classifier_3d
```

Example (2D skeletons, left camera only):

```bash
python3 train_on_dataset.py \
  --dataset-dir dataset \
  --use-2d \
  --camera-view left \
  --model-type lstm \
  --sequence-length 30 \
  --batch-size 32
```

`BalletSequenceDataset`:
- Reads `3d_skeletons/` **or** a specific `2d_skeletons/<view>/` directory
- Uses `labels.csv` or `frame_labels.csv` to assign a label to each sequence (via sliding windows)
- Normalizes coordinates and optionally applies simple augmentation

#### 2. Visualizing Skeletons and Labels

- **2D skeletons (all 3 views + labels)**:

  ```bash
  python3 visualize_2d_skeletons_with_labels.py \
    --dataset-dir dataset \
    --split train \
    --num-samples 8 \
    --output-dir dataset/skeleton_visualizations
  ```

  Produces `train_2d_skeletons_with_labels.png`.

- **3D skeletons + labels**:

  ```bash
  python3 visualize_skeletons_with_labels.py \
    --dataset-dir dataset \
    --split train \
    --num-samples 12 \
    --output-dir dataset/skeleton_visualizations
  ```

  Produces `train_skeletons_with_labels.png`.

#### 3. Visualizing Label Clusters (2D and 3D)

- **2D middle-view skeleton clusters (PCA)**:

  ```bash
  python3 visualize_label_clustering_2d.py \
    --dataset-dir dataset \
    --split train \
    --view middle \
    --anchor-joint 23 \
    --max-per-label 500 \
    --output-path dataset/skeleton_visualizations/train_2d_label_clusters_middle.png
  ```

- **3D skeleton clusters (PCA in 2D or 3D)**:

  ```bash
  # 2 principal components (2D scatter)
  python3 visualize_label_clustering_3d.py \
    --dataset-dir dataset \
    --split train \
    --anchor-joint 23 \
    --max-per-label 500 \
    --num-pcs 2 \
    --output-path dataset/skeleton_visualizations/train_3d_label_clusters.png

  # 3 principal components (3D scatter)
  python3 visualize_label_clustering_3d.py \
    --dataset-dir dataset \
    --split train \
    --anchor-joint 23 \
    --max-per-label 500 \
    --num-pcs 3 \
    --output-path dataset/skeleton_visualizations/train_3d_label_clusters_3pc.png
  ```

Both clustering scripts:
- Collect all labeled frames (ignoring `no_label`)
- Align skeletons on a chosen joint (default 23 = left hip)
- Normalize scale
- Run PCA and plot clusters colored by label

### Key Assumptions / Notes

- Reference camera for timing and labels is **`arabesque_left`**.
- FPS used for mapping timestamps to frames is approximately **29.97** (read from the video file).
- Some frames have label `no_label` where the dancer is between annotated steps; these are still present as skeletons but are treated as unlabeled.
- Joints follow the MediaPipe Pose indexing (33 landmarks) for both 2D and 3D skeletons.

If you just want to plug this dataset into your own model:
- Treat each `.npy` in `3d_skeletons/` as a frame-level feature `(num_joints, 3)`.
- Use `frame_labels.csv` to map frame indices to the `label` you care about.
- For sequence models, create sliding windows over frame indices and assign each window the majority (or center) label.


