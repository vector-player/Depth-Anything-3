# 📚 Depth-Anything-3 Usage Guide

A comprehensive guide to using Depth-Anything-3 for depth estimation, camera pose estimation, and 3D reconstruction.

## 📋 Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Python API](#python-api)
4. [Command-Line Interface (CLI)](#command-line-interface-cli)
5. [Model Selection](#model-selection)
6. [Parameters Reference](#parameters-reference)
7. [Export Formats](#export-formats)
8. [Examples](#examples)
9. [Advanced Features](#advanced-features)

---

## 🚀 <span id="installation">Installation</span>

### Prerequisites
- Python 3.9 - 3.13
- CUDA-capable GPU (recommended)
- PyTorch >= 2.0

### Install Depth-Anything-3

```bash
# Install core dependencies
pip install xformers torch>=2 torchvision

# Install Depth-Anything-3 (basic)
pip install -e .

# Install with Gradio app support (requires Python >= 3.10)
pip install -e ".[app]"

# Install with all features including Gaussian Splatting
pip install -e ".[all]"

# Install gsplat for Gaussian Splatting features (optional)
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
```

### Using Conda Environment

```bash
# Create conda environment
conda create -n da3 python=3.10 -y
conda activate da3

# Install dependencies
pip install xformers torch>=2 torchvision
pip install -e .
pip install -e ".[app]"
```

---

## ⚡ <span id="quick-start">Quick Start</span>

### Python API - Basic Usage

```python
import glob
import torch
from depth_anything_3.api import DepthAnything3

# Initialize model
device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)

# Load images
images = sorted(glob.glob("path/to/images/*.png"))

# Run inference
prediction = model.inference(images)

# Access results
print(f"Depth shape: {prediction.depth.shape}")        # (N, H, W)
print(f"Confidence shape: {prediction.conf.shape}")     # (N, H, W)
print(f"Extrinsics shape: {prediction.extrinsics.shape}")  # (N, 4, 4)
print(f"Intrinsics shape: {prediction.intrinsics.shape}")   # (N, 3, 3)
```

### Command-Line Interface - Basic Usage

```bash
# Process a single image
da3 image path/to/image.jpg --export-dir ./output

# Process a video
da3 video path/to/video.mp4 --export-dir ./output

# Process a directory of images
da3 images path/to/images/ --export-dir ./output

# Auto-detect input type
da3 auto path/to/input --export-dir ./output
```

---

## 🐍 <span id="python-api">Python API</span>

### Initialization

```python
from depth_anything_3.api import DepthAnything3
import torch

# Method 1: Load from Hugging Face Hub
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to("cuda")

# Method 2: Initialize with model name
model = DepthAnything3(model_name="da3-large")
model = model.to("cuda")
```

### Inference Method

```python
prediction = model.inference(
    image,                          # Required: List of images
    extrinsics=None,                 # Optional: Camera extrinsics (N, 4, 4)
    intrinsics=None,                 # Optional: Camera intrinsics (N, 3, 3)
    align_to_input_ext_scale=True,   # Align to input pose scale
    infer_gs=False,                  # Enable Gaussian Splatting
    use_ray_pose=False,              # Use ray-based pose estimation
    ref_view_strategy="saddle_balanced",  # Reference view selection
    render_exts=None,                # Render extrinsics for gs_video
    render_ixts=None,                # Render intrinsics for gs_video
    render_hw=None,                  # Render resolution for gs_video
    process_res=504,                 # Processing resolution
    process_res_method="upper_bound_resize",  # Resize method
    export_dir=None,                 # Export directory
    export_format="mini_npz",       # Export format
    export_feat_layers=None,         # Feature layers to export
    conf_thresh_percentile=40.0,    # GLB confidence threshold
    num_max_points=1_000_000,       # GLB max points
    show_cameras=True,               # GLB show cameras
    feat_vis_fps=15,                 # Feature vis FPS
    export_kwargs={}                 # Additional export arguments
)
```

### Input Types

The `image` parameter accepts:
- **File paths**: `["image1.jpg", "image2.png"]`
- **NumPy arrays**: `[np.array(img1), np.array(img2)]`
- **PIL Images**: `[Image.open("img1.jpg"), Image.open("img2.jpg")]`

### Prediction Object

The `inference()` method returns a `Prediction` object with:

```python
prediction.depth              # np.ndarray: (N, H, W) - Depth maps
prediction.conf               # np.ndarray: (N, H, W) - Confidence maps
prediction.extrinsics         # np.ndarray: (N, 4, 4) - Camera extrinsics
prediction.intrinsics         # np.ndarray: (N, 3, 3) - Camera intrinsics
prediction.processed_images   # np.ndarray: (N, H, W, 3) - Processed images
prediction.sky                # np.ndarray: (N, H, W) - Sky segmentation mask
prediction.aux                # dict - Auxiliary outputs (features, etc.)
prediction.gaussians          # Gaussians object - 3DGS data (if infer_gs=True)
prediction.is_metric          # int - Whether depth is metric
prediction.scale_factor       # float - Metric scale factor
```

---

## 💻 <span id="command-line-interface-cli">Command-Line Interface (CLI)</span>

### Available Commands

#### `da3 auto` - Auto Mode (Recommended)

Automatically detects input type and processes accordingly.

```bash
da3 auto INPUT_PATH [OPTIONS]
```

**Input Types Supported:**
- Single image file (.jpg, .png, .jpeg, .webp, .bmp, .tiff, .tif)
- Image directory
- Video file (.mp4, .avi, .mov, .mkv, .flv, .wmv, .webm, .m4v)
- COLMAP directory (containing `images/` and `sparse/` subdirectories)

**Example:**
```bash
da3 auto path/to/input --export-dir ./output --export-format glb
```

#### `da3 image` - Single Image Processing

```bash
da3 image IMAGE_PATH [OPTIONS]
```

**Example:**
```bash
da3 image photo.jpg --export-dir ./output --export-format mini_npz-glb
```

#### `da3 images` - Image Directory Processing

```bash
da3 images IMAGES_DIR [OPTIONS]
```

**Example:**
```bash
da3 images ./dataset --export-dir ./output --image-extensions "png,jpg,webp"
```

#### `da3 video` - Video Processing

```bash
da3 video VIDEO_PATH [OPTIONS]
```

**Example:**
```bash
da3 video video.mp4 --fps 2.0 --export-dir ./output
```

#### `da3 colmap` - COLMAP Dataset Processing

```bash
da3 colmap COLMAP_DIR [OPTIONS]
```

**Example:**
```bash
da3 colmap ./colmap_data --sparse-subdir 0 --export-dir ./output
```

#### `da3 backend` - Backend Service

Start a backend service to keep the model in GPU memory for faster processing.

```bash
da3 backend [OPTIONS]
```

**Example:**
```bash
da3 backend --model-dir depth-anything/DA3NESTED-GIANT-LARGE --host 0.0.0.0 --port 8008
```

Then use `--use-backend` flag in other commands:
```bash
da3 auto input.jpg --use-backend --backend-url http://localhost:8008
```

#### `da3 gradio` - Gradio Web Application

Launch an interactive web interface.

```bash
da3 gradio --model-dir MODEL_DIR --workspace-dir WORKSPACE_DIR --gallery-dir GALLERY_DIR
```

#### `da3 gallery` - Gallery Server

Launch a gallery server to view results.

```bash
da3 gallery --gallery-dir ./workspace
```

### Use Case: Generate K-Matrix (Camera Intrinsics) from Image

DA3 can automatically estimate camera intrinsics (K-matrix) from a single image using the CLI. This is useful when you need camera calibration parameters but don't have access to a calibration pattern.

#### Method 1: Using `da3 image` with `mini_npz` Export

```bash
# Process a single image and export intrinsics
da3 image path/to/image.jpg \
    --export-dir ./output \
    --export-format mini_npz

# Extract K-matrix from the NPZ file
python -c "
import numpy as np
data = np.load('./output/exports/mini_npz/results.npz')
K = data['intrinsics'][0]  # Get first image's K-matrix
np.savetxt('K.txt', K, fmt='%.6f')
print('K-matrix saved to K.txt')
print(f'K-matrix:\n{K}')
"
```

#### Method 2: Using `da3 auto` for Multiple Images

```bash
# Process a directory of images
da3 auto ./images/ \
    --export-dir ./output \
    --export-format mini_npz

# Extract K-matrix for a specific image (e.g., first image)
python -c "
import numpy as np
data = np.load('./output/exports/mini_npz/results.npz')
K = data['intrinsics'][0]  # First image's K-matrix
np.savetxt('K.txt', K, fmt='%.6f')
print(f'K-matrix for first image:\n{K}')
"
```

#### Method 3: Batch Extract K-Matrices for All Images

```bash
# Process images and export intrinsics
da3 images ./dataset/ \
    --export-dir ./output \
    --export-format mini_npz

# Extract K-matrices for all images
python -c "
import numpy as np
import os

data = np.load('./output/exports/mini_npz/results.npz')
intrinsics = data['intrinsics']  # All K-matrices

# Save each K-matrix
os.makedirs('./K_matrices', exist_ok=True)
for i, K in enumerate(intrinsics):
    np.savetxt(f'./K_matrices/K_{i:04d}.txt', K, fmt='%.6f')
    print(f'Saved K-matrix {i} to K_matrices/K_{i:04d}.txt')
print(f'Total: {len(intrinsics)} K-matrices extracted')
"
```

#### Understanding the K-Matrix Output

The K-matrix is a 3×3 camera intrinsic matrix with the following format:

```
[[fx,  0, cx],
 [0,  fy, cy],
 [0,   0,  1]]
```

Where:
- **fx, fy**: Focal lengths in pixels (X and Y directions)
- **cx, cy**: Principal point coordinates (image center) in pixels

#### Use Cases

- **3D Reconstruction**: Use K-matrix with depth maps to generate accurate point clouds
- **SLAM/Tracking**: Provide camera intrinsics for visual SLAM systems (e.g., pysrt3d, ORB-SLAM)
- **Camera Calibration**: Estimate camera parameters when calibration patterns are unavailable
- **Multi-View Stereo**: Use intrinsics for pose-conditioned depth estimation
- **AR/VR Applications**: Provide camera parameters for augmented reality applications

#### Tips

- The estimated K-matrix is automatically computed by DA3 during inference
- For best results, use images with good lighting and clear features
- The K-matrix values are in pixels, so they depend on the image resolution
- You can use the `--process-res` parameter to control the processing resolution

---

## 🎯 <span id="model-selection">Model Selection</span>

### Available Models

| Model Name | Parameters | Features | License |
|------------|-----------|----------|---------|
| **DA3NESTED-GIANT-LARGE-1.1** | 1.40B | All features + Metric depth | CC BY-NC 4.0 |
| **DA3NESTED-GIANT-LARGE** | 1.40B | All features + Metric depth | CC BY-NC 4.0 |
| **DA3-GIANT-1.1** | 1.15B | Any-view + GS | CC BY-NC 4.0 |
| **DA3-GIANT** | 1.15B | Any-view + GS | CC BY-NC 4.0 |
| **DA3-LARGE-1.1** | 0.35B | Any-view | CC BY-NC 4.0 |
| **DA3-LARGE** | 0.35B | Any-view | CC BY-NC 4.0 |
| **DA3-BASE** | 0.12B | Any-view | Apache 2.0 |
| **DA3-SMALL** | 0.08B | Any-view | Apache 2.0 |
| **DA3METRIC-LARGE** | 0.35B | Metric depth + Sky seg | Apache 2.0 |
| **DA3MONO-LARGE** | 0.35B | Monocular depth + Sky seg | Apache 2.0 |

**Note:** Models with `-1.1` suffix are retrained versions with better performance, especially for street scenes.

### Model Features

- **Any-view Model**: Monocular depth, multi-view depth, pose estimation, pose-conditioned depth
- **GS Support**: 3D Gaussian Splatting (only Giant models)
- **Metric Depth**: Real-world scale depth estimation
- **Sky Segmentation**: Sky mask prediction

### Choosing a Model

- **For general use**: `DA3-LARGE-1.1` or `DA3-BASE`
- **For best quality**: `DA3NESTED-GIANT-LARGE-1.1`
- **For metric depth**: `DA3NESTED-GIANT-LARGE-1.1` or `DA3METRIC-LARGE`
- **For Gaussian Splatting**: `DA3-GIANT-1.1` or `DA3NESTED-GIANT-LARGE-1.1`
- **For lightweight**: `DA3-SMALL` or `DA3-BASE`

---

## ⚙️ <span id="parameters-reference">Parameters Reference</span>

### Input Parameters

#### `image` (required)
- **Type**: `List[Union[np.ndarray, Image.Image, str]]`
- **Description**: List of input images
- **Example**: `["img1.jpg", "img2.png"]` or `[np.array(img1), np.array(img2)]`

#### `extrinsics` (optional)
- **Type**: `Optional[np.ndarray]`
- **Shape**: `(N, 4, 4)`
- **Description**: Camera extrinsic matrices (world-to-camera transformation)
- **Use case**: Pose-conditioned depth estimation

#### `intrinsics` (optional)
- **Type**: `Optional[np.ndarray]`
- **Shape**: `(N, 3, 3)`
- **Description**: Camera intrinsic matrices (focal length, principal point)
- **Use case**: Pose-conditioned depth estimation

### Processing Parameters

#### `process_res` (default: 504)
- **Type**: `int`
- **Description**: Base resolution for processing
- **Note**: Higher values improve quality but increase computation time

#### `process_res_method` (default: "upper_bound_resize")
- **Type**: `str`
- **Options**: `"upper_bound_resize"`, `"lower_bound_resize"`
- **Description**: Method for resizing images
  - `upper_bound_resize`: Resize so that the longer side equals `process_res`
  - `lower_bound_resize`: Resize so that the shorter side equals `process_res`

### Pose Estimation Parameters

#### `align_to_input_ext_scale` (default: True)
- **Type**: `bool`
- **Description**: When True, replace predicted extrinsics with input ones and rescale depth to match metric scale

#### `use_ray_pose` (default: False)
- **Type**: `bool`
- **Description**: Use ray-based pose estimation instead of camera decoder
- **Note**: Generally more accurate but slightly slower

#### `ref_view_strategy` (default: "saddle_balanced")
- **Type**: `str`
- **Options**: `"first"`, `"middle"`, `"saddle_balanced"`, `"saddle_sim_range"`
- **Description**: Strategy for selecting reference view from multiple views
  - `saddle_balanced`: Balanced selection (recommended)
  - `saddle_sim_range`: Largest similarity range
  - `middle`: Middle view (good for video sequences)
  - `first`: First view (not recommended)

### Gaussian Splatting Parameters

#### `infer_gs` (default: False)
- **Type**: `bool`
- **Description**: Enable 3D Gaussian Splatting branch
- **Required**: For `gs_ply` and `gs_video` export formats
- **Note**: Only supported by Giant models

#### `render_exts` (optional)
- **Type**: `Optional[np.ndarray]`
- **Shape**: `(M, 4, 4)`
- **Description**: Camera extrinsics for novel view rendering (gs_video)

#### `render_ixts` (optional)
- **Type**: `Optional[np.ndarray]`
- **Shape**: `(M, 3, 3)`
- **Description**: Camera intrinsics for novel view rendering (gs_video)

#### `render_hw` (optional)
- **Type**: `Optional[Tuple[int, int]]`
- **Description**: Output resolution `(height, width)` for rendered frames

### Export Parameters

#### `export_dir` (optional)
- **Type**: `Optional[str]`
- **Description**: Directory to save exported files
- **Note**: If None, no files are exported

#### `export_format` (default: "mini_npz")
- **Type**: `str`
- **Description**: Export format(s), can combine multiple with hyphens
- **Options**: See [Export Formats](#export-formats) section

#### `export_feat_layers` (optional)
- **Type**: `Optional[List[int]]`
- **Description**: Layer indices to export intermediate features
- **Example**: `[0, 5, 10, 15, 20]`

### GLB Export Parameters

#### `conf_thresh_percentile` (default: 40.0)
- **Type**: `float`
- **Description**: Lower percentile for adaptive confidence threshold
- **Note**: Points below this percentile are filtered out

#### `num_max_points` (default: 1,000,000)
- **Type**: `int`
- **Description**: Maximum number of points in point cloud
- **Note**: Point cloud is downsampled if exceeded

#### `show_cameras` (default: True)
- **Type**: `bool`
- **Description**: Include camera wireframes in GLB export

### Feature Visualization Parameters

#### `feat_vis_fps` (default: 15)
- **Type**: `int`
- **Description**: Frame rate for feature visualization video

### Export Keyword Arguments

#### `export_kwargs` (default: {})
- **Type**: `dict[str, dict[str, Any]]`
- **Description**: Additional arguments for specific export formats
- **Example**:
  ```python
  export_kwargs = {
      "gs_ply": {
          "gs_views_interval": 1,
      },
      "gs_video": {
          "trj_mode": "interpolate_smooth",
          "chunk_size": 1,
          "vis_depth": None,
      },
  }
  ```

---

## 📤 <span id="export-formats">Export Formats</span>

### `mini_npz`
- **Description**: Minimal NPZ format with essential data
- **Contents**: `depth`, `conf`, `exts`, `ixts`
- **Use case**: Lightweight storage

### `npz`
- **Description**: Full NPZ format with comprehensive data
- **Contents**: `depth`, `conf`, `exts`, `ixts`, `image`, etc.
- **Use case**: Complete data export

### `glb`
- **Description**: 3D visualization format (glTF binary)
- **Contents**: Point cloud, camera poses, colors
- **Use case**: 3D visualization and inspection
- **Features**: Confidence filtering, downsampling, sky handling

### `gs_ply`
- **Description**: Gaussian Splatting point cloud (PLY format)
- **Requirements**: `infer_gs=True`, Giant model
- **Use case**: 3DGS reconstruction
- **Viewers**: [SuperSplat](https://superspl.at/editor), [SPARK](https://sparkjs.dev/viewer/)

### `gs_video`
- **Description**: Rasterized 3DGS video
- **Requirements**: `infer_gs=True`, Giant model
- **Use case**: Novel view synthesis video
- **Note**: Can use `render_exts`, `render_ixts`, `render_hw` for custom viewpoints

### `feat_vis`
- **Description**: Feature visualization video
- **Requirements**: `export_feat_layers` must be specified
- **Use case**: Model interpretability

### `depth_vis`
- **Description**: Depth visualization images
- **Contents**: Color-coded depth maps
- **Use case**: Visual inspection

### `colmap`
- **Description**: COLMAP format export
- **Requirements**: Input must be image paths
- **Use case**: Integration with COLMAP pipeline

### Combining Formats

Multiple formats can be combined with hyphens:
```python
export_format = "mini_npz-glb-feat_vis"
```

---

## 💡 <span id="examples">Examples</span>

### Example 1: Basic Depth Estimation

```python
from depth_anything_3.api import DepthAnything3
import torch

model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1")
model = model.to("cuda")

prediction = model.inference(
    image=["image1.jpg", "image2.jpg"],
    export_dir="./output",
    export_format="glb"
)

print(f"Depth maps: {prediction.depth.shape}")
print(f"Camera poses: {prediction.extrinsics.shape}")
print(f"Camera intrinsics: {prediction.intrinsics.shape}")
```

### Example 2: Pose-Conditioned Depth Estimation

```python
import numpy as np

# Provide known camera parameters
extrinsics = np.array([...])  # (N, 4, 4)
intrinsics = np.array([...])  # (N, 3, 3)

prediction = model.inference(
    image=["img1.jpg", "img2.jpg"],
    extrinsics=extrinsics,
    intrinsics=intrinsics,
    align_to_input_ext_scale=True
)
```

### Example 3: Gaussian Splatting Export

```python
# Use Giant model for GS support
model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT-1.1")
model = model.to("cuda")

prediction = model.inference(
    image=image_paths,
    export_dir="./output",
    export_format="gs_ply-gs_video",
    infer_gs=True
)
```

### Example 4: Feature Visualization

```python
prediction = model.inference(
    image=image_paths,
    export_dir="./output",
    export_format="feat_vis",
    export_feat_layers=[0, 5, 10, 15, 20],
    feat_vis_fps=30
)
```

### Example 5: High-Resolution Processing

```python
prediction = model.inference(
    image=image_paths,
    process_res=1024,
    process_res_method="upper_bound_resize",
    export_dir="./output",
    export_format="glb",
    num_max_points=2_000_000
)
```

### Example 6: Video Processing (CLI)

```bash
# Extract frames at 2 FPS and process
da3 video video.mp4 \
    --fps 2.0 \
    --export-dir ./output \
    --export-format glb \
    --process-res 756
```

### Example 7: Using Backend Service

```bash
# Terminal 1: Start backend
da3 backend --model-dir depth-anything/DA3NESTED-GIANT-LARGE --port 8008

# Terminal 2: Use backend for processing
da3 auto input.jpg \
    --use-backend \
    --backend-url http://localhost:8008 \
    --export-dir ./output
```

### Example 8: COLMAP Dataset Processing

```bash
da3 colmap ./colmap_dataset \
    --sparse-subdir 0 \
    --align-to-input-ext-scale \
    --export-dir ./output \
    --export-format colmap-glb
```

### Example 9: Generate K-Matrix (Camera Intrinsics) from Image

DA3 can automatically estimate camera intrinsics (K-matrix) from a single image without requiring a calibration pattern.

#### Method 1: Using Python API

```python
from depth_anything_3.api import DepthAnything3
import torch
import numpy as np

# Initialize model
model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1")
model = model.to("cuda")

# Process a single image
prediction = model.inference(
    image=["path/to/image.jpg"],
    export_dir="./output",
    export_format="mini_npz"
)

# Extract K-matrix (camera intrinsics)
K = prediction.intrinsics[0]  # Shape: (3, 3)

# K-matrix format:
# [[fx,  0, cx],
#  [0,  fy, cy],
#  [0,   0,  1]]
# where:
# - fx, fy: focal lengths in pixels
# - cx, cy: principal point (image center) in pixels

print(f"K-matrix:\n{K}")
print(f"Focal length X: {K[0, 0]:.2f} pixels")
print(f"Focal length Y: {K[1, 1]:.2f} pixels")
print(f"Principal point: ({K[0, 2]:.2f}, {K[1, 2]:.2f}) pixels")

# Save K-matrix to file
np.savetxt("K.txt", K, fmt="%.6f")
print("K-matrix saved to K.txt")
```

#### Method 2: Using CLI with NPZ Export

```bash
# Process image and export intrinsics
da3 image path/to/image.jpg \
    --export-dir ./output \
    --export-format mini_npz

# Extract K-matrix from NPZ file
python -c "
import numpy as np
data = np.load('./output/exports/mini_npz/results.npz')
K = data['intrinsics'][0]  # First image's K-matrix
np.savetxt('K.txt', K, fmt='%.6f')
print('K-matrix saved to K.txt')
print(f'K-matrix:\n{K}')
"
```

#### Method 3: Using estimate_k_matrix.py Script

If you have the `estimate_k_matrix.py` utility script:

```bash
# Estimate K-matrix from a single image
python estimate_k_matrix.py \
    --image path/to/image.jpg \
    --output K.txt

# The script will:
# 1. Load the image
# 2. Run DA3 inference
# 3. Extract camera intrinsics
# 4. Save K-matrix to K.txt
```

#### Use Cases for K-Matrix

- **3D Reconstruction**: Use K-matrix with depth maps to generate point clouds
- **SLAM/Tracking**: Provide camera intrinsics for visual SLAM systems (e.g., pysrt3d)
- **Camera Calibration**: Estimate camera parameters when calibration patterns are unavailable
- **Multi-View Stereo**: Use intrinsics for pose-conditioned depth estimation

#### Example: Using K-Matrix with pysrt3d

```python
import numpy as np
from pysrt3d import Tracker

# Load K-matrix
K = np.loadtxt("K.txt")

# Get image dimensions
h, w = 1080, 1920  # Your image height and width

# Initialize tracker with K-matrix
tracker = Tracker(imwidth=w, imheight=h, K=K)

# Use tracker for SLAM/tracking
# ...
```

---

## 🚀 <span id="advanced-features">Advanced Features</span>

### Metric Depth Estimation

For models that output metric depth (DA3NESTED-GIANT-LARGE, DA3METRIC-LARGE):

```python
prediction = model.inference(image=images)

# Depth is already in meters for nested models
depth_in_meters = prediction.depth

# For DA3METRIC-LARGE, convert using focal length
if hasattr(prediction, 'intrinsics') and prediction.intrinsics is not None:
    focal = (prediction.intrinsics[0, 0, 0] + prediction.intrinsics[0, 1, 1]) / 2
    metric_depth = focal * prediction.depth / 300.0
```

### Sky Segmentation

Some models provide sky segmentation masks:

```python
prediction = model.inference(image=images)

if prediction.sky is not None:
    sky_mask = prediction.sky  # (N, H, W) boolean array
```

### Intermediate Features

Extract features from specific transformer layers:

```python
prediction = model.inference(
    image=images,
    export_feat_layers=[0, 5, 10, 15, 20]
)

# Access features
if 'feat_layer_0' in prediction.aux:
    features = prediction.aux['feat_layer_0']
```

### Custom Camera Trajectory for GS Video

```python
import numpy as np

# Define custom camera trajectory
render_exts = np.array([...])  # (M, 4, 4)
render_ixts = np.array([...])  # (M, 3, 3)
render_hw = (1080, 1920)

prediction = model.inference(
    image=image_paths,
    export_dir="./output",
    export_format="gs_video",
    infer_gs=True,
    render_exts=render_exts,
    render_ixts=render_ixts,
    render_hw=render_hw
)
```

### Batch Processing with Backend

```bash
# Start backend
da3 backend --model-dir depth-anything/DA3NESTED-GIANT-LARGE

# Process multiple scenes
for scene in scene1 scene2 scene3; do
    da3 auto ./data/$scene \
        --export-dir ./workspace/$scene \
        --use-backend \
        --auto-cleanup
done
```

---

## 📝 Tips and Best Practices

1. **Model Selection**: Use `DA3-LARGE-1.1` for general use, `DA3NESTED-GIANT-LARGE-1.1` for best quality
2. **Backend Service**: Use backend service when processing multiple tasks to avoid reloading models
3. **Resolution**: Higher `process_res` improves quality but increases computation time and memory
4. **GPU Memory**: Be mindful of GPU memory when processing high-resolution inputs
5. **Export Formats**: Combine formats with hyphens (e.g., `mini_npz-glb-feat_vis`)
6. **Ray Pose**: Use `use_ray_pose=True` for better pose accuracy (slightly slower)
7. **Reference View**: Use `ref_view_strategy="middle"` for video sequences
8. **Metric Depth**: Use nested models or DA3METRIC-LARGE for real-world scale depth

---

## 🔗 Additional Resources

- **GitHub Repository**: https://github.com/ByteDance-Seed/Depth-Anything-3
- **Hugging Face Models**: https://huggingface.co/depth-anything
- **Project Page**: https://depth-anything-3.github.io
- **Paper**: https://arxiv.org/abs/2511.10647

---

## ❓ Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `process_res` or `num_max_points`
2. **Slow Inference**: Use backend service or reduce `process_res`
3. **GS Export Fails**: Ensure `infer_gs=True` and using a Giant model
4. **Import Error**: Check installation and Python version (3.9-3.13)
5. **Model Download Fails**: Check internet connection or use Hugging Face mirror

### Getting Help

```bash
# View CLI help
da3 --help
da3 auto --help

# Check model availability
python -c "from depth_anything_3.api import DepthAnything3; print(DepthAnything3.from_pretrained.__doc__)"
```

---

**Last Updated**: Based on Depth-Anything-3 repository documentation
