# Analysis of Depth-Anything-3 Output

## 📁 Output Directory Contents

```
/root/Depth-Anything-3/user/output/earbud/
├── scene.glb          (16 MB) - 3D visualization file
├── scene.jpg          (19 KB) - Preview thumbnail
└── depth_vis/         - Depth visualization images
```

## 🔍 What is `scene.glb`?

**`scene.glb`** is a **glTF Binary** file - a 3D scene format that contains:

1. **3D Point Cloud**: Generated from the depth maps and colored using the original images
2. **Camera Wireframes**: Visual representation of camera poses (if `show_cameras=True`)
3. **Scene Metadata**: Contains alignment information (`hf_alignment` matrix)
4. **Embedded Camera Parameters**: The intrinsics and extrinsics are embedded in the 3D scene structure

### GLB Format Details:
- **Format**: glTF 2.0 binary format
- **Size**: 16 MB (contains point cloud with colors)
- **Viewable in**: 
  - [Three.js Viewer](https://threejs.org/editor/)
  - [glTF Viewer](https://gltf-viewer.donmccurdy.com/)
  - Blender (import as glTF)
  - Any glTF-compatible 3D viewer

### What GLB Contains:
- ✅ Depth maps converted to 3D points
- ✅ Colors from original images
- ✅ Camera poses (extrinsics) as wireframe pyramids
- ✅ Camera intrinsics (embedded in scene structure)
- ❌ **NOT a separate K-matrix text file**

## ❓ Why No K-Matrix File?

The **K-matrix (camera intrinsics) is NOT saved as a separate text file** because:

1. **Default Export Format**: The `da3 auto` command uses `--export-format glb` by default
2. **GLB Format Purpose**: GLB is designed for 3D visualization, not for exporting raw camera parameters
3. **Intrinsics Are Embedded**: The camera intrinsics are embedded within the GLB file structure but not exported as a separate `.txt` file

### Where Intrinsics Are Stored:

The intrinsics **ARE** computed and available, but they're only saved when using specific export formats:

- ✅ **`mini_npz`**: Saves intrinsics in `exports/mini_npz/results.npz`
- ✅ **`npz`**: Saves intrinsics in `exports/npz/results.npz`
- ❌ **`glb`**: Embeds intrinsics in 3D scene but doesn't create separate file

## 🔧 How to Get the K-Matrix

### Option 1: Re-run with NPZ Export Format (Recommended)

```bash
cd /root/Depth-Anything-3
conda activate da3

# Re-run with mini_npz format to get intrinsics
da3 auto /user/input/earbud/images \
    --export-dir /root/Depth-Anything-3/user/output/earbud \
    --export-format mini_npz-glb \
    --process-res 384  # Use lower resolution to avoid OOM
```

This will create:
```
/user/output/earbud/
├── exports/
│   └── mini_npz/
│       └── results.npz  # Contains intrinsics!
└── scene.glb
```

Then extract K-matrix:
```python
import numpy as np

# Load the NPZ file
data = np.load('/root/Depth-Anything-3/user/output/earbud/exports/mini_npz/results.npz')

# Get intrinsics (shape: N, 3, 3)
intrinsics = data['intrinsics']

# Get first image's K-matrix
K = intrinsics[0]

# Save as text file
np.savetxt('K.txt', K, fmt='%.6f')
print("K-matrix:")
print(K)
```

### Option 2: Use Python API Directly

```python
from depth_anything_3.api import DepthAnything3
import torch
import numpy as np
from pathlib import Path

# Initialize model
model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1")
model = model.to("cuda")

# Load images
import glob
images = sorted(glob.glob("/user/input/earbud/images/*.png"))

# Run inference (without export, or with mini_npz)
prediction = model.inference(
    images,
    export_dir="./output",
    export_format="mini_npz",  # This saves intrinsics
    process_res=384  # Lower resolution to avoid OOM
)

# Extract K-matrix from prediction
K = prediction.intrinsics[0]  # First image's K-matrix

# Save to text file
output_path = Path("./output/K.txt")
np.savetxt(str(output_path), K, fmt='%.6f')
print(f"K-matrix saved to {output_path}")
print(f"\nK-matrix:\n{K}")
```

### Option 3: Use the `estimate_k_matrix.py` Script

```bash
# Process a single image to get K-matrix
python /root/lib/estimate_k_matrix.py \
    --image /user/input/earbud/images/earbuds_003_00110.png \
    --output /root/Depth-Anything-3/user/output/earbud/K.txt
```

## 📊 Summary

| Export Format | Saves K-Matrix? | Location |
|--------------|----------------|----------|
| `glb` | ❌ No (embedded only) | Embedded in 3D scene |
| `mini_npz` | ✅ Yes | `exports/mini_npz/results.npz` |
| `npz` | ✅ Yes | `exports/npz/results.npz` |
| `colmap` | ✅ Yes | COLMAP format files |

## 🎯 Recommended Solution

To get the K-matrix file, re-run the command with `mini_npz` format:

```bash
da3 auto /user/input/earbud/images \
    --export-dir /root/Depth-Anything-3/user/output/earbud \
    --export-format mini_npz-glb \
    --process-res 384
```

This will:
1. ✅ Create `scene.glb` (3D visualization)
2. ✅ Create `exports/mini_npz/results.npz` (contains intrinsics)
3. ✅ Allow you to extract K-matrix from the NPZ file

Then extract K-matrix:
```python
import numpy as np
data = np.load('/root/Depth-Anything-3/user/output/earbud/exports/mini_npz/results.npz')
K = data['intrinsics'][0]
np.savetxt('K.txt', K, fmt='%.6f')
```
