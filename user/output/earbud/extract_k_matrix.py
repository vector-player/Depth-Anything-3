#!/usr/bin/env python3
"""
Extract K-matrix from Depth-Anything-3 output.

This script extracts camera intrinsics (K-matrix) from:
1. NPZ files (mini_npz or npz format)
2. Or re-runs inference to get intrinsics
"""

import argparse
import sys
from pathlib import Path
import numpy as np

def extract_from_npz(npz_path: Path, output_path: Path, image_idx: int = 0) -> bool:
    """Extract K-matrix from NPZ file."""
    if not npz_path.exists():
        print(f"Error: NPZ file not found: {npz_path}")
        return False
    
    try:
        data = np.load(npz_path)
        
        if 'intrinsics' not in data:
            print(f"Error: 'intrinsics' not found in NPZ file.")
            print(f"Available keys: {list(data.keys())}")
            return False
        
        intrinsics = data['intrinsics']
        print(f"Found intrinsics with shape: {intrinsics.shape}")
        
        if image_idx >= len(intrinsics):
            print(f"Error: Image index {image_idx} out of range (0-{len(intrinsics)-1})")
            return False
        
        K = intrinsics[image_idx]
        
        # Save K-matrix
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(str(output_path), K, fmt='%.6f')
        
        print(f"\n✓ K-matrix extracted and saved to: {output_path}")
        print(f"\nK-matrix (3x3) for image {image_idx}:")
        print(K)
        print(f"\nParameters:")
        print(f"  fx (focal length X): {K[0, 0]:.6f} pixels")
        print(f"  fy (focal length Y): {K[1, 1]:.6f} pixels")
        print(f"  cx (principal point X): {K[0, 2]:.6f} pixels")
        print(f"  cy (principal point Y): {K[1, 2]:.6f} pixels")
        
        return True
        
    except Exception as e:
        print(f"Error: Failed to extract K-matrix: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract K-matrix from Depth-Anything-3 NPZ output"
    )
    
    parser.add_argument(
        '--npz',
        type=str,
        help='Path to NPZ file (e.g., exports/mini_npz/results.npz)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='K.txt',
        help='Output K-matrix file path (default: K.txt)'
    )
    
    parser.add_argument(
        '--image-idx',
        type=int,
        default=0,
        help='Image index to extract (default: 0 for first image)'
    )
    
    parser.add_argument(
        '--search',
        action='store_true',
        help='Search for NPZ files in current directory'
    )
    
    args = parser.parse_args()
    
    # Search for NPZ files if requested
    if args.search:
        current_dir = Path.cwd()
        npz_files = list(current_dir.rglob("*.npz"))
        if npz_files:
            print("Found NPZ files:")
            for i, npz_file in enumerate(npz_files):
                print(f"  {i}: {npz_file}")
            if not args.npz and npz_files:
                args.npz = str(npz_files[0])
                print(f"\nUsing: {args.npz}")
        else:
            print("No NPZ files found in current directory.")
            print("\nTo get K-matrix, re-run da3 with mini_npz format:")
            print("  da3 auto <input> --export-format mini_npz-glb --export-dir <output>")
            return 1
    
    if not args.npz:
        print("Error: --npz file path required (or use --search to find NPZ files)")
        parser.print_help()
        return 1
    
    npz_path = Path(args.npz).resolve()
    output_path = Path(args.output).resolve()
    
    success = extract_from_npz(npz_path, output_path, args.image_idx)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
