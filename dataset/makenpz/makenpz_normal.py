#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_normalized_npz.py

Normalize raw NPZ samples (each NPZ contains a 'pointcloud' key)
using per-sample clip normalization, and save the normalized point clouds
to a new NPZ directory while preserving the original relative paths.

Optionally, normalization statistics can also be saved as JSON files.

Example:
python make_normalized_npz.py \
    --input /data/raw_npz \
    --output /data/norm_npz \
    --pattern .npz \
    --overwrite
"""

import os
import sys
import argparse
import json
import glob
from typing import Tuple, Dict

import numpy as np
import torch


# ------------------ clip_normalize ------------------

def clip_normalize(clip: torch.Tensor, epsilon: float = 1e-6) -> Tuple[torch.Tensor, Dict]:
    """
    Perform per-sample normalization for a single clip.

    Supported input shape:
        [T, N, 5]
    where:
        5 = [x, y, z, velocity, intensity]

    Returns:
        (clip_norm, used_stats)

    used_stats contains:
        - centroid
        - xyz_scale
        - v_scale
        - i_scale
    """
    assert clip.ndim == 3 and clip.shape[2] >= 5, \
        "clip must have shape [T, N, >=5], where the last 5 dims are x,y,z,v,i"

    T, N, C = clip.shape
    device = clip.device
    dtype = clip.dtype

    flat = clip.reshape(-1, C)  # [T*N, C]
    xyz = flat[:, :3]

    nonzero_mask = (xyz.abs().sum(dim=1) > 1e-8)

    # Return directly if all points are zero.
    if nonzero_mask.sum() == 0:
        stats = {
            "centroid": [0.0, 0.0, 0.0],
            "xyz_scale": 1.0,
            "v_scale": 1.0,
            "i_scale": 1.0,
        }
        return clip.clone(), stats

    coords_valid = xyz[nonzero_mask]       # [M, 3]
    v_valid = flat[:, 3][nonzero_mask]     # [M]
    i_valid = flat[:, 4][nonzero_mask]     # [M]

    # Compute centroid.
    centroid = coords_valid.mean(dim=0)

    # Compute xyz scaling using p99 distance.
    dists = torch.linalg.norm(coords_valid - centroid.unsqueeze(0), dim=1)

    try:
        p99 = float(torch.quantile(dists, 0.99).item())
    except Exception:
        p99 = float(np.percentile(dists.cpu().numpy(), 99.0))

    xyz_scale = max(p99, float(epsilon))

    if xyz_scale < epsilon:
        maxd = float(dists.max().item()) if dists.numel() > 0 else 1.0
        xyz_scale = max(maxd, float(epsilon))

    # Doppler scaling.
    v_abs = v_valid.abs()

    try:
        v_p99 = float(torch.quantile(v_abs, 0.99).item())
    except Exception:
        v_p99 = float(np.percentile(v_abs.cpu().numpy(), 99.0))

    v_scale = max(v_p99, float(epsilon))

    if v_scale < epsilon:
        v_scale = 1.0

    # Intensity scaling after log1p transform.
    i_log = torch.log1p(torch.clamp(i_valid, min=0.0))

    try:
        i_p99 = float(torch.quantile(i_log, 0.99).item())
    except Exception:
        i_p99 = float(np.percentile(i_log.cpu().numpy(), 99.0))

    i_scale = max(i_p99, float(epsilon))

    if i_scale < epsilon:
        i_scale = 1.0

    # Write normalized values back.
    flat_out = flat.clone()

    coords_norm = (coords_valid - centroid.unsqueeze(0)) / float(xyz_scale)
    flat_out[nonzero_mask, :3] = coords_norm

    v_norm = (v_valid / float(v_scale)).to(dtype)
    flat_out[nonzero_mask, 3] = v_norm

    i_log_norm = (
        torch.log1p(torch.clamp(i_valid, min=0.0)) / float(i_scale)
    ).to(dtype)

    flat_out[nonzero_mask, 4] = i_log_norm

    clip_norm = flat_out.view(T, N, C)

    stats = {
        "centroid": [float(x) for x in centroid.cpu().numpy().tolist()],
        "xyz_scale": float(xyz_scale),
        "v_scale": float(v_scale),
        "i_scale": float(i_scale),
    }

    return clip_norm, stats


# ------------------ Single-file processing ------------------

def process_one_npz(
    in_path: str,
    out_path: str,
    save_stats: bool = True,
    overwrite: bool = False
) -> Tuple[bool, Dict]:
    """
    Load an NPZ file containing 'pointcloud',
    apply clip normalization, and save the result.

    Returns:
        (ok, metadata)

    metadata includes:
        - input_shape
        - output_shape
        - stats
        - skipped_reason (if skipped)
    """
    meta = {}

    if not os.path.isfile(in_path):
        return False, {"skipped_reason": "input_not_exist"}

    if (not overwrite) and os.path.exists(out_path):
        # Skip if the output already exists.
        return False, {
            "skipped_reason": "exists",
            "out_path": out_path,
        }

    try:
        data = np.load(in_path, allow_pickle=True)
    except Exception as e:
        return False, {"skipped_reason": f"load_error:{e}"}

    if "pointcloud" not in data:
        return False, {"skipped_reason": "no_key_pointcloud"}

    pc = data["pointcloud"]

    # Expected shape: [T, N, >=5]
    if pc.ndim != 3 or pc.shape[2] < 5:
        return False, {"skipped_reason": f"invalid_shape:{pc.shape}"}

    # Convert to torch tensor and normalize.
    try:
        clip_t = torch.as_tensor(pc[:, :, :5], dtype=torch.float32)
        clip_norm_t, stats = clip_normalize(clip_t)
        clip_norm = clip_norm_t.cpu().numpy()
    except Exception as e:
        return False, {"skipped_reason": f"normalize_error:{e}"}

    # Save compressed NPZ.
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        np.savez_compressed(out_path, pointcloud=clip_norm)
    except Exception as e:
        return False, {"skipped_reason": f"save_error:{e}"}

    # Optionally save normalization statistics.
    if save_stats:
        json_path = out_path.replace(".npz", ".norm.json")

        try:
            with open(json_path, "w") as f:
                json.dump(stats, f, indent=2)
        except Exception:
            # Ignore JSON save errors.
            pass

    meta = {
        "input_shape": pc.shape,
        "output_shape": clip_norm.shape,
        "stats": stats,
    }

    return True, meta


# ------------------ Dataset traversal ------------------

def build_file_list(
    input_root: str,
    pattern: str = ".npz",
    recursive: bool = True
):
    """
    Find all files matching the given pattern under input_root.

    Returns:
        List of absolute file paths.
    """
    input_root = os.path.abspath(input_root)

    matched = []

    if recursive:
        for dp, dn, fnames in os.walk(input_root):
            for fn in fnames:
                if fn.lower().endswith(pattern.lower()):
                    matched.append(os.path.join(dp, fn))
    else:
        for fn in os.listdir(input_root):
            if fn.lower().endswith(pattern.lower()):
                matched.append(os.path.join(input_root, fn))

    matched.sort()

    return matched


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply clip normalization to raw NPZ samples "
            "and save them as normalized NPZ files."
        )
    )

    parser.add_argument(
        "--input",
        "-i",
        default="UniMM-HAR/npz/T32P64/CSub",
        help="Root directory of raw NPZ files",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="UniMM-HAR/npz/T32P64/CSub_normal",
        help="Output directory for normalized NPZ files",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default=".npz",
        help="File extension pattern to match",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively search subdirectories",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files",
    )

    parser.add_argument(
        "--save-stats",
        action="store_true",
        default=True,
        help="Save normalization statistics as .norm.json",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Only print files to be processed without saving",
    )

    args = parser.parse_args()

    in_root = os.path.abspath(args.input)
    out_root = os.path.abspath(args.output)

    files = build_file_list(
        in_root,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    if len(files) == 0:
        print("No matching files found:", in_root)
        return

    print(f"Found {len(files)} files.")
    print(f"Input root : {in_root}")
    print(f"Output root: {out_root}")

    n_ok = 0
    n_skip = 0

    for fp in files:
        # Compute relative output path.
        rel = os.path.relpath(fp, in_root)
        out_path = os.path.join(out_root, rel)

        # Ensure output extension is .npz.
        if not out_path.lower().endswith(".npz"):
            out_path = out_path + ".npz"

        if args.dry_run:
            print("[DRY]", fp, "->", out_path)
            continue

        ok, meta = process_one_npz(
            fp,
            out_path,
            save_stats=args.save_stats,
            overwrite=args.overwrite,
        )

        if ok:
            n_ok += 1
            print(
                f"[OK] {rel} -> saved, "
                f"out_shape={meta.get('output_shape')}, "
                f"stats={meta.get('stats')}"
            )
        else:
            reason = meta.get("skipped_reason", "unknown")
            n_skip += 1
            print(f"[SKIP] {rel}: {reason}")

    print("Done.")
    print("Converted :", n_ok)
    print("Skipped   :", n_skip)


if __name__ == "__main__":
    main()