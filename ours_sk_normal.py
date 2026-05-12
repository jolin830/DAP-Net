import os
import re
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import math


def clip_normalize(clip: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Per-sample normalization for a single clip (supports [T, N, 5], where 5 = x, y, z, v, i).

    Strategy:
      - For xyz: center all non-zero points by their centroid, then divide by the 99th percentile
        distance to the centroid (p99 distance).
      - For Doppler v: use the 99th percentile of |v| as the scale (fall back to 1.0 if needed).
      - For Intensity i: apply log1p first, then use the 99th percentile after log transform.
      - Zero points (all xyz = 0) are kept unchanged and are used as padded frames.

    Input:
      clip: Tensor with shape [T, N, 5]

    Output:
      clip_norm: Tensor with the same shape as input, normalized (float tensor)
    """
    assert clip.ndim == 3 and clip.shape[2] >= 5, "clip must be [T, N, >=5], with the last 5 dims as x, y, z, v, i"
    T, N, C = clip.shape
    device = clip.device
    dtype = clip.dtype

    # Flatten for statistics computation while keeping the original indexing for writing back.
    flat = clip.view(-1, C)  # [T*N, C]

    # Find valid points: any non-zero component in xyz.
    xyz = flat[:, :3]  # [T*N, 3]
    nonzero_mask = (xyz.abs().sum(dim=1) > 1e-8)  # [T*N], True indicates a valid point

    if nonzero_mask.sum() == 0:
        # All points are zero (the entire clip is padded); return the original clip directly.
        return clip.clone()

    # Extract valid-point tensors.
    coords_valid = xyz[nonzero_mask]          # [M, 3]
    v_valid = flat[:, 3][nonzero_mask]        # [M]
    i_valid = flat[:, 4][nonzero_mask]        # [M]

    # 1) Center xyz using the centroid of valid points.
    centroid = coords_valid.mean(dim=0)       # [3]

    # 2) xyz scale: 99th percentile distance to the centroid.
    dists = torch.linalg.norm(coords_valid - centroid.unsqueeze(0), dim=1)  # [M]
    try:
        p99 = torch.quantile(dists, 0.99).item()
    except Exception:
        # Compatibility fallback for older PyTorch versions without quantile.
        p99 = float(np.percentile(dists.cpu().numpy(), 99.0))
    xyz_scale = max(p99, float(epsilon))
    if xyz_scale < epsilon:
        # If p99 is too small, fall back to max distance or 1.0.
        maxd = float(dists.max().item()) if dists.numel() > 0 else 1.0
        xyz_scale = max(maxd, float(epsilon))

    # 3) Doppler scale: 99th percentile of |v| (fall back to 1.0 if needed).
    v_abs = v_valid.abs()
    try:
        v_p99 = torch.quantile(v_abs, 0.99).item()
    except Exception:
        v_p99 = float(np.percentile(v_abs.cpu().numpy(), 99.0))
    v_scale = max(v_p99, float(epsilon))
    if v_scale < epsilon:
        v_scale = 1.0

    # 4) Intensity: apply log1p first, then compute the 99th percentile.
    i_log = torch.log1p(torch.clamp(i_valid, min=0.0))
    try:
        i_p99 = torch.quantile(i_log, 0.99).item()
    except Exception:
        i_p99 = float(np.percentile(i_log.cpu().numpy(), 99.0))
    i_scale = max(i_p99, float(epsilon))
    if i_scale < epsilon:
        i_scale = 1.0

    # Store statistics for optional debugging.
    # used_stats = {
    #     'centroid': centroid.cpu().numpy().tolist(),
    #     'xyz_scale': float(xyz_scale),
    #     'v_scale': float(v_scale),
    #     'i_scale': float(i_scale)
    # }

    # Normalize valid points and write them back.
    flat_out = flat.clone()

    # xyz normalization: (coords - centroid) / xyz_scale
    coords_norm = (coords_valid - centroid.unsqueeze(0)) / float(xyz_scale)
    flat_out[nonzero_mask, :3] = coords_norm

    # v normalization: v / v_scale
    v_norm = (v_valid / float(v_scale)).to(dtype)
    flat_out[nonzero_mask, 3] = v_norm

    # intensity normalization: log1p(i) / i_scale
    i_log_norm = (torch.log1p(torch.clamp(i_valid, min=0.0)) / float(i_scale)).to(dtype)
    flat_out[nonzero_mask, 4] = i_log_norm

    # Reshape back to the original form.
    clip_norm = flat_out.view(T, N, C)

    return clip_norm


def _parse_action_idx_from_name(name: str) -> int:
    """
    Parse Axxx from a filename and return its numeric value (no -1 offset).

    Example:
        D001A003E000P000S0001.npz -> 3 (int)
    """
    m = re.search(r"A(\d{3})", name)
    if m is None:
        raise ValueError(f"Filename '{name}' does not contain 'Axxx'.")
    return int(m.group(1))


def remap_labels(labels, start=0):
    """
    Remap labels to 0..N-1 and return the remapped list, preserving order.
    """
    unique_sorted = sorted(set(labels))
    mapping = {old: i + start for i, old in enumerate(unique_sorted)}
    mapped_labels = [mapping[x] for x in labels]
    return mapped_labels


class BDSubject_SK(Dataset):
    """
    Compatible with 5D point clouds [T, Ni, 5], where 5 = x, y, z, Doppler, Intensity.
    Each sample is loaded from the 'pointcloud' key in an NPZ file.
    """
    def __init__(self, root, num_points=64, train=True, scale_aug=True, inputC=3):
        super().__init__()
        self.num_points = int(num_points)
        self.train = bool(train)
        self.scale_aug = bool(scale_aug)
        self.inputC = inputC

        split_dir = os.path.join(root, "train" if train else "test")
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split folder not found: {split_dir}")

        paths = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
        if len(paths) == 0:
            raise ValueError(f"No .npz files found in {split_dir}")

        self.videos = []
        self.labels = []
        self.names = []
        for p in paths:
            name = os.path.basename(p)
            try:
                label = _parse_action_idx_from_name(name)
            except Exception:
                continue
            self.videos.append(p)
            self.labels.append(label)
            self.names.append(name)

        if len(self.videos) == 0:
            raise ValueError("No valid .npz samples found after checking filenames.")
        self.labels = remap_labels(self.labels)

        self.num_classes = max(self.labels) + 1 if len(self.labels) > 0 else 0

        # Default fusion parameters (can be overridden externally if needed).
        self.frame_fusion_w = 1
        self.frame_fusion_s = 1

    def __len__(self):
        return len(self.videos)

    def _sample_points(self, pts: torch.Tensor, num_points: int, inputC: int) -> torch.Tensor:
        """
        Sample a fixed number of points from a single frame.

        pts: [M, inputC]
        """
        m = pts.shape[0]
        if m == 0:
            return torch.zeros((num_points, inputC), dtype=torch.float32)
        if m >= num_points:
            idx = torch.randperm(m)[:num_points]
            return pts[idx]
        # m < num_points: repeat and randomly fill the remainder.
        repeat, residue = divmod(num_points, m)
        base = torch.arange(m, device=pts.device).repeat(repeat)
        if residue > 0:
            extra = torch.randperm(m, device=pts.device)[:residue]
            idx = torch.cat([base, extra], dim=0)
        else:
            idx = base
        return pts[idx]

    def __getitem__(self, index):
        path = self.videos[index]
        label = self.labels[index]
        name = self.names[index]

        # Load point clouds (ensure at least 5 dims: x, y, z, v, i).
        data = np.load(path, allow_pickle=True)["pointcloud"]
        T = data.shape[0]

        # Process each frame; each frame keeps num_points points.
        clip_list = []
        for t in range(T):
            frame = torch.as_tensor(data[t], dtype=torch.float32)  # [Ni, 5]
            # frame = self._sample_points(frame, self.num_points, self.inputC)    # [N, 5] # not used here
            clip_list.append(frame)
        clip = torch.stack(clip_list, dim=0)  # [T, N, 5]

        # Normalize to a suitable scale (per-sample).
        clip = clip_normalize(clip)  # [T, N, 5]
        # clip = clip[:, :, :self.inputC]

        # clip = clip.reshape(-1, self.inputC)    # for PC

        # Apply simple isotropic scaling augmentation during training (optional).
        if self.train and self.scale_aug:
            scales = torch.empty(5).uniform_(0.9, 1.1).to(clip.device)
            clip = clip * scales

        return clip.float(), label, name


if __name__ == "__main__":
    import argparse, random
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="UniMM-HAR/npz/CSub", help="Dataset root directory (contains train/ and test/)")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--train", default=True, action="store_true")
    args = parser.parse_args()

    ds = BDSubject_SK(root=args.root, num_points=args.num_points, train=args.train)
    print(f"Samples: {len(ds)}, num_classes={ds.num_classes}")
    r = random.Random(0)
    for _ in range(3):
        i = r.randrange(len(ds))
        clip, label, idx = ds[i]
        print(f"idx={idx} label={label} clip.shape={tuple(clip.shape)}")  # -> (T*N, 5)