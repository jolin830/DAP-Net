
import os
import argparse

import numpy as np
import pandas as pd


# ---------------- Time & Point Helpers ----------------

def temporal_downsample_indices(n_frames: int, T: int):
    """
    Uniformly downsample frame indices to T frames.

    Cases:
    - n_frames >= T:
        Return T uniformly sampled indices.
    - n_frames == 0:
        Return an empty list.
    - n_frames < T:
        Return original indices.
        Zero-padding should be handled by the caller.
    """
    if n_frames == 0:
        return []

    if T <= 1:
        return [n_frames // 2]  # Select the middle frame.

    if n_frames <= T:
        return list(range(n_frames))

    # Uniformly sample T indices.
    step = (n_frames - 1) / (T - 1)
    idx = [int(np.floor(i * step + 1e-8)) for i in range(T)]

    # Ensure the first and last frames are included.
    idx[0] = 0
    idx[-1] = n_frames - 1

    return idx


def farthest_point_sample_numpy(xyz: np.ndarray, m: int) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) based on XYZ coordinates.

    Args:
        xyz: [N, 3]
        m  : number of sampled points

    Returns:
        indices: [m]
    """
    n = xyz.shape[0]

    if n == 0:
        return np.array([], dtype=np.int64)

    m = min(m, n)

    centroids = np.zeros(m, dtype=np.int64)
    distances = np.full(n, 1e10, dtype=np.float64)

    farthest = np.random.randint(0, n)

    for i in range(m):
        centroids[i] = farthest

        centroid = xyz[farthest][None, :]
        dist = np.sum((xyz - centroid) ** 2, axis=1)

        distances = np.minimum(distances, dist)
        farthest = int(np.argmax(distances))

    return centroids


def frame_to_fixed_points(frame_df: pd.DataFrame, P: int) -> np.ndarray:
    """
    Convert a single frame into a fixed-size point cloud [P, 5].

    Rules:
    - N > P:
        Apply FPS downsampling.
    - N < P:
        Repeat-sample points with replacement.
    - N == 0:
        Return a zero-filled frame.

    Expected columns:
        X, Y, Z, Doppler, Intensity
    """
    cols = ["X", "Y", "Z", "Doppler", "Intensity"]

    pts = (
        frame_df[cols].to_numpy(dtype=np.float32)
        if not frame_df.empty
        else np.zeros((0, 5), dtype=np.float32)
    )

    N = pts.shape[0]

    if N == 0:
        return np.zeros((P, 5), dtype=np.float32)

    if N > P:
        idx = farthest_point_sample_numpy(pts[:, :3], P)
        return pts[idx]

    if N < P:
        extra = np.random.choice(N, P - N, replace=True)
        return np.concatenate([pts, pts[extra]], axis=0)

    return pts  # N == P


# ---------------- CSV -> NPZ (Single File) ----------------

def csv_to_npz(csv_path: str, out_npz_path: str, T: int, P: int):
    """
    Convert a single CSV file into NPZ format.

    Output:
        key='pointcloud'
        shape=[T, P, 5]

    Temporal processing:
    - n_frames > T:
        Uniformly downsample to T frames.
    - n_frames < T:
        Pad trailing zero frames.

    Point processing:
    - N > P:
        FPS downsampling.
    - N < P:
        Repeat-sampling.
    - N == 0:
        Zero-filled frame.
    """
    df = pd.read_csv(csv_path)

    required = {"Frame", "X", "Y", "Z", "Doppler", "Intensity"}

    if not required.issubset(df.columns):
        miss = required - set(df.columns)
        raise ValueError(f"{csv_path} is missing columns: {miss}")

    # Normalize frame indices and sort.
    df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["Frame"]).copy()
    df["Frame"] = df["Frame"].astype(int)

    # Extract unique frame IDs.
    unique_frames = sorted(df["Frame"].unique().tolist())
    n_frames = len(unique_frames)

    seq = np.zeros((T, P, 5), dtype=np.float32)

    if n_frames == 0:
        # Save an all-zero sequence.
        np.savez_compressed(out_npz_path, pointcloud=seq)
        return

    # Select frame indices from unique_frames.
    if n_frames >= T:
        sel_idx = temporal_downsample_indices(n_frames, T)
        sel_frames = [unique_frames[i] for i in sel_idx]
        t_limit = T
    else:
        sel_frames = unique_frames[:]
        t_limit = n_frames

    # Fill valid frames.
    for t_rel in range(t_limit):
        fr = sel_frames[t_rel]

        frame_df = df.loc[df["Frame"] == fr]
        seq[t_rel] = frame_to_fixed_points(frame_df, P)

    # Remaining frames stay zero-filled.

    # Save NPZ.
    os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)

    np.savez_compressed(out_npz_path, pointcloud=seq)

    print(out_npz_path.split("/")[-1] + " finish!")


# ---------------- Batch Driver ----------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Convert CSV(Frame,X,Y,Z,Doppler,Intensity) "
            "to NPZ[T,P,5]. "
            "Temporal downsampling/padding and FPS-based point sampling are applied."
        )
    )

    ap.add_argument("--dataset", type=str, default="mRI")

    ap.add_argument(
        "--input-dir",
        default="UniMM-HAR/csv",
        help="Root directory containing CSV files",
    )

    ap.add_argument(
        "--output-dir",
        default="UniMM-HAR/npz",
        help="Output NPZ root directory",
    )

    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--P", type=int, default=64)

    ap.add_argument(
        "--pattern",
        type=str,
        default=".csv",
    )

    args = ap.parse_args()

    in_root = os.path.join(args.input_dir, args.dataset, "samples")

    out_root = os.path.join(
        args.output_dir,
        f"T{args.T}P{args.P}",
        args.dataset,
    )

    converted = 0
    skipped = 0

    for root, _, files in os.walk(in_root):
        for fn in files:
            if not fn.lower().endswith(args.pattern.lower()):
                continue

            csv_path = os.path.join(root, fn)

            rel_path = os.path.relpath(csv_path, in_root)
            out_rel = os.path.splitext(rel_path)[0] + ".npz"

            out_npz_path = os.path.join(out_root, out_rel)

            try:
                csv_to_npz(
                    csv_path,
                    out_npz_path,
                    args.T,
                    args.P,
                )

                converted += 1

            except Exception as e:
                skipped += 1
                print(f"[SKIP] {csv_path}: {e}")

    print(
        f"Done. Converted CSVs: {converted}, "
        f"Skipped: {skipped}. "
        f"Output -> {out_root}"
    )


if __name__ == "__main__":
    main()