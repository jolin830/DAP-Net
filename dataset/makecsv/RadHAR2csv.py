import os
import csv
import glob
import pandas as pd
import numpy as np
from typing import List
from collections import defaultdict
import json
import argparse
from datetime import datetime
import re
from typing import Dict, Tuple, Optional
import sys

from read_label_csv import *


def get_dataset_code(ds_map_path: str, dataset_name: str, default_code: int = 1) -> int:
    if not ds_map_path or not os.path.isfile(ds_map_path):
        return int(default_code)
    with open(ds_map_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if all(str(k).isdigit() for k in m.keys()):  # id -> name
        for k, v in m.items():
            if str(v).lower() == dataset_name.lower():
                return int(k)
        return int(default_code)
    else:  # name -> id
        return int(m.get(dataset_name, default_code))


# ---------- Parse raw RadHAR .txt files into per-frame point clouds ----------
def parse_mmwave_txt_frames_xyzdint(file_path: str) -> List[np.ndarray]:
    """
    Return a list of per-frame point arrays with shape [Ni, 5].
    Column order: X, Y, Z, Doppler, Intensity

    Notes:
    - The raw fields may include: x, y, z, velocity, intensity, doppler, doppler_bin, etc.
    - Doppler is preferred from 'doppler'; if missing, use 'doppler_bin';
      otherwise use 'velocity'; if still missing, set to 0.
    - Intensity is set to 0 if absent.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    blocks = [b.strip() for b in raw.split("---") if b.strip()]
    frames: List[List[List[float]]] = []
    started = False
    frame_idx = -1

    for blk in blocks:
        # Parse the key-value pairs for one point.
        kv = {}
        for ln in blk.splitlines():
            ln = ln.strip()
            if ":" in ln:
                k, v = ln.split(":", 1)
                kv[k.strip().lower()] = v.strip()

        # A new frame starts when point_id == 0.
        pid = int(kv.get("point_id", kv.get("point id", "-1")))
        if pid == 0:
            frame_idx += 1
            frames.append([])
            started = True
        if not started:
            continue

        # Read X, Y, Z, velocity, intensity.
        try:
            x = float(kv.get("x", "nan"))
            y = float(kv.get("y", "nan"))
            z = float(kv.get("z", "nan"))
            v = float(kv.get("velocity", "nan"))
            intensity = float(kv.get("intensity", "nan"))
            feat = [x, y, z, v, intensity]
        except ValueError:
            continue

        # Filter out NaN values.
        if any([np.isnan(val) for val in feat]):
            continue

        frames[frame_idx].append(feat)

    frames_np: List[np.ndarray] = []
    for pts in frames:
        if len(pts) == 0:
            frames_np.append(np.zeros((0, 5), dtype=np.float32))
        else:
            frames_np.append(np.asarray(pts, dtype=np.float32))
    return frames_np


# ---------- Converter ----------
class RadHARCSVFormatConverter:
    """
    Read the RadHAR raw directory structure, where actions are stored as subfolders
    and files are in .txt format. For each file, apply sliding windows with
    window/stride and export NTU-style CSV files:

        D{ds:03d}A{act:03d}P{person:03d}S{sample:04d}.csv

    Constraints:
    - A and P start from 0; S starts from 1.
    - S is counted per action class, independently for each action.
    - No point sampling or reshaping is performed; original points are written row by row.
    - Only windows that fully cover window_size are exported.
    """
    def __init__(self, args):
        self.args = args
        self.root = args.dataset_root
        self.output_dir = args.output_dir
        self.samples_dir = os.path.join(self.output_dir, "samples")
        self.window_size = int(args.window_size)
        self.stride = int(args.stride)

        os.makedirs(self.samples_dir, exist_ok=True)

        # Load unified label mappings.
        (
            self.action2id,
            self.id2action,
            self.alias2action,
            self.dataset_alias2action,
            self.action_counts,
        ) = read_label_csv(args.label_csv_path)

        # Parse dataset code.
        self.dataset_code = get_dataset_code(args.ds_map, "RadHAR", default_code=1)

        # Counters.
        self.per_action_sample_counter = defaultdict(int)  # S counter per action
        self.per_action_person_map: Dict[int, Dict[str, int]] = defaultdict(dict)  # P assignment per action
        self.total_written = 0

        # Action subdirectories.
        if args.sub_dirs:
            self.sub_dirs = [d for d in args.sub_dirs]

        if hasattr(args, "class_map") and args.class_map:
            with open(args.class_map, "r", encoding="utf-8") as f:
                self.class_mapping = json.load(f)

    # Map a RadHAR subdirectory name (action name) to the unified action and ID.
    def _unify_action(self, raw_action_dir: str) -> Optional[Tuple[str, int]]:
        action = self.class_mapping.get(raw_action_dir)
        # action = map_raw_actions(raw_action_dir, self.class_mapping)
        canon, uid = resolve_unified_action(
            "radhar",
            action,
            self.alias2action,
            self.dataset_alias2action,
            self.action2id,
        )

        return (canon, uid) if uid is not None else None

    # Assign P for each subject within one action class, starting from 0.
    def _get_person_id(self, action_id: int, file_path: str) -> int:
        stem = os.path.splitext(os.path.basename(file_path))[0]
        pm = self.per_action_person_map[action_id]
        if stem not in pm:
            pm[stem] = len(pm)  # 0, 1, 2, ...
        return pm[stem]

    # Build sample IDs: A and P start from 0; S starts from 1 and is counted per action.
    def _make_sample_id(self, action_id: int) -> str:
        self.per_action_sample_counter[action_id] += 1
        S = self.per_action_sample_counter[action_id]  # 1, 2, 3, ...
        return f"D{self.dataset_code:03d}A{action_id:03d}S{S:04d}"

    def run(self):
        print(f"[RadHAR->CSV] root={self.root}, T={self.window_size}, stride={self.stride}")
        print(f"Sub-dirs(actions): {self.sub_dirs}")

        for state in ["train", "test"]:
            for act_dir in self.sub_dirs:
                uni = self._unify_action(act_dir)
                if uni is None:
                    print(f"[WARN] Failed to map action directory '{act_dir}' to a unified label. Skipping.")
                    continue

                canon_name, action_id = uni
                act_path = os.path.join(self.root, state, act_dir)
                files = sorted(glob.glob(os.path.join(act_path, "*.txt")))
                if not files:
                    print(f"[WARN] No .txt files found in: {act_path}")
                    continue

                for fp in files:
                    frames = parse_mmwave_txt_frames_xyzdint(fp)  # List of [Ni, 5]
                    n = len(frames)
                    if n < self.window_size:
                        continue

                    # Sliding window over complete windows only.
                    for s in range(0, n - self.window_size + 1, self.stride):
                        e = s + self.window_size

                        rows = []
                        for rel, t in enumerate(range(s, e)):
                            pts = frames[t]
                            if pts.size == 0:
                                # Empty frame: write no rows, keep strict "no point -> no row" behavior.
                                continue

                            # One row per point: Frame(relative index), X, Y, Z, Doppler, Intensity.
                            for p in pts:
                                rows.append([rel, float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])])

                        if not rows:
                            # Skip this window if all frames are empty.
                            continue

                        sample_id = self._make_sample_id(action_id)
                        out_path = os.path.join(self.samples_dir, f"{sample_id}.csv")
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        df = pd.DataFrame(rows, columns=["Frame", "X", "Y", "Z", "Doppler", "Intensity"])
                        df.to_csv(out_path, index=False)
                        self.total_written += 1

                print(f"[{act_dir} -> A{action_id}] done. current total = {self.total_written}")

        print(f"[DONE] total CSV samples: {self.total_written}")

        # Optional: save a simple summary file.
        meta = {
            "dataset": "RadHAR",
            "dataset_code": self.dataset_code,
            "window_size": self.window_size,
            "stride": self.stride,
            "total_samples": self.total_written,
            "per_action_samples": {int(k): int(v) for k, v in self.per_action_sample_counter.items()},
        }
        with open(os.path.join(self.output_dir, "radhar_csv_summary.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


# CLI arguments
def build_args():
    ap = argparse.ArgumentParser("Convert MMFI dataset to CSV format (NTU style)")
    ap.add_argument(
        "--dataset",
        type=str,
        default="RadHAR",
        choices=["FastHAR", "RadHAR", "MMFI", "mRI"],
        help="Which dataset to convert",
    )
    ap.add_argument("--dataset-root", type=str, default="RadHAR", help="Path to the dataset")
    ap.add_argument("--sub-dirs", default=["boxing", "jack", "jump", "squats", "walk"], help="Action subdirectories")
    ap.add_argument("--window-size", default=60, help="Window size")
    ap.add_argument("--stride", default=10, help="Sliding window stride")
    ap.add_argument("--output-dir", type=str, default="UniMM-HAR/csv/RadHAR")
    ap.add_argument("--label-csv-path", type=str, default="UniMM-HAR/label.csv", help="Path to UniMM-HAR/label.csv")
    ap.add_argument("--class-map", type=str, default="UniMM-HAR/RadHAR/class_mapping_csv.json", help="Action-to-ID JSON file")
    ap.add_argument(
        "--ds-map",
        type=str,
        default="UniMM-HAR/ds_mapping.json",
        help="Dataset code mapping JSON file (e.g., {'1':'RadHAR','4':'FastHAR'})",
    )

    return ap.parse_args()


def main():
    args = build_args()
    conv = RadHARCSVFormatConverter(args)
    conv.run()


if __name__ == "__main__":
    main()