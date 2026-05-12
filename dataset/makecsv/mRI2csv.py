import os
import csv
import glob
import json
import argparse
import sys
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np

from read_label_csv import *


def _cm_get(cm, k):
    """Compatibility helper for class_mapping keys of both '0' and 0."""
    return cm.get(k, cm.get(str(k), cm.get(int(k)) if isinstance(k, str) and k.isdigit() else None))


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


def parse_subject_to_id(subject: str) -> int:
    if subject is None:
        return 0
    s = str(subject)
    m = re.search(r"(?:subject|s)\s*0*([0-9]+)", s, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"[0-9]+", s)
    if m:
        return int(m.group(0))
    return 0


class mRICSVFormatConverter:
    """
    Apply a sliding window over each JSON-labeled segment [a, b] (inclusive).

    - Window length: T = window_size
    - Stride: stride
    - Frames beyond the segment end are padded with zero rows

    Each sample is saved as DxxAxxPxxxSxxxx.csv with columns:
    Frame, X, Y, Z, Doppler, Intensity

    - Frame is the relative index within the window: 0 .. T-1
    - Raw points are written row by row without sampling
    - Padded frames are represented by one all-zero row
    """
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset
        self.csv_folder = args.dataset_root
        self.seg_path = args.seg_path
        self.output_dir = args.output_dir
        self.samples_dir = os.path.join(self.output_dir, "samples")
        self.window_size = int(args.window_size)
        self.stride = int(args.stride)

        os.makedirs(self.samples_dir, exist_ok=True)

        (
            self.action2id,
            self.id2action,
            self.alias2action,
            self.dataset_alias2action,
            self.action_counts,
        ) = read_label_csv(args.label_csv_path)

        self.class_mapping = None
        if getattr(args, "class_map", None):
            if os.path.isfile(args.class_map):
                with open(args.class_map, "r", encoding="utf-8") as f:
                    self.class_mapping = json.load(f)

        self.dataset_code = get_dataset_code(args.ds_map, args.dataset)
        self.action_sample_counters = defaultdict(int)
        self.total_samples = 0

    def map_mri_raw_action(self, raw_action: str):
        """
        Map an mRI raw label to the unified action name and class ID.

        Rules:
        - pose_1 .. pose_10 -> class_mapping['0' .. '9']
        - free_form          -> class_mapping['10']
        - walk               -> class_mapping['11']

        Returns:
            (action_name, class_id)
            or (None, None) if the label cannot be mapped
        """
        if not hasattr(self, "class_mapping") or not self.class_mapping:
            return None, None

        s = str(raw_action).strip().lower().replace("-", "_").replace(" ", "_")

        key_id = None

        # pose_1 ~ pose_10 -> 0 ~ 9
        m = re.match(r"pose[_\-]?(\d+)$", s)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 10:
                key_id = n - 1  # pose_1 -> 0, ..., pose_10 -> 9

        # free_form -> 10
        if key_id is None and s in {"free_form", "freeform"}:
            key_id = 10

        # walk -> 11
        if key_id is None and s in {"walk", "walking", "walking_in_a_straight_line"}:
            key_id = 11

        if key_id is None:
            # Other labels (e.g., 't pose') are not forcibly mapped here.
            return None, None

        action_name = _cm_get(self.class_mapping, key_id)
        if action_name is None:
            # The key does not exist in class_mapping.
            return None, None

        return action_name, int(key_id)

    def run(self):
        print(
            f"[mRI->CSV] root={self.csv_folder}, seg_dir={self.seg_path}, "
            f"T={self.window_size}, stride={self.stride}"
        )

        json_files = sorted([f for f in os.listdir(self.seg_path) if f.endswith(".json")])
        for jf in json_files:
            jpath = os.path.join(self.seg_path, jf)
            with open(jpath, "r", encoding="utf-8") as f:
                label_data = json.load(f)

            subject = label_data["subject"]
            person_id = parse_subject_to_id(subject)
            csv_file = f"{subject}.csv"
            csv_path = os.path.join(self.csv_folder, csv_file)
            if not os.path.isfile(csv_path):
                print(f"[WARN] radar CSV missing: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            need_cols = {"Frame #", "X", "Y", "Z", "Doppler", "Intensity"}
            if not need_cols.issubset(df.columns):
                print(f"[WARN] CSV columns missing in {csv_file}, need {need_cols}")
                continue

            frames_col = pd.to_numeric(df["Frame #"], errors="coerce").astype("Int64")

            # Slide the window independently within each labeled segment.
            labels_dict = label_data.get("labels", {})
            for raw_action, rng in labels_dict.items():
                raw_action, cid = self.map_mri_raw_action(raw_action)  # e.g., 'pose_3' -> (action_name, 2)
                if raw_action is not None:
                    if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
                        continue

                    a, b = int(rng[0]), int(rng[1])
                    # The interval is inclusive: [a, b]. Convert it to [a, b+1).
                    seg_start = a
                    seg_end_excl = b + 1

                    # Sliding window inside the segment.
                    s = seg_start
                    while s < seg_end_excl:
                        e = s + self.window_size
                        actual_end = min(e, seg_end_excl)

                        # Real number of frames in this window.
                        real_T = actual_end - s

                        # Map to the unified global action ID.
                        canon, uid = resolve_unified_action(
                            "mRI", raw_action, self.alias2action, self.dataset_alias2action, self.action2id
                        )

                        if uid is None:
                            # Skip this window if mapping fails.
                            s += self.stride
                            continue

                        sample_id = self._make_sample_id(uid, person_id)
                        rows = self._extract_points_in_range(df, frames_col, s, actual_end)

                        # Pad remaining frames with zero rows (one zero row per missing frame).
                        pad_frames = self.window_size - real_T
                        if pad_frames > 0:
                            for rel in range(real_T, self.window_size):
                                rows.append([rel, 0.0, 0.0, 0.0, 0.0, 0.0])

                        # Write the file as long as the window contains at least one row.
                        if rows:
                            out_path = os.path.join(self.samples_dir, f"{sample_id}.csv")
                            out_df = pd.DataFrame(
                                rows,
                                columns=["Frame", "X", "Y", "Z", "Doppler", "Intensity"],
                            )
                            out_df.to_csv(out_path, index=False)
                            self.total_samples += 1

                        s += self.stride

            print(f"{csv_file} done!")

        print(f"[DONE] total samples written: {self.total_samples}")

    def _extract_points_in_range(self, df: pd.DataFrame, frames_col: pd.Series, s: int, e: int):
        """
        Return rows in the form:
        [ [rel, X, Y, Z, Doppler, Intensity], ... ]

        Only frames t in [s, e) are included.
        rel = t - s
        All points inside each frame are written row by row without sampling.
        """
        out = []
        for t in range(s, e):
            mask = (frames_col == t)
            if not mask.any():
                continue

            sub = df.loc[mask, ["X", "Y", "Z", "Doppler", "Intensity"]]
            rel = t - s
            for _, r in sub.iterrows():
                out.append(
                    [
                        rel,
                        float(r["X"]),
                        float(r["Y"]),
                        float(r["Z"]),
                        float(r["Doppler"]),
                        float(r["Intensity"]),
                    ]
                )
        return out

    def _make_sample_id(self, unified_id: int, person_id: int) -> str:
        A = unified_id  # All indices are zero-based here.
        P = int(person_id) - 1
        self.action_sample_counters[unified_id] += 1
        S = self.action_sample_counters[unified_id]
        return f"D{self.dataset_code:03d}A{A:03d}P{P:03d}S{S:04d}"


def build_args():
    ap = argparse.ArgumentParser("Convert MMFI dataset to CSV format (NTU style)")
    ap.add_argument(
        "--dataset",
        type=str,
        default="mRI",
        choices=["FastHAR", "RadHAR", "MMFI", "mRI"],
        help="Which dataset to convert",
    )
    ap.add_argument("--dataset-root", type=str, default="mRI/raw_data/radar", help="Path to the dataset")
    ap.add_argument("--seg-path", type=str, default="mRI/raw_data/videolabels", help="Path to the segment labels")
    ap.add_argument("--window-size", default=32, help="Window size")
    ap.add_argument("--stride", type=str, default=16, help="Sliding window stride")
    ap.add_argument("--output-dir", type=str, default="UniMM-HAR/csv/mRI/TEST")
    ap.add_argument("--label-csv-path", type=str, default="UniMM-HAR/label.csv", help="Path to UniMM-HAR/label.csv")
    ap.add_argument("--class-map", type=str, default="UniMM-HAR/mRI/class_mapping.json", help="Action-to-ID JSON file")
    ap.add_argument(
        "--ds-map",
        type=str,
        default="UniMM-HAR/ds_mapping.json",
        help="Dataset code mapping JSON file (e.g., {'1': 'RadHAR', '4': 'FastHAR'})",
    )

    return ap.parse_args()


def main():
    args = build_args()
    conv = mRICSVFormatConverter(args)
    conv.run()


if __name__ == "__main__":
    main()