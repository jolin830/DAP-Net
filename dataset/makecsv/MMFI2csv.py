import os
import csv
import glob
import json
import argparse
import sys
from datetime import datetime
from collections import defaultdict
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

from read_label_csv import *


class MMFICSVFormatConverter:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset
        self.output_dir = args.output_dir
        self.samples_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(self.samples_dir, exist_ok=True)

        (
            self.action2id,
            self.id2action,
            self.alias2action,
            self.dataset_alias2action,
            self.action_counts,
        ) = read_label_csv(args.label_csv_path)

        self.dataset_code = get_dataset_code(args.ds_map, args.dataset)

        if hasattr(args, "class_map") and args.class_map:
            with open(args.class_map, "r", encoding="utf-8") as f:
                self.class_mapping = json.load(f)

        self.action_sample_counters = defaultdict(int)
        self.splits = {"train": [], "val": [], "test": []}
        self.stats = {
            "total_samples": 0,
            "total_frames": 0,
            "min_frames": float("inf"),
            "max_frames": 0,
        }

    def correct_str(self, ori_str):
        return str(int(ori_str[1:]) - 1)

    def convert_to_csv(self, dataset_name: str, root_dir: str, seg_path: Optional[str] = None):
        """
        Convert the MMFI dataset to CSV format and save each sample as a separate CSV file.
        """
        print(f"Start converting {dataset_name} dataset to CSV format...")

        if seg_path is None:
            raise ValueError("seg_path is required for MMFI CSV conversion.")

        with open(seg_path, "r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                scene = row["Environment"].strip()
                subject = row["Student"].strip()
                action = row["Action"].strip()
                label = self.correct_str(action)  # Convert "A01" -> "0"
                segments = self._parse_segments_field(row["Segments"])

                for start, end in segments:
                    sample_data = {
                        "scene": scene,
                        "subject": subject,
                        "action": action,
                        "start": start,
                        "end": end,
                    }

                    unified_action, unified_id, original_action = self._resolve_dataset_action(label)

                    frames = self._load_frames(root_dir, scene, subject, action, start, end)
                    if frames != []:
                        sample_id = self._generate_sample_id(unified_id, scene, subject)
                        self.stats["total_samples"] += 1

                        self._save_csv(sample_id, frames, action, subject, unified_id)
                        print(f"Saved {sample_id}")

    def _resolve_dataset_action(self, original_label, original_action_name=None):
        """Map a dataset-specific action label to the unified action space."""
        if self.class_mapping:
            # Use the class mapping if available.
            if str(original_label) in self.class_mapping:
                original_action = self.class_mapping[str(original_label)]
            else:
                original_action = f"class_{original_label}"
        else:
            # Fall back to the original label or the provided action name.
            original_action = original_action_name or f"class_{original_label}"

        unified_action, unified_id = resolve_unified_action(
            self.dataset_name.lower(),
            original_action,
            self.alias2action,
            self.dataset_alias2action,
            self.action2id,
        )

        return unified_action, unified_id, original_action

    def _parse_segments_field(self, seg_str: str) -> List[Tuple[int, int]]:
        # Example: "1-7; 8-15; 16-21" -> [(1, 7), (8, 15), (16, 21)]
        chunks = [s.strip() for s in seg_str.split(";") if s.strip()]
        segs = []
        for ch in chunks:
            if "-" in ch:
                a, b = ch.split("-", 1)
                segs.append((int(a.strip()), int(b.strip())))
        return segs

    def _generate_sample_id(self, unified_id, scene, subject):
        """Generate an NTU-style sample ID."""
        scene = self.correct_str(scene)
        subject = self.correct_str(subject)
        self.action_sample_counters[unified_id] += 1
        sample_count = self.action_sample_counters[unified_id]
        sample_id = (
            f"D{self.dataset_code:03d}"
            f"A{unified_id:03d}"
            f"E{int(scene):03d}"
            f"P{int(subject):03d}"
            f"S{sample_count:04d}"
        )
        return sample_id

    def _load_frames(self, root_dir, scene, subject, action, start, end):
        """Load frame data and return the raw point clouds for the given interval."""
        paths = [
            os.path.join(root_dir, scene, subject, action, "mmwave", f"frame{t:03d}.bin")
            for t in range(start, end + 1)
        ]
        frames = []  # Store the point cloud data of each frame
        for p in paths:
            frames.append(self._read_mmwave_bin(p))
        return frames

    def _read_mmwave_bin(self, file_path):
        """Parse a single frame binary file."""
        # This function reads the raw point cloud data for each frame
        # Format: Frame, X, Y, Z, Doppler, Intensity
        # Returns: ndarray with shape [N, 6]
        with open(file_path, "rb") as f:
            raw = f.read()

        arr = np.frombuffer(raw, dtype=np.float64).reshape(-1, 5).astype(np.float32)

        return arr

    def _save_csv(self, sample_id, frames, action, subject, unified_id):
        """Save the sample as a CSV file."""
        sample_path = os.path.join(self.samples_dir, f"{sample_id}.csv")
        all_data = []

        for t, frame in enumerate(frames):
            frame_data = np.column_stack((np.full(len(frame), t), frame))
            all_data.append(frame_data)

        all_data = np.vstack(all_data)
        df = pd.DataFrame(
            all_data,
            columns=["Frame", "X", "Y", "Z", "Doppler", "Intensity"],
        )
        df.to_csv(sample_path, index=False)

    def finalize(self):
        dataset_info = {
            "total_samples": self.stats["total_samples"],
            "total_frames": self.stats["total_frames"],
            "min_frames": self.stats["min_frames"],
            "max_frames": self.stats["max_frames"],
            "creation_date": datetime.now().isoformat(),
        }
        with open(os.path.join(self.output_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2)


def build_args():
    """Build command-line arguments."""
    ap = argparse.ArgumentParser("Convert MMFI dataset to CSV format (NTU style)")
    ap.add_argument(
        "--dataset",
        type=str,
        default="MMFI",
        choices=["FastHAR", "RadHAR", "MMFI"],
        help="Dataset to convert",
    )
    ap.add_argument("--dataset-root", type=str, default="MMFi_Dataset/unzip_files", help="Path to the dataset")
    ap.add_argument("--seg-path", type=str, default="MMFi_Dataset/MMFi_action_segments.csv", help="Path to the segment file")
    ap.add_argument("--output-dir", type=str, default="UniMM-HAR/csv/MMFI")
    ap.add_argument("--label-csv-path", type=str, default="UniMM-HAR/label.csv", help="Path to UniMM-HAR/label.csv")
    ap.add_argument("--class-map", type=str, default="UniMM-HAR/MMFI/class_mapping.json", help="Action-to-ID JSON file")
    ap.add_argument(
        "--ds-map",
        type=str,
        default="UniMM-HAR/ds_mapping.json",
        help="Dataset code mapping JSON file (e.g., {'1': 'RadHAR', '4': 'FastHAR'})",
    )

    return ap.parse_args()


def main():
    args = build_args()
    converter = MMFICSVFormatConverter(args)

    if args.dataset == "FastHAR":
        converter.convert_to_csv("FastHAR", args.dataset_root)
    elif args.dataset == "RadHAR":
        converter.convert_to_csv("RadHAR", args.dataset_root)
    elif args.dataset == "MMFI":
        converter.convert_to_csv("MMFI", args.dataset_root, args.seg_path)

    converter.finalize()


if __name__ == "__main__":
    main()