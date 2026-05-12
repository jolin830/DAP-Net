# read_label_csv.py

import os
import json
import re
from typing import Dict, Tuple

import pandas as pd


def _norm_text(s: str) -> str:
    """Normalize text by trimming spaces, merging consecutive spaces, and converting to lowercase."""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_dataset_name(s: str) -> str:
    """Normalize dataset names into a unified format."""
    s = _norm_text(s)
    s = s.replace("_", "-")

    # Normalize common aliases.
    if s in {"mmfi", "mm-fi", "mm-fi ", "mm fi"}:
        return "mm-fi"

    if s in {"fasthar", "fast-har"}:
        return "fasthar"

    if s in {"radhar", "rad-har"}:
        return "radhar"

    if s in {"mri"}:
        return "mri"

    return s


def _split_sources_cell(cell: str):
    """
    Split a Source field into multiple 'dataset: alias' pairs.

    Supported separators:
    - comma (,)
    - Chinese comma (，)
    - semicolon (;)
    """
    parts = re.split(r"[;,，]", str(cell))
    out = []

    for p in parts:
        p = p.strip()
        if not p:
            continue

        if ":" in p:
            ds, alias = p.split(":", 1)
            out.append((_norm_dataset_name(ds), _norm_text(alias)))
        else:
            # Leave dataset name empty if it is not provided.
            out.append(("", _norm_text(p)))

    return out


def read_label_csv(label_csv_path: str):
    """
    Read label.csv and return:

    (
        action2id,
        id2action,
        alias2action,
        dataset_alias2action,
        action_counts
    )
    """
    df = pd.read_csv(label_csv_path, encoding="utf-8-sig")

    action2id: Dict[str, int] = {}
    id2action: Dict[int, str] = {}
    alias2action: Dict[str, str] = {}
    dataset_alias2action: Dict[Tuple[str, str], str] = {}
    action_counts: Dict[str, int] = {}

    for _, row in df.iterrows():
        cid = int(row["ID"])
        action_name = str(row["Action"]).strip()
        action_name_norm = _norm_text(action_name)
        count = int(row["Sequence"]) if pd.notna(row["Sequence"]) else 0

        # Build the primary mappings.
        action2id[action_name] = cid
        id2action[cid] = action_name
        action_counts[action_name] = count

        # The Source field may contain aliases from different datasets.
        # Build alias -> canonical action mappings.
        for ds_norm, alias_norm in _split_sources_cell(row["Source"]):
            if alias_norm:
                # Dataset-independent alias mapping.
                alias2action[alias_norm] = action_name

                # Dataset-specific alias mapping.
                # This mapping has higher priority when ambiguity exists.
                if ds_norm:
                    dataset_alias2action[(ds_norm, alias_norm)] = action_name

        # Add the canonical action name itself into alias mappings.
        alias2action[action_name_norm] = action_name
        dataset_alias2action[("", action_name_norm)] = action_name

    return (
        action2id,
        id2action,
        alias2action,
        dataset_alias2action,
        action_counts,
    )


def resolve_unified_action(
    dataset_name: str,
    raw_action_name: str,
    alias2action: Dict[str, str],
    dataset_alias2action: Dict[Tuple[str, str], str],
    action2id: Dict[str, int],
):
    """
    Resolve a raw dataset-specific action into the unified action space.

    Matching priority:
    1. Exact match on (dataset, alias)
    2. Alias-only match
    3. Return (None, None) if matching fails
    """
    ds = _norm_dataset_name(dataset_name)
    alias = _norm_text(raw_action_name)

    # First try (dataset, alias).
    key = (ds, alias)
    if key in dataset_alias2action:
        canon = dataset_alias2action[key]
        return canon, action2id[canon]

    # Then try alias-only matching.
    if alias in alias2action:
        canon = alias2action[alias]
        return canon, action2id[canon]

    return None, None


def get_dataset_code(ds_map_path: str, dataset_name: str, default_code: int = 1) -> int:
    """
    Example ds_mapping.json format:

    {
        "1": "RadHAR",
        "2": "mRI",
        "3": "MM-Fi",
        "4": "FastHAR"
    }
    """
    if not os.path.isfile(ds_map_path):
        return int(default_code)

    with open(ds_map_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    # Support both formats:
    # {"1":"RadHAR", ...}
    # {"RadHAR":1, ...}
    if all(k.isdigit() for k in m.keys()):  # id -> name
        for k, v in m.items():
            if str(v).lower() == dataset_name.lower():
                return int(k)
    else:  # name -> id
        return int(m.get(dataset_name, default_code))

    return int(default_code)


if __name__ == "__main__":
    label_csv = "label.csv"

    a2i, i2a, a_alias, ds_alias, counts = read_label_csv(label_csv)

    # Example:
    # Map the mRI action "walking in a straight line"
    # into the unified label space.
    canon, cid = resolve_unified_action(
        "mRI",
        "walking in a straight line",
        a_alias,
        ds_alias,
        a2i,
    )

    print("resolve:", canon, cid)

    # Save class_mapping (action_name -> id)
    # for use in other scripts.
    with open("class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(a2i, f, ensure_ascii=False, indent=2)