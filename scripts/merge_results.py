#!/usr/bin/env python3
"""Merge per-config sweep CSVs into a single file.

Reads all output/sweep_config_*.csv files and concatenates them into
output/full_sweep_results.csv.  Reports how many configs completed and
flags any missing indices.

Usage:
    python scripts/merge_results.py
"""

import glob
import re
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    output_dir = Path("output")
    pattern = str(output_dir / "sweep_config_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files matching {pattern}")
        sys.exit(1)

    # Detect which config indices are present
    found_indices: set[int] = set()
    for f in files:
        m = re.search(r"sweep_config_(\d+)\.csv", f)
        if m:
            found_indices.add(int(m.group(1)))

    expected = set(range(54))
    missing = sorted(expected - found_indices)

    print(f"Found {len(files)} config files")
    if missing:
        print(f"Missing configs: {missing}")
    else:
        print("All 54 configs present")

    # Concatenate
    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    out_path = output_dir / "full_sweep_results.csv"
    merged.to_csv(out_path, index=False)

    print(f"Total rows: {len(merged)}")
    print(f"Merged file: {out_path}")


if __name__ == "__main__":
    main()
