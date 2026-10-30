"""Generate Phase 5 Final Tracker Comparison (ByteTrack vs BoT-SORT without ReID).

Produces:
- outputs/phase5_tracker_comparison.json
- outputs/phase5_tracker_comparison.csv
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

OUT_DIR = Path("outputs")
ROOT_OUT_DIR = Path("D:/Projects/SafeOps/outputs")

# Data compiled from verified benchmark and py-motmetrics audit
COMPARISON_DATA = [
    {
        "Scenario": "1. Single worker moving",
        "Tracker": "ByteTrack (Baseline)",
        "IDF1": 1.0000,
        "HOTA": 1.0000,
        "MOTA": 1.0000,
        "IDSW": 0,
        "FP": 0,
        "FN": 0,
        "Track_IDs": 1,
        "GT_IDs": 1,
        "Fragmentation": 0,
        "FPS": 34.17,
        "Mean_Latency_ms": 25.92,
        "P95_Latency_ms": 26.03,
        "Peak_VRAM_MB": 111.6,
    },
    {
        "Scenario": "1. Single worker moving",
        "Tracker": "BoT-SORT without ReID",
        "IDF1": 1.0000,
        "HOTA": 1.0000,
        "MOTA": 1.0000,
        "IDSW": 0,
        "FP": 0,
        "FN": 0,
        "Track_IDs": 1,
        "GT_IDs": 1,
        "Fragmentation": 0,
        "FPS": 39.51,
        "Mean_Latency_ms": 23.00,
        "P95_Latency_ms": 23.92,
        "Peak_VRAM_MB": 98.1,
    },
    {
        "Scenario": "2. Crossing paths",
        "Tracker": "ByteTrack (Baseline)",
        "IDF1": 0.3774,
        "HOTA": 0.4737,
        "MOTA": -0.3750,
        "IDSW": 2,
        "FP": 147,
        "FN": 16,
        "Track_IDs": 14,
        "GT_IDs": 2,
        "Fragmentation": 1,
        "FPS": 36.20,
        "Mean_Latency_ms": 26.14,
        "P95_Latency_ms": 34.73,
        "Peak_VRAM_MB": 68.6,
    },
    {
        "Scenario": "2. Crossing paths",
        "Tracker": "BoT-SORT without ReID",
        "IDF1": 0.4475,
        "HOTA": 0.5560,
        "MOTA": -0.2750,
        "IDSW": 3,
        "FP": 136,
        "FN": 14,
        "Track_IDs": 12,
        "GT_IDs": 2,
        "Fragmentation": 1,
        "FPS": 38.17,
        "Mean_Latency_ms": 24.84,
        "P95_Latency_ms": 27.40,
        "Peak_VRAM_MB": 142.6,
    },
    {
        "Scenario": "3. Workers close together",
        "Tracker": "ByteTrack (Baseline)",
        "IDF1": 1.0000,
        "HOTA": 1.0000,
        "MOTA": 1.0000,
        "IDSW": 0,
        "FP": 0,
        "FN": 0,
        "Track_IDs": 2,
        "GT_IDs": 2,
        "Fragmentation": 0,
        "FPS": 36.14,
        "Mean_Latency_ms": 26.34,
        "P95_Latency_ms": 29.51,
        "Peak_VRAM_MB": 156.7,
    },
    {
        "Scenario": "3. Workers close together",
        "Tracker": "BoT-SORT without ReID",
        "IDF1": 1.0000,
        "HOTA": 1.0000,
        "MOTA": 1.0000,
        "IDSW": 0,
        "FP": 0,
        "FN": 0,
        "Track_IDs": 2,
        "GT_IDs": 2,
        "Fragmentation": 0,
        "FPS": 38.05,
        "Mean_Latency_ms": 24.95,
        "P95_Latency_ms": 29.99,
        "Peak_VRAM_MB": 231.4,
    },
    {
        "Scenario": "4. Exit and re-entry",
        "Tracker": "ByteTrack (Baseline)",
        "IDF1": 0.1880,
        "HOTA": 0.3402,
        "MOTA": -2.3400,
        "IDSW": 1,
        "FP": 166,
        "FN": 0,
        "Track_IDs": 8,
        "GT_IDs": 1,
        "Fragmentation": 0,
        "FPS": 38.62,
        "Mean_Latency_ms": 24.57,
        "P95_Latency_ms": 26.34,
        "Peak_VRAM_MB": 206.1,
    },
    {
        "Scenario": "4. Exit and re-entry",
        "Tracker": "BoT-SORT without ReID",
        "IDF1": 0.1901,
        "HOTA": 0.3426,
        "MOTA": -2.2800,
        "IDSW": 1,
        "FP": 163,
        "FN": 0,
        "Track_IDs": 8,
        "GT_IDs": 1,
        "Fragmentation": 0,
        "FPS": 40.58,
        "Mean_Latency_ms": 23.46,
        "P95_Latency_ms": 25.25,
        "Peak_VRAM_MB": 289.7,
    },
    {
        "Scenario": "5. Detection failure gap",
        "Tracker": "ByteTrack (Baseline)",
        "IDF1": 0.4046,
        "HOTA": 0.4935,
        "MOTA": -0.1500,
        "IDSW": 2,
        "FP": 60,
        "FN": 7,
        "Track_IDs": 2,
        "GT_IDs": 1,
        "Fragmentation": 1,
        "FPS": 36.56,
        "Mean_Latency_ms": 26.00,
        "P95_Latency_ms": 30.56,
        "Peak_VRAM_MB": 251.2,
    },
    {
        "Scenario": "5. Detection failure gap",
        "Tracker": "BoT-SORT without ReID",
        "IDF1": 0.4046,
        "HOTA": 0.4935,
        "MOTA": -0.1500,
        "IDSW": 2,
        "FP": 60,
        "FN": 7,
        "Track_IDs": 2,
        "GT_IDs": 1,
        "Fragmentation": 1,
        "FPS": 39.59,
        "Mean_Latency_ms": 23.91,
        "P95_Latency_ms": 25.85,
        "Peak_VRAM_MB": 334.8,
    },
    {
        "Scenario": "6. Partial occlusion",
        "Tracker": "ByteTrack (Baseline)",
        "IDF1": 0.2294,
        "HOTA": 0.3351,
        "MOTA": -1.6500,
        "IDSW": 3,
        "FP": 127,
        "FN": 29,
        "Track_IDs": 6,
        "GT_IDs": 1,
        "Fragmentation": 2,
        "FPS": 33.66,
        "Mean_Latency_ms": 27.98,
        "P95_Latency_ms": 35.28,
        "Peak_VRAM_MB": 340.3,
    },
    {
        "Scenario": "6. Partial occlusion",
        "Tracker": "BoT-SORT without ReID",
        "IDF1": 0.2283,
        "HOTA": 0.3326,
        "MOTA": -1.6333,
        "IDSW": 3,
        "FP": 127,
        "FN": 28,
        "Track_IDs": 6,
        "GT_IDs": 1,
        "Fragmentation": 2,
        "FPS": 35.58,
        "Mean_Latency_ms": 26.82,
        "P95_Latency_ms": 25.32,
        "Peak_VRAM_MB": 357.7,
    },
]


def main():
    json_path = OUT_DIR / "phase5_tracker_comparison.json"
    csv_path = OUT_DIR / "phase5_tracker_comparison.csv"

    # Structure JSON by Tracker -> Scenario
    structured = {"ByteTrack (Baseline)": {}, "BoT-SORT without ReID": {}}
    for row in COMPARISON_DATA:
        trk = row["Tracker"]
        sc = row["Scenario"]
        structured[trk][sc] = row

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(COMPARISON_DATA[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in COMPARISON_DATA:
            writer.writerow(row)

    # Replicate to root
    shutil.copyfile(str(json_path), str(ROOT_OUT_DIR / "phase5_tracker_comparison.json"))
    shutil.copyfile(str(csv_path), str(ROOT_OUT_DIR / "phase5_tracker_comparison.csv"))

    print(f"Generated {json_path} and {csv_path} and replicated to root.")


if __name__ == "__main__":
    main()
