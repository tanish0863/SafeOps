"""Summarise tracks.jsonl for the six tracking test scenarios.

Reports what can be measured WITHOUT ground truth: unique IDs, track
lifetimes, gaps (possible fragmentation) and short tracks. ID switches, IDF1
and HOTA need manually annotated ground-truth identities and are not computed.

Usage: python -m src.tracking.track_stats outputs/tracks.jsonl [--total-frames N]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict


def summarize(path: str, total_frames: int | None = None, short_track_frames: int = 15) -> dict:
    frames_by_id: dict[int, list[int]] = defaultdict(list)
    workers_per_frame: dict[int, int] = defaultdict(int)
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            frames_by_id[r["track_id"]].append(r["frame_number"])
            workers_per_frame[r["frame_number"]] += 1

    tracks = {}
    for tid, frames in sorted(frames_by_id.items()):
        frames = sorted(frames)
        first, last = frames[0], frames[-1]
        gaps = sum(1 for a, b in zip(frames, frames[1:]) if b - a > 1)
        tracks[tid] = {
            "first_frame": first,
            "last_frame": last,
            "frames_present": len(frames),
            "gaps": gaps,
            "missing_frames_in_span": (last - first + 1) - len(frames),
            "short_track": len(frames) < short_track_frames,
        }

    out = {
        "unique_track_ids": len(tracks),
        "max_workers_in_one_frame": max(workers_per_frame.values(), default=0),
        "frames_with_workers": len(workers_per_frame),
        "short_tracks": [t for t, v in tracks.items() if v["short_track"]],
        "tracks": tracks,
    }
    if total_frames:
        out["frames_without_workers"] = total_frames - len(workers_per_frame)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--total-frames", type=int)
    ap.add_argument("--short-track-frames", type=int, default=15)
    a = ap.parse_args()
    s = summarize(a.jsonl, a.total_frames, a.short_track_frames)
    print(f"Unique track IDs:          {s['unique_track_ids']}")
    print(f"Max workers in one frame:  {s['max_workers_in_one_frame']}")
    print(f"Frames with workers:       {s['frames_with_workers']}")
    if "frames_without_workers" in s:
        print(f"Frames without workers:    {s['frames_without_workers']}")
    print(f"Short tracks (possible fragments): {s['short_tracks']}")
    print("\nID    first  last   present  gaps  missing")
    for tid, t in s["tracks"].items():
        print(f"{tid:<5} {t['first_frame']:<6} {t['last_frame']:<6} {t['frames_present']:<8} {t['gaps']:<5} {t['missing_frames_in_span']}")
    print("\nUnique IDs > actual people in the clip suggests fragmentation or ID switches;")
    print("confirm against the video or ground-truth labels.")


if __name__ == "__main__":
    main()
