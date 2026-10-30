"""
evaluation_metrics.py — Vision System Accuracy Evaluation

This script compares your system's predictions with manually-defined ground truth
and calculates accuracy metrics for:
  - Person Detection
  - Helmet Detection  
  - Zone Violation Detection

Usage:
  1. Define expected outputs in ground_truth.csv (see GROUND_TRUTH_TEMPLATE.md)
  2. Collect system predictions in predictions.csv (same format as ground truth)
  3. Run: python evaluation_metrics.py
  4. Get accuracy percentages and error analysis

Input Format (predictions.csv):
  frame_id,timestamp_seconds,person_present,helmet_visible,in_restricted_zone
  (optional fields; at minimum need frame_id and one attribute)
"""

from pathlib import Path
from typing import List, Dict, Tuple
import csv


class GroundTruth:
    """Load and store ground truth from CSV."""

    def __init__(self, csv_path: str):
        self.data: Dict[int, Dict] = {}
        self.load(csv_path)

    def load(self, csv_path: str) -> None:
        """Load ground truth from CSV file."""
        path = Path(csv_path)
        if not path.exists():
            print(f"[ERROR] Ground truth file not found: {csv_path}")
            return

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_id = int(row["frame_id"])
                self.data[frame_id] = {
                    "person_present": row["person_present"].strip(),
                    "helmet_visible": row["helmet_visible"].strip(),
                    "in_restricted_zone": row["in_restricted_zone"].strip(),
                }
        print(f"[INFO] Loaded {len(self.data)} ground truth entries")

    def get(self, frame_id: int) -> Dict | None:
        """Get ground truth for a frame."""
        return self.data.get(frame_id)


class Predictions:
    """Load and store system predictions from CSV."""

    def __init__(self, csv_path: str):
        self.data: Dict[int, Dict] = {}
        self.load(csv_path)

    def load(self, csv_path: str) -> None:
        """Load predictions from CSV file."""
        path = Path(csv_path)
        if not path.exists():
            print(f"[ERROR] Predictions file not found: {csv_path}")
            return

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_id = int(row["frame_id"])
                self.data[frame_id] = {
                    "person_present": row.get("person_present", "Unknown").strip(),
                    "helmet_visible": row.get("helmet_visible", "Unknown").strip(),
                    "in_restricted_zone": row.get("in_restricted_zone", "Unknown").strip(),
                }
        print(f"[INFO] Loaded {len(self.data)} predictions")

    def get(self, frame_id: int) -> Dict | None:
        """Get prediction for a frame."""
        return self.data.get(frame_id)


class Evaluator:
    """Compare predictions with ground truth and compute metrics."""

    def __init__(self, ground_truth: GroundTruth, predictions: Predictions):
        self.gt = ground_truth
        self.pred = predictions
        self.results = {
            "person_detection": {"tp": 0, "fp": 0, "fn": 0},
            "helmet_detection": {"tp": 0, "fp": 0, "fn": 0},
            "zone_detection": {"tp": 0, "fp": 0, "fn": 0},
        }
        self.mismatches: List[Dict] = []

    def evaluate(self) -> None:
        """Compare all frames and compute metrics."""
        all_frames = set(self.gt.data.keys()) | set(self.pred.data.keys())

        for frame_id in sorted(all_frames):
            gt = self.gt.get(frame_id)
            pred = self.pred.get(frame_id)

            if gt is None:
                print(f"[WARN] Frame {frame_id} in predictions but not in ground truth")
                continue

            if pred is None:
                print(f"[WARN] Frame {frame_id} in ground truth but no prediction")
                # Treat missing predictions as False predictions
                pred = {
                    "person_present": "No",
                    "helmet_visible": "N/A",
                    "in_restricted_zone": "N/A",
                }

            self._compare_frame(frame_id, gt, pred)

    def _compare_frame(self, frame_id: int, gt: Dict, pred: Dict) -> None:
        """Compare a single frame's ground truth vs prediction."""
        gt_person = gt["person_present"] == "Yes"
        pred_person = pred["person_present"] == "Yes"

        # Person Detection
        if gt_person and pred_person:
            self.results["person_detection"]["tp"] += 1
        elif gt_person and not pred_person:
            self.results["person_detection"]["fn"] += 1
            self.mismatches.append(
                {
                    "frame": frame_id,
                    "type": "person",
                    "error": "MISSED_DETECTION",
                    "expected": "Yes",
                    "predicted": "No",
                }
            )
        elif not gt_person and pred_person:
            self.results["person_detection"]["fp"] += 1
            self.mismatches.append(
                {
                    "frame": frame_id,
                    "type": "person",
                    "error": "FALSE_POSITIVE",
                    "expected": "No",
                    "predicted": "Yes",
                }
            )

        # Helmet Detection (only evaluate if both GT and pred have person)
        if gt_person and pred_person:
            gt_helmet = gt["helmet_visible"] == "Yes"
            pred_helmet = pred["helmet_visible"] == "Yes"

            if gt_helmet and pred_helmet:
                self.results["helmet_detection"]["tp"] += 1
            elif gt_helmet and not pred_helmet:
                self.results["helmet_detection"]["fn"] += 1
                self.mismatches.append(
                    {
                        "frame": frame_id,
                        "type": "helmet",
                        "error": "FALSE_NEGATIVE (helmet missed)",
                        "expected": "Yes",
                        "predicted": "No",
                    }
                )
            elif not gt_helmet and pred_helmet:
                self.results["helmet_detection"]["fp"] += 1
                self.mismatches.append(
                    {
                        "frame": frame_id,
                        "type": "helmet",
                        "error": "FALSE_POSITIVE (detected when none)",
                        "expected": "No",
                        "predicted": "Yes",
                    }
                )

        # Zone Violation Detection (only evaluate if both GT and pred have person)
        if gt_person and pred_person:
            gt_zone = gt["in_restricted_zone"] == "Yes"
            pred_zone = pred["in_restricted_zone"] == "Yes"

            if gt_zone and pred_zone:
                self.results["zone_detection"]["tp"] += 1
            elif gt_zone and not pred_zone:
                self.results["zone_detection"]["fn"] += 1
                self.mismatches.append(
                    {
                        "frame": frame_id,
                        "type": "zone",
                        "error": "MISSED_VIOLATION",
                        "expected": "Yes",
                        "predicted": "No",
                    }
                )
            elif not gt_zone and pred_zone:
                self.results["zone_detection"]["fp"] += 1
                self.mismatches.append(
                    {
                        "frame": frame_id,
                        "type": "zone",
                        "error": "FALSE_ALARM",
                        "expected": "No",
                        "predicted": "Yes",
                    }
                )

    def print_metrics(self) -> None:
        """Print accuracy metrics."""
        print("\n" + "=" * 70)
        print("VISION SYSTEM EVALUATION METRICS")
        print("=" * 70)

        for metric_name, counts in self.results.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]

            total = tp + fp + fn
            if total == 0:
                accuracy = 0.0
                precision = 0.0
                recall = 0.0
            else:
                accuracy = (tp / total) * 100 if total > 0 else 0
                precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
                recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0

            print(f"\n[{metric_name.upper()}]")
            print(f"  True Positives (TP):  {tp}")
            print(f"  False Positives (FP): {fp}")
            print(f"  False Negatives (FN): {fn}")
            print(f"  Total Predictions:    {total}")
            print(f"  ---")
            print(f"  Accuracy:  {accuracy:6.2f}%")
            print(f"  Precision: {precision:6.2f}%  (TP / (TP+FP))")
            print(f"  Recall:    {recall:6.2f}%    (TP / (TP+FN))")

    def print_errors(self, top_n: int = 10) -> None:
        """Print top mismatches for debugging."""
        if not self.mismatches:
            print("\n[NO ERRORS] All predictions matched ground truth!")
            return

        print(f"\n[ERROR ANALYSIS - Top {min(top_n, len(self.mismatches))} Mismatches]")
        print("-" * 70)
        for i, error in enumerate(self.mismatches[:top_n], 1):
            print(
                f"{i}. Frame {error['frame']:4d} | Type: {error['type']:8s} | "
                f"{error['error']:30s} | GT={error['expected']} Pred={error['predicted']}"
            )

    def save_report(self, output_path: str = "evaluation_report.txt") -> None:
        """Save evaluation report to file."""
        with open(output_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("VISION SYSTEM EVALUATION REPORT\n")
            f.write("=" * 70 + "\n")

            for metric_name, counts in self.results.items():
                tp = counts["tp"]
                fp = counts["fp"]
                fn = counts["fn"]
                total = tp + fp + fn
                accuracy = (tp / total) * 100 if total > 0 else 0

                f.write(f"\n[{metric_name.upper()}]\n")
                f.write(f"  TP={tp}, FP={fp}, FN={fn}, Total={total}\n")
                f.write(f"  Accuracy: {accuracy:.2f}%\n")

            if self.mismatches:
                f.write(f"\n[ERRORS]\n")
                for error in self.mismatches:
                    f.write(
                        f"  Frame {error['frame']}: {error['type']} - {error['error']}\n"
                    )

        print(f"\n[INFO] Report saved to {output_path}")


def main() -> None:
    # Load ground truth and predictions
    gt = GroundTruth("ground_truth.csv")
    predictions = Predictions("predictions.csv")

    if not gt.data or not predictions.data:
        print("[ERROR] Could not load files. Exiting.")
        return

    # Evaluate
    evaluator = Evaluator(gt, predictions)
    evaluator.evaluate()

    # Display results
    evaluator.print_metrics()
    evaluator.print_errors(top_n=15)
    evaluator.save_report()


if __name__ == "__main__":
    main()
