"""Run ONCE while online. Weights are then stored in models/ for fully offline use.
Usage: python scripts/download_model.py [yolo26n-pose.pt | yolo11n-pose.pt]"""
import sys
from pathlib import Path

from ultralytics import YOLO

name = sys.argv[1] if len(sys.argv) > 1 else "yolo26n-pose.pt"
Path("models").mkdir(exist_ok=True)
YOLO(name)  # downloads into the current directory
src = Path(name)
if src.exists():
    src.replace(Path("models") / name)
    print("saved to", Path("models") / name)
else:
    print("download location not found; move the .pt file into models/ manually")
