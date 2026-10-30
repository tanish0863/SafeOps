#!/usr/bin/env python3
"""
train_ppe_model.py

Train a YOLOv8 helmet detector on a YOLO-format dataset.

Expected dataset structure:
dataset/
 ├── train/images
 ├── train/labels
 ├── valid/images
 ├── valid/labels
 └── data.yaml

If data.yaml is missing, this script can generate it automatically.

Example:
    python train_ppe_model.py --dataset D:/datasets/ppe --epochs 50 --model yolov8s.pt
"""

from __future__ import annotations

import argparse
import shutil
import traceback
from pathlib import Path

from ultralytics import YOLO

from validate_ppe_dataset import validate_split, write_data_yaml


def is_dataset_ready(dataset_root: Path) -> bool:
    return (dataset_root / "train" / "images").is_dir() and (dataset_root / "train" / "labels").is_dir() and (dataset_root / "valid" / "images").is_dir() and (dataset_root / "valid" / "labels").is_dir()


def prepare_dataset_split(source_root: Path, output_root: Path, val_ratio: float = 0.2, seed: int = 42) -> Path:
    """Create train/valid folders from a single-source train export."""
    import random
    import shutil

    rng = random.Random(seed)
    source_images = source_root / "train" / "images"
    source_labels = source_root / "train" / "labels"

    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError(f"Source dataset must contain train/images and train/labels: {source_root}")

    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output_root / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (output_root / "valid" / "images").mkdir(parents=True, exist_ok=True)
    (output_root / "valid" / "labels").mkdir(parents=True, exist_ok=True)

    image_paths = sorted([path for path in source_images.iterdir() if path.is_file()])
    rng.shuffle(image_paths)
    valid_count = max(1, int(len(image_paths) * val_ratio)) if len(image_paths) > 1 else 0
    valid_set = {path.stem for path in image_paths[:valid_count]}

    for image_path in image_paths:
        label_path = source_labels / f"{image_path.stem}.txt"
        if image_path.stem in valid_set:
            target_images = output_root / "valid" / "images"
            target_labels = output_root / "valid" / "labels"
        else:
            target_images = output_root / "train" / "images"
            target_labels = output_root / "train" / "labels"

        shutil.copy2(image_path, target_images / image_path.name)
        if label_path.exists():
            shutil.copy2(label_path, target_labels / label_path.name)
        else:
            (target_labels / f"{image_path.stem}.txt").write_text("", encoding="utf-8")

    source_yaml = source_root / "data.yaml"
    if source_yaml.exists():
        shutil.copy2(source_yaml, output_root / "data.yaml")

    return output_root


def ensure_data_yaml(dataset_root: Path, names: list[str], generate: bool) -> Path:
    data_yaml = dataset_root / "data.yaml"
    if data_yaml.exists():
        return data_yaml
    if not generate:
        raise FileNotFoundError(f"Missing data.yaml at {data_yaml}")
    return write_data_yaml(dataset_root, names)


def publish_best_weight(best_weight: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weight, output_path)
    return output_path


def build_train_kwargs(args: argparse.Namespace) -> dict:
    train_kwargs = {
        "task": args.task,
        "data": str(args.data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "exist_ok": args.exist_ok,
        "pretrained": True,
        "verbose": True,
        # Torch/Ultralytics checkpoint saving can fail on some Windows+CPU setups.
        # Disable periodic checkpoint writes and publish a final model after training.
        "save": False,
    }

    if args.augment:
        train_kwargs.update({
            "augment": True,
            "degrees": args.degrees,
            "translate": args.translate,
            "scale": args.scale,
            "shear": args.shear,
            "fliplr": args.fliplr,
            "mosaic": args.mosaic,
            "close_mosaic": args.close_mosaic,
        })

    return train_kwargs


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 PPE helmet detector")
    parser.add_argument("--dataset", required=True, help="Path to the dataset root")
    parser.add_argument("--model", default="yolov8s.pt", help="Base YOLO model to fine-tune")
    parser.add_argument("--task", default="segment", choices=["detect", "segment"], help="Ultralytics task type")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 lets Ultralytics auto-select)")
    parser.add_argument("--device", default=None, help="Device to train on, e.g. cpu, 0, 0,1")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--project", default="runs/detect", help="Training output directory")
    parser.add_argument("--name", default="train", help="Run name")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reuse of the run directory")
    parser.add_argument("--generate-data-yaml", action="store_true", help="Generate data.yaml if missing")
    parser.add_argument("--names", nargs=2, default=["helmet", "no_helmet"], help="YOLO class names")
    parser.add_argument("--publish-best", action="store_true", help="Copy best.pt to models/ppe_helmet.pt after training")
    parser.add_argument("--prepare-split", action="store_true", help="Create a train/valid split from a train-only export")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio when preparing a dataset")
    parser.add_argument("--augment", action="store_true", help="Enable stronger YOLO augmentation during training")
    parser.add_argument("--degrees", type=float, default=10.0, help="Random rotation degrees for augmentation")
    parser.add_argument("--translate", type=float, default=0.10, help="Random translation fraction for augmentation")
    parser.add_argument("--scale", type=float, default=0.50, help="Random scale fraction for augmentation")
    parser.add_argument("--shear", type=float, default=2.0, help="Random shear degrees for augmentation")
    parser.add_argument("--fliplr", type=float, default=0.50, help="Horizontal flip probability")
    parser.add_argument("--mosaic", type=float, default=1.00, help="Mosaic augmentation probability")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic this many epochs before the end")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent

    dataset_root = Path(args.dataset).expanduser().resolve()
    if not dataset_root.exists():
        print(f"Dataset root not found: {dataset_root}")
        return 1

    if args.prepare_split and not is_dataset_ready(dataset_root):
        prepared_root = dataset_root.parent / f"{dataset_root.name}_prepared"
        print(f"Preparing train/valid split at: {prepared_root}")
        dataset_root = prepare_dataset_split(dataset_root, prepared_root, val_ratio=args.val_ratio)

    train_summary = validate_split(dataset_root / "train")
    valid_summary = validate_split(dataset_root / "valid")
    if train_summary["issues"] or valid_summary["issues"]:
        print("Dataset structure issues detected:")
        for issue in train_summary["issues"] + valid_summary["issues"]:
            print(f"  - {issue}")
        return 1

    data_yaml = ensure_data_yaml(dataset_root, args.names, args.generate_data_yaml)
    args.data_yaml = data_yaml
    print(f"Using data.yaml: {data_yaml}")
    print(f"Training base model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Epochs: {args.epochs}, imgsz: {args.imgsz}, batch: {args.batch}, device: {args.device or 'auto'}")
    print("Checkpoint saving during training is disabled for stability on this environment.")
    if args.augment:
        print(
            "Augmentation enabled: "
            f"degrees={args.degrees}, translate={args.translate}, scale={args.scale}, "
            f"shear={args.shear}, fliplr={args.fliplr}, mosaic={args.mosaic}, close_mosaic={args.close_mosaic}"
        )

    model = YOLO(args.model)
    try:
        metrics = model.train(**build_train_kwargs(args))
    except Exception as exc:
        print(f"Training failed: {exc}")
        traceback.print_exc()
        return 1

    trainer = getattr(model, "trainer", None)
    if trainer is None:
        print("Training completed, but trainer state was not available.")
        return 1

    save_dir = Path(trainer.save_dir)
    best_weight = Path(trainer.best) if getattr(trainer, "best", None) else save_dir / "weights" / "best.pt"
    print(f"Training completed. Save dir: {save_dir}")
    print(f"Best weight: {best_weight}")
    if metrics is not None:
        print(f"Metrics: {metrics}")

    if args.publish_best:
        target = repo_root / "models" / "ppe_helmet.pt"
        if best_weight.exists():
            published = publish_best_weight(best_weight, target)
            print(f"Published best weight to: {published}")
        else:
            model.save(str(target))
            print(f"Published trained model to: {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())