# Dataset Improvement Workflow

This repo now includes a practical loop for improving PPE accuracy in the order requested:

1. Mine hard examples where the model returns UNKNOWN.
2. Build a head-focused training set.
3. Add confusion cases into the no_helmet class.
4. Audit label quality on random samples.
5. Retrain with augmentation.
6. Remove low-value images.

## 1. Mine UNKNOWN cases

Run the model on videos or images and save only UNKNOWN outputs:

```bash
python mine_unknown_cases.py --source ./videos --output ./hard_examples --save-crop
```

Useful outputs:
- `hard_examples/unknown/crops/`
- `hard_examples/unknown/frames/`
- `hard_examples/unknown/manifest.csv`

## 2. Create head-focused crops

Generate a derived dataset that keeps the top part of the image and rewrites labels into YOLO box format:

```bash
python prepare_head_focus_dataset.py --source ./datasets/with-or-without-helmet.yolov8 --output ./datasets/helmet_head_focus --head-fraction 0.40 --keep-class-ids 1 2 --force
```

Train the result with detection mode:

```bash
python train_ppe_model.py --dataset ./datasets/helmet_head_focus --task detect --augment --publish-best
```

## 3. Review label quality

Render 20 random annotated images so you can inspect box/polygon quality quickly:

```bash
python audit_dataset_labels.py --dataset ./datasets/with-or-without-helmet.yolov8 --count 20 --output ./label_review
```

Check these problems:
- loose boxes
- partial heads
- missing helmets
- class mismatch
- broken polygons

## 4. Retrain with augmentation

Use stronger augmentation when training on the cleaned dataset:

```bash
python train_ppe_model.py --dataset ./datasets/with-or-without-helmet_prepared.yolov8 --augment --publish-best
```

Important flags:
- `--augment`
- `--degrees`
- `--translate`
- `--scale`
- `--shear`
- `--fliplr`
- `--mosaic`
- `--close-mosaic`

## 5. Remove useless data

Delete or exclude samples where:
- the head is not visible
- the image is too blurry
- the subject is too small
- the label is obviously wrong

## 6. Re-evaluate

Run the balanced sample evaluation after each retrain:

```bash
python evaluate_sample_images.py --samples ./samples
```

## Notes

- UNKNOWN is intentional. It means the model did not make a confident PPE decision.
- For production, UNKNOWN should not be silently converted into helmet or safe.
- The head crop script converts segmentation rows into bounding boxes so the output is best used with `--task detect`.
