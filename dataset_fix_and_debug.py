import os
import cv2
import random
import numpy as np


BLUR_THRESHOLD = 100.0
HEAD_TOP_RATIO = 0.40
HEAD_CENTER_WIDTH_RATIO = 0.60
OUTPUT_SIZE = 224
MIN_CROP_SIZE = 50
EMPTY_STD_THRESHOLD = 8.0
EMPTY_NONZERO_RATIO = 0.05


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {0: "helmet", 1: "no_helmet"}


def safe_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def list_images(images_dir):
    if not os.path.isdir(images_dir):
        return []
    files = []
    for name in os.listdir(images_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTS:
            files.append(name)
    files.sort()
    return files


def image_to_label_path(image_path, labels_dir):
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(labels_dir, base + ".txt")


def parse_label_file(label_path):
    rows = []
    if not os.path.isfile(label_path):
        return rows

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return rows

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            if w <= 0 or h <= 0:
                continue
            rows.append((cls_id, x, y, w, h))
        except Exception:
            continue
    return rows


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    bw = w * img_w
    bh = h * img_h
    cx = xc * img_w
    cy = yc * img_h

    x1 = int(round(cx - bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    x2 = int(round(cx + bw / 2.0))
    y2 = int(round(cy + bh / 2.0))

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w, x2))
    y2 = max(0, min(img_h, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def compute_head_crop(x1, y1, x2, y2, img_w, img_h):
    bw = x2 - x1
    bh = y2 - y1

    head_h = int(round(bh * HEAD_TOP_RATIO))
    head_h = max(1, head_h)

    head_w = int(round(bw * HEAD_CENTER_WIDTH_RATIO))
    head_w = max(1, head_w)

    center_x = x1 + bw / 2.0
    left = int(round(center_x - head_w / 2.0))
    right = int(round(center_x + head_w / 2.0))

    top = y1
    bottom = y1 + head_h

    left = max(0, left)
    top = max(0, top)
    right = min(img_w, right)
    bottom = min(img_h, bottom)

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def is_crop_low_quality(crop):
    h, w = crop.shape[:2]
    if w < MIN_CROP_SIZE or h < MIN_CROP_SIZE:
        return True

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    std_val = float(np.std(gray))
    nonzero_ratio = float(np.count_nonzero(gray > 10)) / float(gray.size)

    if std_val < EMPTY_STD_THRESHOLD:
        return True
    if nonzero_ratio < EMPTY_NONZERO_RATIO:
        return True
    return False


def remove_blurry_images(dataset_root):
    removed_count = 0
    total_images = 0

    for split in ["train", "valid"]:
        images_dir = os.path.join(dataset_root, split, "images")
        labels_dir = os.path.join(dataset_root, split, "labels")

        image_names = list_images(images_dir)
        total_images += len(image_names)

        for image_name in image_names:
            image_path = os.path.join(images_dir, image_name)
            img = cv2.imread(image_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()

            if variance < BLUR_THRESHOLD:
                try:
                    os.remove(image_path)
                except Exception:
                    pass

                label_path = image_to_label_path(image_path, labels_dir)
                if os.path.isfile(label_path):
                    try:
                        os.remove(label_path)
                    except Exception:
                        pass

                removed_count += 1

    return total_images, removed_count


def prepare_output_structure(output_root):
    for split in ["train", "valid"]:
        safe_makedirs(os.path.join(output_root, split, "images"))
        safe_makedirs(os.path.join(output_root, split, "labels"))


def process_split(dataset_root, output_root, split, stats, debug_records):
    src_images_dir = os.path.join(dataset_root, split, "images")
    src_labels_dir = os.path.join(dataset_root, split, "labels")

    dst_images_dir = os.path.join(output_root, split, "images")
    dst_labels_dir = os.path.join(output_root, split, "labels")

    image_names = list_images(src_images_dir)

    for image_name in image_names:
        image_path = os.path.join(src_images_dir, image_name)
        label_path = image_to_label_path(image_path, src_labels_dir)

        img = cv2.imread(image_path)
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        labels = parse_label_file(label_path)
        if not labels:
            continue

        base = os.path.splitext(image_name)[0]
        obj_idx = 0

        for cls_id, xc, yc, bw, bh in labels:
            bbox = yolo_to_xyxy(xc, yc, bw, bh, img_w, img_h)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            head_box = compute_head_crop(x1, y1, x2, y2, img_w, img_h)
            if head_box is None:
                stats["removed_low_quality_crops"] += 1
                continue

            hx1, hy1, hx2, hy2 = head_box
            crop = img[hy1:hy2, hx1:hx2]
            if crop.size == 0:
                stats["removed_low_quality_crops"] += 1
                continue

            if is_crop_low_quality(crop):
                stats["removed_low_quality_crops"] += 1
                continue

            crop_resized = cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)

            out_name = f"{base}_obj{obj_idx}.jpg"
            out_img_path = os.path.join(dst_images_dir, out_name)
            out_lbl_path = os.path.join(dst_labels_dir, f"{base}_obj{obj_idx}.txt")

            ok = cv2.imwrite(out_img_path, crop_resized)
            if not ok:
                continue

            # The crop is the focused object region, so bbox spans the crop.
            yolo_line = f"{cls_id} 0.500000 0.500000 1.000000 1.000000\n"
            try:
                with open(out_lbl_path, "w", encoding="utf-8") as f:
                    f.write(yolo_line)
            except Exception:
                continue

            stats["final_cropped_images"] += 1
            if cls_id == 0:
                stats["helmet_count"] += 1
            elif cls_id == 1:
                stats["no_helmet_count"] += 1

            debug_records.append((out_img_path, cls_id, (int(OUTPUT_SIZE * 0.0), int(OUTPUT_SIZE * 0.0), int(OUTPUT_SIZE * 1.0), int(OUTPUT_SIZE * 1.0))))
            obj_idx += 1


def print_stats(stats):
    print("=" * 70)
    print("DATASET STATS")
    print("=" * 70)
    print(f"Total images (before blur filter): {stats['total_images']}")
    print(f"Removed blurry images:            {stats['removed_blurry_images']}")
    print(f"Removed low-quality crops:        {stats['removed_low_quality_crops']}")
    print(f"Final cropped images:             {stats['final_cropped_images']}")
    print(f"Helmet count:                     {stats['helmet_count']}")
    print(f"No-helmet count:                  {stats['no_helmet_count']}")
    print("=" * 70)


def visualize_debug(debug_records, max_samples=20):
    if not debug_records:
        print("[DEBUG] No cropped images available for visualization.")
        return

    samples = random.sample(debug_records, min(max_samples, len(debug_records)))

    print("[DEBUG] Press any key for next image, or 'q' to quit.")
    for img_path, cls_id, bbox in samples:
        img = cv2.imread(img_path)
        if img is None:
            continue

        x1, y1, x2, y2 = bbox
        label = CLASS_NAMES.get(cls_id, str(cls_id))

        cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
        cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        try:
            cv2.imshow("dataset_fix_and_debug", img)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
        except cv2.error:
            print("[DEBUG] OpenCV window display failed on this environment.")
            break

    cv2.destroyAllWindows()


def main():
    dataset_root = os.path.join(os.getcwd(), "dataset")
    output_root = os.path.join(os.getcwd(), "dataset_head")

    train_images = os.path.join(dataset_root, "train", "images")
    train_labels = os.path.join(dataset_root, "train", "labels")
    valid_images = os.path.join(dataset_root, "valid", "images")
    valid_labels = os.path.join(dataset_root, "valid", "labels")

    required_dirs = [train_images, train_labels, valid_images, valid_labels]
    for d in required_dirs:
        if not os.path.isdir(d):
            print(f"[ERROR] Missing required folder: {d}")
            return

    stats = {
        "total_images": 0,
        "removed_blurry_images": 0,
        "removed_low_quality_crops": 0,
        "final_cropped_images": 0,
        "helmet_count": 0,
        "no_helmet_count": 0,
    }

    print("[INFO] Running blur filter...")
    total_images, removed_blurry = remove_blurry_images(dataset_root)
    stats["total_images"] = total_images
    stats["removed_blurry_images"] = removed_blurry

    print("[INFO] Preparing output dataset structure...")
    prepare_output_structure(output_root)

    print("[INFO] Running head-focused cropping...")
    debug_records = []
    process_split(dataset_root, output_root, "train", stats, debug_records)
    process_split(dataset_root, output_root, "valid", stats, debug_records)

    print_stats(stats)

    print("[INFO] Starting visual debug mode (20 random samples)...")
    visualize_debug(debug_records, max_samples=20)

    print(f"[INFO] Done. Output dataset: {output_root}")


if __name__ == "__main__":
    main()
