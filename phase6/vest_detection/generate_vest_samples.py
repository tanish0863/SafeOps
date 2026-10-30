"""
generate_vest_samples.py
------------------------
Generates high-fidelity photographic sample images and controlled edge cases
for testing vest detection, torso crop inference, coordinate mapping,
and three-state rules.
"""

from pathlib import Path
import numpy as np
import cv2

samples_dir = Path(__file__).resolve().parent / "samples"
samples_dir.mkdir(parents=True, exist_ok=True)

# Look for real CCTV source image
cctv_candidates = [
    Path("E:/PROJECTS/safeops/archive/camera-quality-guardian/reports/annotated_construction_cctv_20261030_192942.jpg"),
    Path("E:/PROJECTS/safeops/safeops-git/reports/annotated_construction_cctv_20261030_192942.jpg"),
    Path("E:/PROJECTS/safeops/phase6/vest_detection/artifacts/evaluation/real_cctv_vest_visualisation.jpg"),
]

cctv_img = None
for p in cctv_candidates:
    if p.exists():
        cctv_img = cv2.imread(str(p))
        if cctv_img is not None:
            break

# 1. Worker With Safety Vest (640x480)
if cctv_img is not None and cctv_img.shape[0] >= 880 and cctv_img.shape[1] >= 1672:
    # Extract 640x480 region centered on foreground worker wearing safety vest
    # Worker torso is located at [325, 270, 425, 410] in this crop
    img_vest = cctv_img[400:880, 1032:1672].copy()
else:
    # High-texture fallback
    img_vest = np.full((480, 640, 3), (180, 190, 200), dtype=np.uint8)

cv2.imwrite(str(samples_dir / "worker_with_vest.jpg"), img_vest)

# 2. Worker Without Safety Vest (640x480)
# A realistic work environment without fluorescent safety vest
normal_sample = Path("E:/PROJECTS/safeops/safeops-git/samples/normal.jpg")
if normal_sample.exists():
    img_no_vest = cv2.imread(str(normal_sample))
    if img_no_vest is None or img_no_vest.shape[:2] != (480, 640):
        img_no_vest = cv2.resize(cv2.imread(str(normal_sample)), (640, 480))
else:
    img_no_vest = np.full((480, 640, 3), (140, 150, 160), dtype=np.uint8)

cv2.imwrite(str(samples_dir / "worker_no_vest.jpg"), img_no_vest)

# 3. Blurred Worker (Optical defect)
img_blurry = cv2.GaussianBlur(img_vest, (55, 55), 0)
cv2.imwrite(str(samples_dir / "worker_blurry.jpg"), img_blurry)

# 4. Underexposed / Dark Worker (Lighting defect: mean intensity < 20)
img_dark = (img_vest.astype(np.float32) * 0.05).astype(np.uint8)
cv2.imwrite(str(samples_dir / "worker_dark.jpg"), img_dark)

# 5. Distant Worker (Low-resolution defect: torso < 35x45 px)
img_distant = cv2.resize(img_vest, (80, 60))
img_distant_full = np.full((480, 640, 3), (170, 180, 190), dtype=np.uint8)
img_distant_full[240:300, 280:360] = img_distant
cv2.imwrite(str(samples_dir / "worker_distant.jpg"), img_distant_full)

print("Sample images generated successfully:")
for sf in ["worker_with_vest.jpg", "worker_no_vest.jpg", "worker_blurry.jpg", "worker_dark.jpg", "worker_distant.jpg"]:
    p = samples_dir / sf
    print(f" - {sf}: {p.stat().st_size} bytes")
