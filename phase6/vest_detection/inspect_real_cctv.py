"""
inspect_real_cctv.py
--------------------
Audits real CCTV image, saves visualisations with bounding boxes,
and tests torso inference across three distinct workers.
"""

import sys, os, cv2, json
sys.path.insert(0, os.path.abspath('src'))
from vest_detector import VestDetector

cctv_candidates = [
    r'E:\PROJECTS\safeops\archive\camera-quality-guardian\reports\annotated_construction_cctv_20261030_192942.jpg',
    r'E:\PROJECTS\safeops\safeops-git\reports\annotated_construction_cctv_20261030_192942.jpg',
    r'E:\PROJECTS\safeops\camera-quality-guardian\reports\annotated_construction_cctv_20261030_192942.jpg',
    os.path.abspath('artifacts/evaluation/real_cctv_vest_visualisation.jpg'),
]
cctv_path = None
for p in cctv_candidates:
    if os.path.exists(p):
        cctv_path = p
        break

if not cctv_path:
    print("Error: CCTV image not found.")
    sys.exit(1)

img = cv2.imread(cctv_path)
h, w = img.shape[:2]

detector = VestDetector('models/yolov8n_vest_best.pt', device='cuda')

# 1. Full-image inference
result = detector.detect(img)
detections = result['detections']
print(f"Total detections at threshold >= 0.35: {len(detections)}")

# 2. Draw visualisation
vis_img = img.copy()
for i, det in enumerate(detections):
    box = det['bbox']
    cls_name = det['class_name']
    conf = det['confidence']
    x1, y1, x2, y2 = [int(round(c)) for c in box]

    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"#{i+1} {cls_name} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(vis_img, (x1, max(0, y1 - 20)), (x1 + tw, y1), (0, 255, 0), -1)
    cv2.putText(vis_img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

out_dir = os.path.abspath('artifacts/evaluation')
os.makedirs(out_dir, exist_ok=True)
vis_path = os.path.join(out_dir, 'real_cctv_vest_visualisation.jpg')
cv2.imwrite(vis_path, vis_img)
print(f"Visualisation saved to: {vis_path}")

for i, det in enumerate(detections):
    print(f"Prediction #{i+1}: class={det['class_name']}, conf={det['confidence']:.4f}, bbox={det['bbox']}")

# 3. Test torso inference on three distinct workers:
# Let's locate workers in the CCTV image:
# Worker 1: Standing in foreground right [1357, 670, 1457, 820] (Wearing fluorescent vest)
torso_1 = [1357.0, 670.0, 1457.0, 820.0]
assessment_1 = detector.assess_torso(img, torso_1)

# Worker 2: Standing in mid-ground center/left without fluorescent vest (e.g. worker at [445, 535, 510, 712] or [759, 551, 820, 702])
# Let's inspect torso at [445, 540, 510, 660]
torso_2 = [445.0, 540.0, 510.0, 660.0]
assessment_2 = detector.assess_torso(img, torso_2)

# Worker 3: Far distant / obscured worker in background at [146, 106, 186, 208]
torso_3 = [146.0, 110.0, 186.0, 180.0]
assessment_3 = detector.assess_torso(img, torso_3)

print("\n--- WORKER 1 (CONFIRMED WEARING VEST) ---")
print(json.dumps(assessment_1.to_dict(), indent=2))

print("\n--- WORKER 2 (VISIBLY WITHOUT FLUORESCENT VEST / DARK JACKET) ---")
print(json.dumps(assessment_2.to_dict(), indent=2))

print("\n--- WORKER 3 (DISTANT / OBSCURED IN BACKGROUND) ---")
print(json.dumps(assessment_3.to_dict(), indent=2))
