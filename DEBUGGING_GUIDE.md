# DEBUGGING GUIDE - Identifying False Positives & False Negatives

This guide helps you trace back from evaluation errors to the source module and fix them.

## Quick Diagnosis Flowchart

```
ERROR DETECTED (from evaluation_metrics.py)
    ↓
1. Identify error type: Person? Helmet? Zone?
2. Identify error class: FP or FN?
3. Check frame video reference
4. Trace through module stack
5. Fix specific module or data
```

---

## Error Type 1: False Negatives (FN) in Person Detection

**What it means:** System missed a person that was actually in the frame.

**Evaluation output:**
```
[PERSON_DETECTION]
  False Negatives (FN): 3
  Recall: 75.00%
```

**Common causes:**
1. YOLO confidence threshold too high (strict, misses real people)
2. Person partially occluded or at image boundary
3. Video quality issue (blur, low light)
4. Model trained on different pose/angle

### Debug Steps

**Step 1: Verify ground truth is correct**
```bash
# Review the problematic frame
# Ask: Is there definitely a person in frame?
# Check: frame_XXX.jpg in debug_snapshots/
```

**Step 2: Check YOLO detection output**
Create a minimal test:
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
# Load frame from ground truth
results = model("path/to/frame_XXX.jpg", conf=0.45, verbose=True)
print(f"Detections: {len(results[0].boxes)}")
for box in results[0].boxes:
    print(f"  Class={box.cls}, Conf={box.conf:.3f}")
```

**Step 3: Check if it's a confidence threshold issue**
```python
# Test with lower threshold
for conf_threshold in [0.45, 0.35, 0.25, 0.15]:
    results = model("frame.jpg", conf=conf_threshold)
    print(f"Threshold {conf_threshold}: {len(results[0].boxes)} detections")
```

If lowering threshold finds the person:
- **Fix:** Update `config.py` MIN_CONFIDENCE (but watch for FP increase)
- **Alternative:** Add preprocessing (histogram equalization for dark frames)

**Step 4: Check if it's a tracking issue**
```python
# Verify detection exists but tracking lost it
# In main.py, add debug prints:
print(f"Raw detections: {len(detections)}")  # before tracker
print(f"Tracked persons: {len(self.tracked_persons)}")  # after tracker
```

If raw detections exist but tracking lost them:
- **Fix:** Check DeepSORT parameters in `tracking.py`:
  - Increase MAX_AGE (people disappear too fast)
  - Decrease DETECTION_THRESHOLD (too strict matching)
  - Verify ReID embedder works with current camera

---

## Error Type 2: False Positives (FP) in Person Detection

**What it means:** System detected a person when there wasn't one (ghost detection).

**Evaluation output:**
```
[PERSON_DETECTION]
  False Positives (FP): 5
  Precision: 80.00%
```

**Common causes:**
1. YOLO hallucinating (model made wrong prediction)
2. Background shadows/textures mistaken for people
3. Video noise (compression artifacts)
4. Multiple detections on same person merged incorrectly

### Debug Steps

**Step 1: Save frame at error location**
```python
# Modify main.py to save frames with FP errors
if evaluation_says_FP:
    cv2.imwrite(f"fp_frame_{frame_id}.jpg", frame)
    visualization.draw_detections(frame)  # draw what system saw
    cv2.imwrite(f"fp_frame_{frame_id}_annotated.jpg", frame)
```

**Step 2: Examine YOLO output directly**
```python
# On the saved frame, run: 
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model("fp_frame_123.jpg", conf=0.45)

for box in results[0].boxes:
    class_id = int(box.cls[0])
    class_name = ["person", ...][class_id]
    confidence = float(box.conf[0])
    print(f"Detected: {class_name} ({confidence:.2f})")
    
    # Visualize
    import cv2
    img = cv2.imread("fp_frame_123.jpg")
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite("fp_frame_123_boxes.jpg", img)
```

View `fp_frame_123_boxes.jpg` to see what YOLO found.

**Step 3: Determine root cause**
- **Is it a real person but ground truth wrong?** → Fix ground_truth.csv
- **Is it shadow/texture?** → Lower MIN_CONFIDENCE gradually (trade off for FN increase)
- **Is confidence borderline (e.g., 0.45)?** → Increase MIN_CONFIDENCE to 0.50+
- **Is it video noise?** → Add preprocessing (Gaussian blur to denoise)

**Step 4: Implement fix**

Option A: Strict confidence
```python
# In config.py
MIN_CONFIDENCE = 0.55  # was 0.45 (trade off: may miss some real people)
```

Option B: Preprocessing
```python
# In detection.py, warm_up method
import cv2
frame = cv2.GaussianBlur(frame, (5, 5), 0)  # denoise
```

Option C: Better camera positioning
- Reduce reflections and shadows in camera view
- Improve lighting in monitored area

---

## Error Type 3: False Negatives in Helmet Detection

**What it means:** Person without helmet was detected, but system said helmet was "Yes" (missed violation).

**Evaluation output:**
```
[HELMET_DETECTION]
  False Negatives (FN): 4
  Recall: 80.00%
```

**Common causes:**
1. PPE model confidence too high (strict)
2. Helmet partially visible/occluded (model can't see clearly)
3. Model trained only on specific helmet types
4. Person's head partially invisible in frame

### Debug Steps

**Step 1: Get the frame and crop around person's head**
```python
# From tracking data, save person crop at error frame:
tracked_person = tracked_persons[0]  # frame 34
head_crop = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
cv2.imwrite("fn_helmet_frame34_person_crop.jpg", head_crop)
```

**Step 2: Test PPE model on the crop**
```python
from ultralytics import YOLO
ppe_model = YOLO("models/ppe_helmet.pt")

import cv2
crop = cv2.imread("fn_helmet_frame34_person_crop.jpg")
results = ppe_model(crop, conf=0.45)

for box in results:
    class_id = int(box.cls[0])
    class_names = ["helmet", "no_helmet"]
    print(f"PPE detected: {class_names[class_id]} (conf={box.conf[0]:.3f})")
```

**Step 3: Identify issue**

**IF model says "no_helmet" correctly:**
→ Issue is in confidence thresholding or post-processing
→ Check `ppe_detection.py` dominant_detection logic

**IF model says "helmet" (wrong):**
→ Model is not accurate on this helmet type
→ Save crop as training example for model fine-tuning

**IF model has very low confidence on no_helmet (e.g., 0.20):**
→ Model is uncertain (possible occlusion or bad angle)
→ Lower confidence threshold in `config.py` PPE_CONFIDENCE

**Step 4: Implement fix**

Option A: Lower confidence threshold
```python
# In config.py
PPE_CONFIDENCE_THRESHOLD = 0.35  # was 0.45 (stricter → catches more violations)
```

Option B: Improve post-processing
```python
# In ppe_detection.py, dominant_detection()
# Ensure "no_helmet" has priority when confidence margin is small
if no_helmet_conf > helmet_conf + 0.05:  # was 0.0 (more strict)
    return "no_helmet"
```

Option C: Collect model training data
- Save 50+ frames with FN helmet issues
- Create bbox labels with helmet/no_helmet
- Fine-tune model on your specific environment

---

## Error Type 4: False Positives in Helmet Detection

**What it means:** Person with helmet was detected without helmet (false alarm).

**Evaluation output:**
```
[HELMET_DETECTION]
  False Positives (FP): 2
  Precision: 95.00%
```

**Common causes:**
1. PPE model hallucination (model made wrong call)
2. Model confidence on "no_helmet" is low (borderline)
3. Hat/cap mistaken for helmet
4. Reflection or shadow on head looks like absence

### Debug Steps

**Step 1: Find the frame and save crop**
```python
# Frame 45: person detected, system said "no_helmet" (FP)
head_crop = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
cv2.imwrite("fp_helmet_frame45_crop.jpg", head_crop)
```

**Step 2: Examine both detector and PPE model**
```python
# First, verify person detection is correct
person_bbox = results[0].boxes[0]
print(f"Person detected with confidence {person_bbox.conf:.3f}")

# Then, test PPE on the crop
from ultralytics import YOLO
ppe_model = YOLO("models/ppe_helmet.pt")
crop = cv2.imread("fp_helmet_frame45_crop.jpg")
ppe_results = ppe_model(crop, conf=0.2)  # low conf to see all detections

for box in ppe_results:
    class_names = ["helmet", "no_helmet"]
    print(f"PPE model says: {class_names[int(box.cls)]} (conf={box.conf:.3f})")
```

**Step 3: Check confidence level**
- **If conf > 0.70:** Model is confident → probably right, check ground truth
- **If conf 0.45-0.70:** Borderline → likely visual ambiguity (hat vs helmet)
- **If conf < 0.45:** Model uncertain → should be filtered out

**Step 4: Fix strategy**

Option A: Increase PPE threshold (reduce FP)
```python
# In config.py
PPE_CONFIDENCE_THRESHOLD = 0.55  # was 0.45 (trade-off: may miss some violations)
```

Option B: Use ensemble voting
```python
# In ppe_detection.py:
# Sample crop 5 times with slight augmentations
# Take majority vote on helmet/no_helmet
votes = []
for i in range(5):
    augmented_crop = random_augment(crop)
    result = model.predict(augmented_crop)
    votes.append(result.prediction)
    
final_prediction = mode(votes)  # majority vote
```

Option C: Manual review specific frames
- Save all PPE FP frames
- Review to see if ground truth is wrong
- Update ground_truth.csv if needed

---

## Error Type 5: False Negatives in Zone Detection

**What it means:** Person was in restricted zone, system said "No" (missed violation).

**Evaluation output:**
```
[ZONE_DETECTION]
  False Negatives (FN): 3
  Recall: 70.00%
```

**Common causes:**
1. Zone polygon definition is incorrect/too small
2. Person's foot-point is on zone boundary (floating point edge case)
3. Person fully outside zone but close (detection inaccuracy?)
4. Multiple people—some in zone, some not (aggregation logic wrong)

### Debug Steps

**Step 1: Verify frame and person position**
```python
# Load frame and draw what system detected
frame = cv2.imread("frame_78.jpg")
tracked_person = get_from_tracking_db(frame_id=78)
bbox = tracked_person.bbox
foot_point = (bbox.center_x, bbox.bottom_y)

# Draw on frame
cv2.circle(frame, foot_point, 5, (0, 255, 0), -1)  # green=foot point
cv2.rectangle(frame, bbox...)  # blue=bbox
cv2.imwrite("debug_zone_fn_frame78.jpg", frame)
```

**Step 2: Check zone polygon definition**
```python
# From zone_logic.py:
RESTRICTED_ZONES = [
    {"name": "Zone A", "points": [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]},
    ...
]

# Visualize zones on reference frame
import cv2
frame = cv2.imread("zone_reference.jpg")
for zone in RESTRICTED_ZONES:
    points = np.array(zone["points"])
    cv2.polylines(frame, [points], True, (0, 255, 0), 2)  # green
cv2.imwrite("debug_zones_overlay.jpg", frame)
```

**Step 3: Test cv2.pointPolygonTest on the error frame**
```python
import cv2
from zone_logic import RESTRICTED_ZONES

frame_shape = (720, 1280)  # height, width of your video
zone = RESTRICTED_ZONES[0]
zone_points = np.array(zone["points"], dtype=np.int32)

foot_point = (640, 500)  # example from frame 78
result = cv2.pointPolygonTest(zone_points, foot_point, False)

print(f"Point {foot_point} relative to zone: {result}")
# result > 0: inside
# result < 0: outside
# result = 0: on edge
```

**Step 4: Identify issue**

**IF result > 0 but system said outside:**
→ Bug in zone_logic.py
→ Check aggregation logic (require_full_inside Mode?)

**IF result < 0 but ground truth says inside:**
→ Zone polygon definition is wrong
→ Redraw zone polygon based on camera calibration

**IF result = 0 (on edge):**
→ Floating point precision issue
→ Add small margin (5px) to polygon

**Step 5: Fix**

Option A: Redefine zone polygon
```python
# In config.py (recalibrate on camera):
RESTRICTED_ZONES = [
    {
        "name": "Machinery Zone",
        "points": [(100, 100), (200, 100), (200, 300), (100, 300)],  # update coordinates
    }
]
```

Option B: Use centroid instead of foot-point
```python
# In zone_logic.py ZoneChecker.check_violation():
check_point = (bbox.center_x, bbox.center_y)  # was (center_x, bottom_y)
```

Option C: Add margin for robustness
```python
# In zone_logic.py:
def is_point_in_zone(point, zone_polygon, margin=5):
    # Expand polygon by margin pixels
    expanded_zone = cv2.dilate(
        create_mask(zone_polygon),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*margin, 2*margin))
    )
    return is_inside(point, expanded_zone)
```

---

## Error Type 6: False Positives in Zone Detection

**What it means:** Person was NOT in zone, but system said "Yes" (false alarm).

**Evaluation output:**
```
[ZONE_DETECTION]
  False Positives (FP): 2
  Precision: 90.00%
```

**Most common cause:** Zone polygon definition is too large or incorrectly positioned.

### Debug Steps

**Step 1: Visualize what the system thought**
```python
# Frame where FP occurred
frame = cv2.imread("frame_125.jpg")
tracked_person = get_from_tracking_db(frame_id=125)

# Draw system's decision
bbox = tracked_person.bbox
color = (0, 0, 255) if tracked_person.in_zone else (0, 255, 0)  # red if in zone
cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)
cv2.imwrite("debug_zone_fp_frame125_bbox.jpg", frame)
```

**Step 2: Overlay zone polygon on frame**
```python
import cv2
import numpy as np

frame = cv2.imread("frame_125.jpg")
zone_polygon = np.array(RESTRICTED_ZONES[0]["points"], dtype=np.int32)
cv2.polylines(frame, [zone_polygon], True, (0, 255, 0), 3)  # green polygon

# Also draw foot-point
foot_point = (640, 500)
cv2.circle(frame, foot_point, 10, (0, 0, 255), -1)  # red circle

cv2.imwrite("debug_zone_fp_frame125_overlay.jpg", frame)
```

**Step 3: Check if polygon is wrong**
- Visual inspection: Is the green polygon in the right location?
- Does it cover the correct restricted area?
- Is it too large?

**Step 4: Fix**

If polygon is incorrectly positioned or too large:

```python
# In config.py, redefine zone based on camera view:
RESTRICTED_ZONES = [
    {
        "name": "Machinery Zone A",
        "points": [(50, 150), (150, 150), (150, 300), (50, 300)],  # corrected
    }
]
```

Quick calibration steps:
1. Open frame from video (EVAL_EXAMPLES.md has reference process)
2. Measure pixel coordinates of zone corners using image viewer
3. Update polygon coordinates in config.py
4. Re-run evaluation

---

## Complete Debugging Workflow Example

**Scenario:** Evaluation shows 3 FN (false negatives) in helmet detection on frame 34.

### Walkthrough

```
Step 1: Identify error
-----------
[HELMET_DETECTION]
  False Negatives (FN): 3

Step 2: Find problematic frame
-----------
[ERROR ANALYSIS]
Frame 34 | Type: helmet | FALSE_NEGATIVE | GT=Yes Pred=No

Step 3: Extract person crop at frame 34
-----------
python -c "
import cv2
frame = cv2.imread('frame_34.jpg')
bbox = (x1, y1, x2, y2)  # from tracking DB
crop = frame[y1:y2, x1:x2]
cv2.imwrite('helmet_fn_frame34.jpg', crop)
"

Step 4: Test PPE model on crop
-----------
python -c "
from ultralytics import YOLO
ppe = YOLO('models/ppe_helmet.pt')
results = ppe('helmet_fn_frame34.jpg', conf=0.2)
for box in results:
    print(f'Class: {int(box.cls)}, Conf: {box.conf:.3f}')
"

Output: Class: 1 (no_helmet), Conf: 0.38

Step 5: Check why system missed this
-----------
If conf=0.38 < conf_threshold=0.45:
→ No detection recorded
→ System defaulted to "helmet" (safe assumption)

Step 6: Implement fix
-----------
Lower threshold in config.py:
PPE_CONFIDENCE_THRESHOLD = 0.35

Step 7: Re-evaluate
-----------
python evaluation_metrics.py
→ Helmet detection FN reduced from 3 to 1
→ Check if FP increased significantly
```

---

## Prevention Strategies

### 1. Test in Advance
```bash
# Before deployment, test on 100+ frames
python evaluation_metrics.py

# Set minimum accuracy targets:
# - Person detection: >95%
# - Helmet detection: >90%
# - Zone detection: >90%
```

### 2. Log System Decisions
```python
# In main.py, add detailed logging:
print(f"Frame {i}: Person={person_detected} (conf={conf:.3f}), Helmet={'Yes' if has_helmet else 'No'} (ppe_conf={ppe_conf:.3f})")
```

### 3. Periodic Manual Review
- Every 500 frames, manually spot-check 10 frames
- Compare system output to reality
- Log differences for trending

### 4. Monitor Over Time
```bash
# Weekly evaluation:
# - Compare new predictions.csv to old
# - Track accuracy trends
# - Alert if accuracy drops >5%
```

---

## Quick Reference: Common Fixes

| Problem | Quick Fix |
|---------|-----------|
| FN in person detection | Lower MIN_CONFIDENCE: 0.45 → 0.35 |
| FP in person detection | Raise MIN_CONFIDENCE: 0.45 → 0.55 |
| FN in helmet | Lower PPE_CONFIDENCE: 0.45 → 0.35 |
| FP in helmet | Raise PPE_CONFIDENCE: 0.45 → 0.55 |
| FN in zone | Redraw zone polygon (too small) |
| FP in zone | Redraw zone polygon (too large) |
| Many errors across all | Retrain models on your specific environment |

---

## Getting Help

If errors persist after these steps:

1. **Save debug frames:** Use `cv2.imwrite()` at error locations
2. **Create minimal reproduction:** Single frame that shows the error
3. **Test component in isolation:** 
   ```bash
   python -c "from detection import PersonDetector; ..."
   ```
4. **Check model files:** Verify `yolov8n.pt` and `ppe_helmet.pt` exist and load
5. **Validate input data:** Ensure camera/video file is readable

For model training questions, see `requirements.txt` for ultralytics version and documentation.
