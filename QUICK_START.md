# EVALUATION QUICK START

**Complete workflow to measure vision system accuracy in 5 minutes.**

---

## Phase 1: Prepare Ground Truth (Manual, ~15 min)

1. **Choose video segment:** 100-500 frames (3-15 seconds at 30 FPS)
2. **Export frames:**
   ```bash
   cd "d:\Projects\Hackive (angadi)\vision"
   python -c "
   import cv2
   cap = cv2.VideoCapture(0)  # your video source
   for i in range(100):
       ret, frame = cap.read()
       if ret:
           cv2.imwrite(f'eval_frame_{i}.jpg', frame)
   cap.release()
   "
   ```
3. **Manually label each frame:** Use `GROUND_TRUTH_TEMPLATE.md`
   - Open each `eval_frame_*.jpg` in image viewer
   - For each: note person present (Yes/No), helmet (Yes/No/N/A), zone (Yes/No/N/A)
   - Fill in [ground_truth.csv](ground_truth.csv)

**Example ground_truth.csv:**
```csv
frame_id,timestamp_seconds,person_present,helmet_visible,in_restricted_zone
0,0.0,No,N/A,N/A
1,0.03,No,N/A,N/A
2,0.06,Yes,Yes,No
3,0.1,Yes,No,Yes
```

---

## Phase 2: Collect System Predictions (~30 sec)

Run your vision system and capture output:

```bash
cd "d:\Projects\Hackive (angadi)\vision"
py -3 main.py 2>&1 | tee predictions.log
```

Or, if you need to add logging to main.py first:

```python
# In main.py SafetyPipeline, add to process_frame():
person_str = "Yes" if self.tracked_persons else "No"
helmet_str = "N/A"
zone_str = "N/A"

if self.tracked_persons:
    helmet_violations = sum(1 for p in self.tracked_persons if p.has_no_helmet)
    helmet_str = "No" if helmet_violations > 0 else "Yes"
    zone_violations = sum(1 for p in self.tracked_persons if p.in_restricted_zone)
    zone_str = "Yes" if zone_violations > 0 else "No"

print(f"PREDICTION: frame={self.frame_count} person={person_str} helmet={helmet_str} zone={zone_str}")
```

---

## Phase 3: Convert Log to CSV (~1 min)

Extract predictions from console output:

```bash
cd "d:\Projects\Hackive (angadi)\vision"
py -3 -c "
import re

with open('predictions.log') as f, open('predictions.csv', 'w') as out:
    out.write('frame_id,timestamp_seconds,person_present,helmet_visible,in_restricted_zone\n')
    for line in f:
        match = re.search(r'frame=(\d+) person=(\w+) helmet=(\w+) zone=(\w+)', line)
        if match:
            frame_id = int(match.group(1))
            timestamp = frame_id / 30.0
            out.write(f'{frame_id},{timestamp:.2f},{match.group(2)},{match.group(3)},{match.group(4)}\n')
print('Created predictions.csv')
"
```

---

## Phase 4: Evaluate (~5 sec)

```bash
cd "d:\Projects\Hackive (angadi)\vision"
py -3 evaluation_metrics.py
```

**Output example:**
```
[PERSON_DETECTION]
  Accuracy: 94.30%, Precision: 94.74%, Recall: 96.67%

[HELMET_DETECTION]
  Accuracy: 91.80%, Precision: 94.55%, Recall: 89.66%

[ZONE_DETECTION]
  Accuracy: 90.91%, Precision: 95.00%, Recall: 90.48%

[ERROR ANALYSIS - Top 5 Mismatches]
1. Frame  145 | Person | MISSED_DETECTION     | GT=Yes Pred=No
2. Frame  234 | Helmet | FALSE_NEGATIVE       | GT=Yes Pred=No
3. Frame  89  | Person | FALSE_POSITIVE       | GT=No  Pred=Yes
...
```

---

## Phase 5: Debug Errors (See DEBUGGING_GUIDE.md)

For each error:
1. **Find problematic frame:** `eval_frame_XXX.jpg`
2. **Get system's detection:** Save crop around detected area
3. **Test component in isolation:** Run YOLO/PPE model on crop
4. **Identify root cause:** Confidence threshold? Zone polygon? Model accuracy?
5. **Implement fix:** Adjust config.py or retrain model
6. **Re-evaluate:** Repeat Phase 2-4

---

## Files in Evaluation Framework

| File | Purpose |
|------|---------|
| [ground_truth.csv](ground_truth.csv) | Your manual labels (expected output) |
| [predictions.csv](predictions.csv) | System's predictions (actual output) |
| [evaluation_metrics.py](evaluation_metrics.py) | Comparison script (TP/FP/FN calculation) |
| [GROUND_TRUTH_TEMPLATE.md](GROUND_TRUTH_TEMPLATE.md) | How to manually label frames |
| [LOGGING_FORMAT_SPEC.md](LOGGING_FORMAT_SPEC.md) | System output format definition |
| [EVALUATION_EXAMPLES.md](EVALUATION_EXAMPLES.md) | Worked examples (4 scenarios) |
| [DEBUGGING_GUIDE.md](DEBUGGING_GUIDE.md) | How to fix errors |
| [evaluation_report.txt](evaluation_report.txt) | Auto-generated summary report |

---

## Metric Definitions (Quick Reference)

```
TP (True Positive)  = Ground truth says YES, System says YES ✓
FN (False Negative) = Ground truth says YES, System says NO  ✗ (MISSED)
FP (False Positive) = Ground truth says NO, System says YES  ✗ (FALSE ALARM)

Accuracy = TP / (TP+FP+FN)              [% overall correctness]
Precision = TP / (TP+FP)                [% of detections that were right]
Recall = TP / (TP+FN)                   [% of actual events we caught]

High Precision, Low Recall   → Few false alarms, but many missed violations
Low Precision, High Recall   → Catches all violations, but too many false alarms
```

**Target for production:**
- Accuracy: >90%
- Precision: >95% (avoid alert fatigue)
- Recall: >90% (catch violations)

---

## Common Adjustments

```python
# In config.py:

# More detections (catch more violations, may increase false alarms)
MIN_CONFIDENCE = 0.35  # was 0.45

# Fewer detections (reduce false alarms, may miss violations)
MIN_CONFIDENCE = 0.55  # was 0.45

# Stricter helmet check
PPE_CONFIDENCE_THRESHOLD = 0.50  # was 0.45

# More sensitive zone detection
# Edit RESTRICTED_ZONES polygon to match exact danger area
```

---

## One-Liner Commands

```bash
# Quick eval on sample data
py -3 evaluation_metrics.py

# Test person detection on single frame
py -3 -c "from ultralytics import YOLO; m=YOLO('yolov8n.pt'); m('frame.jpg').show()"

# Extract predictions from log file
py -3 -c "import re; lines = open('predictions.log').readlines(); print([re.search(r'frame=(\d+)', l).group(1) for l in lines if 'PREDICTION' in l])"

# Check Python/environment
py -3 -c "import cv2, torch, ultralytics; print(f'CV2 {cv2.__version__}, Torch {torch.__version__}, YOLO {ultralytics.__version__}')"
```

---

## Checklist: Before First Evaluation

- [ ] `ground_truth.csv` created with 50+ frames
- [ ] `predictions.csv` has same number of frames
- [ ] Both CSVs have required columns: frame_id, timestamp_seconds, person_present, helmet_visible, in_restricted_zone
- [ ] All person_present values are "Yes" or "No"
- [ ] All helmet_visible are "Yes"/"No"/"N/A" (N/A only when person_present=No)
- [ ] Run `py -3 evaluation_metrics.py` successfully
- [ ] Accuracy >85% for all three metrics (or identify why not)

---

## Next Steps

1. **Run evaluation now** with sample files provided
2. **Interpret results** using EVALUATION_EXAMPLES.md
3. **Identify top 3 errors** from [ERROR ANALYSIS]
4. **Debug each error** using DEBUGGING_GUIDE.md
5. **Adjust config.py** based on fixes needed
6. **Re-evaluate** and repeat until targets met

---

## Support

- **How do I create ground truth?** → See GROUND_TRUTH_TEMPLATE.md
- **What output format does the system need?** → See LOGGING_FORMAT_SPEC.md
- **How do I interpret results?** → See EVALUATION_EXAMPLES.md (4 worked scenarios)
- **How do I fix errors?** → See DEBUGGING_GUIDE.md (per error type)
- **My accuracy is too low** → Check DEBUGGING_GUIDE.md "Prevention Strategies"
- **How do I improve accuracy?** → Lower confidence thresholds OR retrain models OR fix zone polygons

---

**Start here:**
```bash
cd "d:\Projects\Hackive (angadi)\vision"
py -3 evaluation_metrics.py
```

See the README files above for detailed explanations!
