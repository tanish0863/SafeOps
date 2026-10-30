# EVALUATION FRAMEWORK - WORKED EXAMPLES

This document shows concrete examples of how to use the evaluation framework and interpret results.

## Example 1: Perfect System (100% Accuracy)

**Scenario:** System detects everything correctly.

### Ground Truth (5 frames)

```csv
frame_id,timestamp_seconds,person_present,helmet_visible,in_restricted_zone
1,0.0,No,N/A,N/A
2,0.03,Yes,Yes,No
3,0.06,Yes,No,No
4,0.1,Yes,No,Yes
5,0.13,No,N/A,N/A
```

### Predictions (system output)

```csv
frame_id,timestamp_seconds,person_present,helmet_visible,in_restricted_zone
1,0.0,No,N/A,N/A
2,0.03,Yes,Yes,No
3,0.06,Yes,No,No
4,0.1,Yes,No,Yes
5,0.13,No,N/A,N/A
```

### Evaluation Result

```
[PERSON_DETECTION]
  True Positives (TP):  2
  False Positives (FP): 0
  False Negatives (FN): 0
  Total Predictions:    5
  ---
  Accuracy:  100.00%
  Precision: 100.00%
  Recall:    100.00%

[HELMET_DETECTION]
  True Positives (TP):  2
  False Positives (FP): 0
  False Negatives (FN): 0
  Total Predictions:    2
  ---
  Accuracy:  100.00%
  Precision: 100.00%
  Recall:    100.00%

[ZONE_DETECTION]
  True Positives (TP):  1
  False Positives (FP): 0
  False Negatives (FN): 0
  Total Predictions:    2
  ---
  Accuracy:  100.00%
  Precision: 100.00%
  Recall:    100.00%
```

**Interpretation:** System is perfect—all TP, no FP/FN errors.

---

## Example 2: System with Errors (Real Scenario)

**Scenario:** System misses a helmet violation and has a false positive zone detection.

### Ground Truth (5 frames)

```csv
frame_id,timestamp_seconds,person_present,helmet_visible,in_restricted_zone
1,0.0,No,N/A,N/A
2,0.03,Yes,Yes,No
3,0.06,Yes,No,No
4,0.1,Yes,No,Yes
5,0.13,No,N/A,N/A
```

### Predictions (system output with errors)

```csv
frame_id,timestamp_seconds,person_present,helmet_visible,in_restricted_zone
1,0.0,No,N/A,N/A
2,0.03,Yes,Yes,No
3,0.06,Yes,Yes,No
4,0.1,Yes,No,No
5,0.13,No,N/A,N/A
```

**Errors:**
- Frame 3: Ground truth says "helmet=No" (violation), but system predicted "helmet=Yes" (missed violation) → **False Negative**
- Frame 4: Ground truth says "zone=Yes", but system predicted "zone=No" (missed zone breach) → **False Negative**

### Evaluation Result

```
[PERSON_DETECTION]
  True Positives (TP):  2
  False Positives (FP): 0
  False Negatives (FN): 0
  Total Predictions:    5
  ---
  Accuracy:  100.00%

[HELMET_DETECTION]
  True Positives (TP):  1
  False Positives (FP): 0
  False Negatives (FN): 1
  Total Predictions:    2
  ---
  Accuracy:  50.00%
  Precision: 100.00%
  Recall:    50.00%

[ZONE_DETECTION]
  True Positives (TP):  0
  False Positives (FP): 0
  False Negatives (FN): 1
  Total Predictions:    2
  ---
  Accuracy:  50.00%
  Precision: 0.00%
  Recall:    0.00%
```

**Interpretation:**
- Person detection: Perfect (100%)
- Helmet detection: **Only 50% accurate**—system missed 1 out of 2 cases (Recall=50%)
- Zone detection: **Only 50% accurate**—system missed 1 out of 1 zone breach (Recall=0%)

**Action:** Check helmet detection model and zone logic for robustness.

---

## Example 3: False Positive Scenario

**Scenario:** System detects person when there isn't one (ghost detection).

### Ground Truth (5 frames)

```csv
frame_id,timestamp_seconds,person_present,helmet_visible,in_restricted_zone
1,0.0,No,N/A,N/A
2,0.03,No,N/A,N/A
3,0.06,Yes,No,No
4,0.1,No,N/A,N/A
5,0.13,No,N/A,N/A
```

### Predictions (system output with false positive)

```csv
frame_id,timestamp_seconds,person_present,helmet_visible,in_restricted_zone
1,0.0,No,N/A,N/A
2,0.03,Yes,No,No
3,0.06,Yes,No,No
4,0.1,No,N/A,N/A
5,0.13,No,N/A,N/A
```

**Error:**
- Frame 2: Ground truth says "person=No", but system predicted "person=Yes" → **False Positive**

### Evaluation Result

```
[PERSON_DETECTION]
  True Positives (TP):  1
  False Positives (FP): 1
  False Negatives (FN): 0
  Total Predictions:    5
  ---
  Accuracy:  80.00%
  Precision: 50.00%
  Recall:    100.00%

[HELMET_DETECTION]
  True Positives (TP):  1
  False Positives (FP): 0
  False Negatives (FN): 0
  Total Predictions:    2
  ---
  Accuracy:  100.00%
```

**Interpretation:**
- Person detection: **80% accurate** but **precision only 50%** (system triggers false alarms)
- This means 1 in 2 person detections is wrong (the FP on frame 2)

**Action:** Check YOLO confidence threshold—lower it to reduce ghost detections, or verify video source quality.

---

## Example 4: Metric Definitions Reference

### Definitions

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Accuracy** | TP / (TP+FP+FN) | Overall correctness across all predictions |
| **Precision** | TP / (TP+FP) | Of detections made, how many were correct? |
| **Recall** | TP / (TP+FN) | Of actual violations, how many did we catch? |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Balanced score (catch violations without false alarms) |

### Error Types

| Error | When | Example | Fix |
|-------|------|---------|-----|
| **TP (True Positive)** | Ground truth=violation, Predicted=violation | Frame 3: no helmet (GT) → no helmet (pred) ✓ | Good! |
| **FP (False Positive)** | Ground truth=no violation, Predicted=violation | Frame 2: no person (GT) → person (pred) ✗ | Lower detect threshold |
| **FN (False Negative)** | Ground truth=violation, Predicted=no violation | Frame 4: helmet=no (GT) → helmet=yes (pred) ✗ | Improve model or post-process |

---

## Example 5: Multi-Metric Dashboard

**Real scenario with all three metrics:**

```
VISION SYSTEM EVALUATION METRICS
======================================================================

[PERSON_DETECTION]
  True Positives (TP):  145
  False Positives (FP): 8
  False Negatives (FN): 5
  Total Predictions:    158
  ---
  Accuracy:  94.30%
  Precision: 94.74%  (TP / (TP+FP))
  Recall:    96.67%  (TP / (TP+FN))

[HELMET_DETECTION]
  True Positives (TP):  52
  False Positives (FP): 3
  False Negatives (FN): 6
  Total Predictions:    61
  ---
  Accuracy:  91.80%
  Precision: 94.55%
  Recall:    89.66%

[ZONE_DETECTION]
  True Positives (TP):  38
  False Positives (FP): 2
  False Negatives (FN): 4
  Total Predictions:    44
  ---
  Accuracy:  90.91%
  Precision: 95.00%
  Recall:    90.48%

[ERROR ANALYSIS - Top 5 Mismatches]
----------------------------------------------------------------------
1. Frame  145 | Type: person   | MISSED_DETECTION               | GT=Yes Pred=No
2. Frame  234 | Type: helmet   | FALSE_NEGATIVE (helmet missed) | GT=Yes Pred=No
3. Frame  89  | Type: person   | FALSE_POSITIVE                 | GT=No Pred=Yes
4. Frame  512 | Type: zone     | FALSE_ALARM                    | GT=No Pred=Yes
5. Frame  445 | Type: helmet   | FALSE_NEGATIVE (helmet missed) | GT=Yes Pred=No
```

**Overall Assessment:**
- ✅ **Person detection: 94.3%** (good, slight FP issue)
- ✅ **Helmet detection: 91.8%** (acceptable, some FN but high precision)
- ✅ **Zone detection: 90.9%** (acceptable, good precision)

**Action items (by priority):**
1. Reduce false positives in person detection (frames 89 and others)
2. Improve helmet detection recall (missing detections on frames 234, 445)
3. Profile zone logic for edge cases

---

## How to Use This Framework

### Step 1: Collect Ground Truth

```bash
# Record video, manually annotate per frame with GROUND_TRUTH_TEMPLATE.md
# Output: ground_truth.csv
```

### Step 2: Collect Predictions

```bash
# Run your vision pipeline
python main.py 2>&1 | tee predictions.log

# Convert log to CSV (use LOGGING_FORMAT_SPEC.md conversion script)
python extract_predictions.py predictions.log > predictions.csv
```

### Step 3: Evaluate

```bash
python evaluation_metrics.py
```

### Step 4: Analyze Mismatches

Look at the [ERROR ANALYSIS] section to identify patterns:
- Many FPs on specific frame count range? → lighting or camera shift
- FNs on helmet detection? → occlusion or model confidence
- FPs on zone detection? → incorrect zone polygon definition

### Step 5: Iterate

- Adjust thresholds in `config.py`
- Retrain model with better labeled data
- Improve PPE detection confidence
- Validate zone polygon accuracy

---

## Quick Reference: Accuracy Targets

**Acceptable ranges for production:**

| Component | Accuracy Target | Notes |
|-----------|-----------------|-------|
| Person Detection | >95% | Must be reliable; FPs → alert fatigue |
| Helmet Detection | >90% | Some FN acceptable (redundant checks) |
| Zone Detection | >90% | FP < 5% (legal/liability) |

**Red flags:**
- Person detection <90% → system unreliable
- Helmet detection <80% → too many missed violations
- Zone detection <85% → incorrect geometry or lighting issues
