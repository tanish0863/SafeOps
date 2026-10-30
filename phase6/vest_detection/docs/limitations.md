# Safety Vest Detection — Limitations & Edge Cases

## 1. Overview

While the Safety Vest Detection module provides high-speed, high-confidence detection on real-time video streams, computer vision models have known physical and environmental limitations. This document specifies those limitations and explains the safeguards implemented to prevent false compliance decisions.

---

## 2. Environmental Limitations

### 2.1 Low-Light and Night Shift Operations
- Limitation: In unlit construction pits or during nocturnal operations with poor illumination (< 25 mean grayscale intensity), the color and retroreflective properties of safety vests cannot be differentiated from regular jackets.
- System Safeguard: Rather than falsely asserting ABSENT (which would penalize compliant workers), the engine flags these frames as UNKNOWN with the reason 'Torso lighting too low'.

### 2.2 Severe Glare and Overexposure
- Limitation: Direct solar glare or high-intensity floodlights directed into the camera lens cause sensor saturation (mean brightness > 245), washing out color information and retroreflective stripe contrast.
- System Safeguard: The optical quality analyzer detects extreme saturation and returns UNKNOWN.

### 2.3 Atmospheric Particles and Lens Dirt
- Limitation: Heavy rain, dense steam, concrete dust, or water droplets on the camera dome reduce high-frequency edges, leading to Laplacian blur variance below the operational threshold.
- System Safeguard: The torso assessor validates image sharpness and marks low-contrast smudged crops as UNKNOWN.

---

## 3. Scale and Distance Limitations

### 3.1 Small Person / Distant Workers
- Limitation: Workers situated far from the camera where the torso bounding box occupies fewer than 48 horizontal or 64 vertical pixels lack sufficient spatial resolution for reliable vest detection.
- System Safeguard: Minimum torso dimensions are strictly enforced (min_torso_width=48, min_torso_height=64). Crops below this threshold return UNKNOWN without invoking model inference.

### 3.2 Extreme Camera Angles
- Limitation: Cameras mounted directly overhead (> 75 degrees zenith angle) show only the worker's hardhat and shoulders, occluding the front and rear vest panels.
- Recommendation: Mount cameras at 15 to 45 degree angles for optimal torso visibility.

---

## 4. Attire and Occlusion Ambiguities

### 4.1 Bright Everyday Clothing
- Limitation: Workers wearing fluorescent yellow, orange, or lime sweatshirts without retroreflective bands may occasionally trigger false-positive vest detections under full-frame detection if confidence threshold is set too low.
- System Safeguard: The module sets a default confidence threshold of 0.45 and validates intersection with the torso region. The model has been trained specifically on NO-Safety Vest examples to penalize non-reflective bright clothing.

### 4.2 Reflective Rain Gear and Winter Parkas
- Limitation: In rainy or cold weather, workers may wear hooded winter parkas or raincoats with reflective piping that are not certified high-visibility safety vests (ANSI/ISEA 107 Class 2/3).
- System Safeguard: Upstream pipeline operators should tune confidence_threshold when monitoring cold-weather sites.

### 4.3 Carrying Equipment / Toolboxes
- Limitation: Workers carrying plywood sheets, conduit bundles, or large toolboxes in front of their chest physically occlude the torso.
- System Safeguard: If no vest is detected and the torso is partially occluded, upstream tracking systems should require multiple consecutive frames before confirming non-compliance.

---

## 5. Negative Class (`NO-Safety Vest`) Sensitivity Limitation

### 5.1 Empirical Verification Finding
During empirical audits on real-world photographs (`worker_no_vest.jpg`, `normal.jpg`, `bright.jpg`, `dark.jpg`, and full-scene CCTV images), the pre-trained model (`models/yolov8n_vest_best.pt`) demonstrated **high precision for `Safety Vest` (Class ID 7)** but **extreme scarcity / low sensitivity for explicit `NO-Safety Vest` (Class ID 4)** predictions.
- On workers wearing standard work attire (shirts, jackets) without a vest, the model predominantly outputs class `Person` (Class ID 5) or produces zero vest-related detections.
- The model almost never emits an explicit `NO-Safety Vest` label on standard civilian/industrial clothing under normal lighting.

### 5.2 Architectural Safeguard: No Manufactured Evidence
To guarantee compliance integrity:
1. **Never Manufacture Negatives:** The engine strictly refuses to synthesize artificial negative predictions or automatically convert an absence of detection into an `ABSENT` verdict.
2. **Strict `UNKNOWN` Fallback:** When a worker's torso is detected without a positive `WITH_VEST` and without validated negative `NO_VEST` evidence, the module strictly outputs:
   - `status: UNKNOWN`
   - `confidence: 0.50`
   - `decision_reasons: ["No safety vest or negative vest evidence detected in the torso region."]`
3. **Requirement for `ABSENT`:** An `ABSENT` verdict is strictly emitted **only** when an explicit, validated `NO_VEST` prediction is output by the model meeting `no_vest_confidence_threshold` and passing all optical quality checks (lighting, sharpness, minimum resolution). Missing detections will never trigger an unauthorized `ABSENT` citation.
