import cv2
from ultralytics import YOLO
import os
import config

# 🔴 MODELS
# STAGE 1: Standard Person Detector (COCO)
person_model = YOLO(config.PERSON_MODEL_PATH)
# STAGE 2: Specialized PPE Model
HELMET_MODEL_PATH = r'D:\Projects\Hackive (angadi)\vision\runs\detect\train\weights\best.pt'
helmet_model = YOLO(HELMET_MODEL_PATH)

print(f'USING PERSON MODEL: {config.PERSON_MODEL_PATH}')
print('USING HELMET MODEL:', HELMET_MODEL_PATH)

cap = cv2.VideoCapture(0)

status_buffer = []
BUFFER_SIZE = 5

def get_stable_status(new_status):
    status_buffer.append(new_status)
    if len(status_buffer) > BUFFER_SIZE:
        status_buffer.pop(0)
    if not status_buffer: return 'UNKNOWN'
    return max(set(status_buffer), key=status_buffer.count)

print('\n--- 2-STAGE PIPELINE ACTIVE ---')
print('Press ESC to exit.\n')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # STAGE 1: Run Person Detection
    person_results = person_model(frame, conf=0.4, verbose=False)[0]

    person_detected = False
    helmet_detected = False
    no_helmet_detected = False

    for box in person_results.boxes:
        cls = int(box.cls[0])
        label = person_model.names[cls]

        if label != 'person':
            continue

        person_detected = True
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # 🔴 DRAW PERSON BOX (BLUE)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 🔴 STAGE 2: WIDE CROP (TOP 75% - Includes upper torso context)
        head_y2 = int(y1 + 0.75 * (y2 - y1))
        # Ensure crop is within frame boundaries
        head_y1_clip = max(0, y1)
        head_y2_clip = min(frame.shape[0], head_y2)
        head_x1_clip = max(0, x1)
        head_x2_clip = min(frame.shape[1], x2)
        
        head = frame[head_y1_clip:head_y2_clip, head_x1_clip:head_x2_clip]

        if head.size == 0:
            continue

        # 🔴 IMPROVED INPUT QUALITY (320x320 for better small object context)
        head_resized = cv2.resize(head, (320, 320))

        # 🔴 DEBUG MODE: LOW CONFIDENCE (Force reveal model uncertainty)
        helmet_results = helmet_model(head_resized, conf=0.05, verbose=False)[0]
        
        # VERIFICATION PRINT (Check if model is actually learning/seeing anything)
        detections = [helmet_model.names[int(b.cls[0])] for b in helmet_results.boxes]
        if len(detections) > 0:
            print(f"Model Detections: {detections}")

        # Get crop dimensions for coordinate mapping back from 320x320
        cw = head_x2_clip - head_x1_clip
        ch = head_y2_clip - head_y1_clip

        for hbox in helmet_results.boxes:
            hcls = int(hbox.cls[0])
            hlabel = helmet_model.names[hcls]
            hconf = float(hbox.conf[0])

            # Get normalized coordinates from resized head
            hx1_norm, hy1_norm, hx2_norm, hy2_norm = hbox.xyxyn[0]

            # Adjust coordinates to original frame based on actual crop size
            hx1 = int(hx1_norm * cw) + head_x1_clip
            hx2 = int(hx2_norm * cw) + head_x1_clip
            hy1 = int(hy1_norm * ch) + head_y1_clip
            hy2 = int(hy2_norm * ch) + head_y1_clip

            if hlabel == 'with_helmet':
                helmet_detected = True
                color = (0, 255, 0) # Green
                display_label = 'HELMET'
            elif hlabel == 'without_helmet':
                no_helmet_detected = True
                color = (0, 0, 255) # Red
                display_label = 'NO HELMET'
            else:
                color = (200, 200, 200)
                display_label = hlabel

            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)
            cv2.putText(frame, f'{display_label} {hconf:.2f}', (hx1, hy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 🔴 FINAL DECISION ENGINE
    if not person_detected:
        status = 'NO PERSON'
    elif no_helmet_detected:
        status = 'NO HELMET'
    elif helmet_detected:
        status = 'HELMET DETECTED'
    else:
        status = 'UNKNOWN'

    stable_status = get_stable_status(status)

    # 🔴 DISPLAY BIG STATUS TEXT
    cv2.rectangle(frame, (0, 0), (450, 80), (0, 0, 0), -1) # Background
    
    if stable_status == 'HELMET DETECTED':
        color = (0, 255, 0)
    elif stable_status == 'NO HELMET':
        color = (0, 0, 255)
    elif stable_status == 'NO PERSON':
        color = (255, 255, 0)
    else:
        color = (128, 128, 128)

    cv2.putText(frame, stable_status, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow('2-Stage Helmet Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
