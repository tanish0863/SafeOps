import os, sys
sys.path.insert(0, os.path.abspath("."))
import cv2, json
from src.tracking.worker_tracker import WorkerTracker
from src.evaluation.mot_metrics import evaluate_tracking, compute_iou

cap = cv2.VideoCapture('videos/scenarios/scenario5_temp_detection_failure.mp4')
tracker = WorkerTracker(half=False)
tracker.reset()
preds = {}
fn = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    workers = tracker.update(frame, frame_number=fn)
    preds[fn] = [{'track_id': w['track_id'], 'bbox_xyxy': w['bbox_xyxy'], 'confidence': w['confidence']} for w in workers]
    fn += 1
cap.release()

with open('videos/scenarios/ground_truth/scenario5_temp_detection_failure_gt.json') as f:
    gt = {int(k): v for k, v in json.load(f).items()}

res = evaluate_tracking(gt, preds)
print('Evaluation result:')
for k, v in res.items():
    print(f'  {k}: {v}')

print('\nFrames 23-34 detailed:')
for f in range(23, 35):
    gts = gt.get(f, [])
    prs = preds.get(f, [])
    print(f'Frame {f}:')
    for g in gts:
        print(f'  GT {g["id"]} bbox={g["bbox"]}')
    for p in prs:
        iou_val = compute_iou(gts[0]['bbox'], p['bbox_xyxy']) if gts else 0.0
        print(f'  Pred {p["track_id"]} bbox={[round(x, 1) for x in p["bbox_xyxy"]]} IoU_with_GT1={round(iou_val, 3)}')

