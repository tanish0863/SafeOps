import os, sys
sys.path.insert(0, os.path.abspath("."))
import cv2, json
from src.tracking.worker_tracker import WorkerTracker
from src.evaluation.mot_metrics import evaluate_tracking, compute_iou

cap = cv2.VideoCapture('videos/scenarios/scenario6_partial_occlusion.mp4')
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

with open('videos/scenarios/ground_truth/scenario6_partial_occlusion_gt.json') as f:
    gt = {int(k): v for k, v in json.load(f).items()}

res = evaluate_tracking(gt, preds)
print('Scenario 6 Evaluation:')
for k, v in res.items():
    print(f'  {k}: {v}')

print('\nFrames 10-25 detailed:')
for f in range(10, 26):
    gts = gt.get(f, [])
    prs = preds.get(f, [])
    prs_str = [(p['track_id'], round(p['confidence'], 2), [round(x, 1) for x in p['bbox_xyxy']]) for p in prs]
    print(f'Frame {f}: GT={len(gts)} | Preds={prs_str}')

