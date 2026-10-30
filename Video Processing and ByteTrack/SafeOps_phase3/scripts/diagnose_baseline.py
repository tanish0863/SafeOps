import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
import json
import warnings
warnings.filterwarnings("ignore")

from src.tracking.worker_tracker import WorkerTracker
from src.evaluation.mot_metrics import evaluate_tracking

scenarios = [
    'scenario1_single_worker_moving',
    'scenario2_two_workers_crossing',
    'scenario3_two_workers_close_together',
    'scenario4_worker_leaving_reentering',
    'scenario5_temp_detection_failure',
    'scenario6_partial_occlusion'
]

print("=== BASELINE EVALUATION RUN ===")
for s in scenarios:
    vpath = f'videos/scenarios/{s}.mp4'
    gpath = f'videos/scenarios/ground_truth/{s}_gt.json'
    cap = cv2.VideoCapture(vpath)
    tracker = WorkerTracker(half=False)
    tracker.reset()
    preds = {}
    fn = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        workers = tracker.update(frame, frame_number=fn)
        preds[fn] = [{'track_id': w['track_id'], 'bbox_xyxy': w['bbox_xyxy'], 'confidence': w['confidence']} for w in workers]
        fn += 1
    cap.release()
    with open(gpath) as f:
        gt = {int(k): v for k, v in json.load(f).items()}
    res = evaluate_tracking(gt, preds)
    print(f"{s}:")
    print(f"  GT={res['ground_truth_trajectories']}, Unique={res['unique_track_ids']}, IDF1={res['idf1']}, HOTA={res['hota']}, IDSW={res['id_switches']}, Frag={res['fragmentation']}, FP={res['false_positives']}, FN={res['false_negatives']}")
