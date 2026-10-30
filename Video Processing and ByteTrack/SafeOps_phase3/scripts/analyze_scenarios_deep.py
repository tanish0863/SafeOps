import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
import json
import warnings
warnings.filterwarnings("ignore")

from src.tracking.worker_tracker import WorkerTracker

def analyze_scenario(s):
    vpath = f"videos/scenarios/{s}.mp4"
    gpath = f"videos/scenarios/ground_truth/{s}_gt.json"
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n==========================================")
    print(f"ANALYZING: {s} ({total_frames} frames, {fps} fps)")
    print(f"==========================================")
    
    with open(gpath) as f:
        gt = {int(k): v for k, v in json.load(f).items()}
        
    tracker = WorkerTracker()
    tracker.reset()
    
    fn = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Inspect raw model detections before ByteTrack
        yolo_res = tracker.model.predict(
            frame,
            classes=[0],
            conf=0.05, # low threshold to see what YOLO sees
            device=tracker.device,
            verbose=False
        )[0]
        raw_confs = yolo_res.boxes.conf.cpu().numpy().tolist() if yolo_res.boxes is not None else []
        raw_boxes = yolo_res.boxes.xyxy.cpu().numpy().tolist() if yolo_res.boxes is not None else []
        
        # Now run tracker update
        workers = tracker.update(frame, frame_number=fn)
        tr_ids = [w["track_id"] for w in workers]
        tr_confs = [round(w["confidence"], 2) for w in workers]
        gt_ids = [g["id"] for g in gt.get(fn, [])]
        
        # Print when anything interesting happens: mismatch in count or IDs
        if len(tr_ids) != len(gt_ids) or max(tr_ids, default=0) > max(gt_ids, default=0) or fn % 10 == 0:
            print(f"Frame {fn:02d}: GT_ids={gt_ids} | TR_ids={tr_ids} (conf={tr_confs}) | raw_yolo_dets={len(raw_boxes)} confs={[round(c, 2) for c in raw_confs]}")
        fn += 1
    cap.release()

if __name__ == "__main__":
    for s in [
        'scenario2_two_workers_crossing',
        'scenario4_worker_leaving_reentering',
        'scenario5_temp_detection_failure',
        'scenario6_partial_occlusion'
    ]:
        analyze_scenario(s)

