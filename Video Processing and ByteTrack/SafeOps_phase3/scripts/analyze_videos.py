import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.tracking.worker_tracker import WorkerTracker
from src.tracking.track_stats import summarize
import cv2, json, os

videos = ['v2.mp4', 'v4.mp4', 'v5.mp4']
for vname in videos:
    vpath = f'D:/Projects/new/videos/{vname}'
    if not os.path.exists(vpath):
        continue
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'=== Testing {vname} ({total} frames @ {fps:.1f} fps) ===')
    tracker = WorkerTracker(tracker='src/tracking/custom_bytetrack.yaml')
    out_jsonl = f'outputs/{vname}.jsonl'
    with open(out_jsonl, 'w') as f:
        fn = 0
        limit = min(300, total)
        while fn < limit:
            ok, frame = cap.read()
            if not ok: break
            workers = tracker.update(frame)
            for w in workers:
                f.write(json.dumps({'frame_number': fn, 'track_id': w['track_id'], 'bbox_xyxy': w['bbox_xyxy']}) + '\n')
            fn += 1
    cap.release()
    s = summarize(out_jsonl, total_frames=fn)
    print(f'{vname} Stats (first {fn} frames):')
    print('  Unique IDs:', s['unique_track_ids'])
    print('  Max workers in frame:', s['max_workers_in_one_frame'])
    print('  Frames with workers:', s['frames_with_workers'])
    print('  Short tracks:', s['short_tracks'])
    for tid, t in s['tracks'].items():
        print(f"   ID {tid}: frame {t['first_frame']}..{t['last_frame']} (present: {t['frames_present']}, gaps: {t['gaps']})")
