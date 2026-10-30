"""MOT (Multi-Object Tracking) Evaluation Metrics for SafeOps Phase 3.

Computes exact ground-truth based MOT metrics:
- ID Switches (IDSW)
- IDF1 (ID Precision, ID Recall, IDF1 via Global Hungarian Matching)
- HOTA (Higher Order Tracking Accuracy with DetA and AssA at alpha=0.5)
- Track Fragmentation (Frag)
- Track Duration (Mostly Tracked MT, Partially Tracked PT, Mostly Lost ML)
- Detection Coverage (Recall, Precision)
- Unique Track IDs
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute standard Intersection over Union (IoU) between two xyxy boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0

    w1 = max(0.0, box1[2] - box1[0])
    h1 = max(0.0, box1[3] - box1[1])
    w2 = max(0.0, box2[2] - box2[0])
    h2 = max(0.0, box2[3] - box2[1])

    area1 = w1 * h1
    area2 = w2 * h2
    if area1 <= 0.0 or area2 <= 0.0:
        return 0.0

    union = area1 + area2 - inter
    return inter / union if union > 0.0 else 0.0


def evaluate_tracking(
    gt_frames: dict[int, list[dict]],
    tracker_frames: dict[int, list[dict]],
    iou_threshold: float = 0.5,
) -> dict:
    """Evaluate tracker predictions against ground-truth annotations across all frames.

    Args:
        gt_frames: {frame_idx: [{"id": int, "bbox": [x1, y1, x2, y2]}]}
        tracker_frames: {frame_idx: [{"track_id": int, "bbox_xyxy": [x1, y1, x2, y2]}]}
        iou_threshold: minimum IoU to consider a detection as a match (default 0.5)

    Returns:
        Dictionary of computed MOT metrics.
    """
    all_frame_indices = sorted(set(gt_frames.keys()) | set(tracker_frames.keys()))

    # Frame-by-frame matching
    tp_total = 0
    fp_total = 0
    fn_total = 0

    # Tracking trajectories
    gt_history: dict[int, list[tuple[int, int]]] = defaultdict(list)  # gt_id -> list of (frame, tracker_id)
    gt_presence: dict[int, list[int]] = defaultdict(list)             # gt_id -> list of frames present
    tracker_presence: dict[int, list[int]] = defaultdict(list)        # tracker_id -> list of frames present

    # Global contingency counts for IDF1: (gt_id, tracker_id) -> count of matches
    global_matches: dict[tuple[int, int], int] = defaultdict(int)
    all_gt_ids = set()
    all_tracker_ids = set()

    # Per-frame TP pairs: frame_idx -> list of (gt_id, tracker_id)
    frame_tp_pairs: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for f in all_frame_indices:
        gts = gt_frames.get(f, [])
        preds = tracker_frames.get(f, [])

        for g in gts:
            all_gt_ids.add(g["id"])
            gt_presence[g["id"]].append(f)
        for p in preds:
            all_tracker_ids.add(p["track_id"])
            tracker_presence[p["track_id"]].append(f)

        if not gts and not preds:
            continue
        if not gts:
            fp_total += len(preds)
            continue
        if not preds:
            fn_total += len(gts)
            continue

        # IoU cost matrix
        iou_matrix = np.zeros((len(gts), len(preds)), dtype=np.float32)
        for i, g in enumerate(gts):
            for j, p in enumerate(preds):
                iou_matrix[i, j] = compute_iou(g["bbox"], p["bbox_xyxy"])

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched_gts = set()
        matched_preds = set()

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_threshold:
                gid = gts[r]["id"]
                tid = preds[c]["track_id"]
                matched_gts.add(r)
                matched_preds.add(c)
                tp_total += 1
                gt_history[gid].append((f, tid))
                global_matches[(gid, tid)] += 1
                frame_tp_pairs[f].append((gid, tid))

        fp_total += len(preds) - len(matched_preds)
        fn_total += len(gts) - len(matched_gts)

    # 1. ID Switches (IDSW) & Track Fragmentation (Frag)
    id_switches = 0
    fragmentation = 0

    for gid, matches in gt_history.items():
        if not matches:
            continue
        # ID switches
        prev_tid = matches[0][1]
        for f, tid in matches[1:]:
            if tid != prev_tid:
                id_switches += 1
                prev_tid = tid

        # Fragmentation: gap in tracked frames where GT was present
        matched_frames = [f for f, _ in matches]
        for f1, f2 in zip(matched_frames, matched_frames[1:]):
            if f2 - f1 > 1:
                # Was GT present in between?
                gt_f_span = [f for f in gt_presence[gid] if f1 < f < f2]
                if gt_f_span:
                    fragmentation += 1

    # 2. Track Duration Categories (MT, PT, ML)
    mostly_tracked = 0
    partially_tracked = 0
    mostly_lost = 0

    for gid, pres in gt_presence.items():
        matched_count = len(gt_history.get(gid, []))
        total_gt_frames = len(pres)
        ratio = matched_count / total_gt_frames if total_gt_frames > 0 else 0.0
        if ratio >= 0.80:
            mostly_tracked += 1
        elif ratio >= 0.20:
            partially_tracked += 1
        else:
            mostly_lost += 1

    # 3. IDF1 via Global Hungarian Matching
    gt_list = sorted(all_gt_ids)
    tr_list = sorted(all_tracker_ids)
    idtp = 0
    if gt_list and tr_list:
        cost_matrix = np.zeros((len(gt_list), len(tr_list)), dtype=np.float32)
        for i, gid in enumerate(gt_list):
            for j, tid in enumerate(tr_list):
                cost_matrix[i, j] = global_matches.get((gid, tid), 0)

        r_ind, c_ind = linear_sum_assignment(-cost_matrix)
        for r, c in zip(r_ind, c_ind):
            idtp += int(cost_matrix[r, c])

    total_gt_boxes = sum(len(gts) for gts in gt_frames.values())
    total_tr_boxes = sum(len(preds) for preds in tracker_frames.values())
    idfp = total_tr_boxes - idtp
    idfn = total_gt_boxes - idtp
    id_denom = 2 * idtp + idfp + idfn
    idf1 = (2 * idtp / id_denom) if id_denom > 0 else 0.0
    id_precision = (idtp / total_tr_boxes) if total_tr_boxes > 0 else 0.0
    id_recall = (idtp / total_gt_boxes) if total_gt_boxes > 0 else 0.0

    # 4. HOTA (alpha = 0.5)
    # DetA = TP / (TP + FP + FN)
    det_denom = tp_total + fp_total + fn_total
    deta = (tp_total / det_denom) if det_denom > 0 else 0.0

    # AssA: for each TP pair (g, h), calculate jaccard of their respective TP matches
    # TP_g: all TPs involving GT g; TP_h: all TPs involving Tracker h; TP_gh: all TPs involving both (g, h)
    tp_g_count = defaultdict(int)
    tp_h_count = defaultdict(int)
    tp_gh_count = defaultdict(int)

    all_tps = []
    for f, pairs in frame_tp_pairs.items():
        for gid, tid in pairs:
            tp_g_count[gid] += 1
            tp_h_count[tid] += 1
            tp_gh_count[(gid, tid)] += 1
            all_tps.append((gid, tid))

    if all_tps:
        ass_scores = []
        for gid, tid in all_tps:
            intersection = tp_gh_count[(gid, tid)]
            union = tp_g_count[gid] + tp_h_count[tid] - intersection
            ass_scores.append(intersection / union if union > 0 else 0.0)
        assa = float(np.mean(ass_scores))
    else:
        assa = 0.0

    hota = float(np.sqrt(deta * assa))

    # 5. Detection Coverage
    recall = (tp_total / total_gt_boxes) if total_gt_boxes > 0 else 0.0
    precision = (tp_total / (tp_total + fp_total)) if (tp_total + fp_total) > 0 else 0.0

    return {
        "ground_truth_trajectories": len(all_gt_ids),
        "unique_track_ids": len(all_tracker_ids),
        "total_gt_detections": total_gt_boxes,
        "total_tracker_detections": total_tr_boxes,
        "true_positives": tp_total,
        "false_positives": fp_total,
        "false_negatives": fn_total,
        "id_switches": id_switches,
        "fragmentation": fragmentation,
        "idf1": round(idf1, 4),
        "id_precision": round(id_precision, 4),
        "id_recall": round(id_recall, 4),
        "deta": round(deta, 4),
        "assa": round(assa, 4),
        "hota": round(hota, 4),
        "detection_coverage_recall": round(recall, 4),
        "precision": round(precision, 4),
        "mostly_tracked": mostly_tracked,
        "partially_tracked": partially_tracked,
        "mostly_lost": mostly_lost,
    }

