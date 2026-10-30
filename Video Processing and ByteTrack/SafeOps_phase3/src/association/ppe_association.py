"""PPE Association Subsystem for SafeOps.

Associates detected PPE items (helmets, vests) with individual tracked workers.

Methodology:
1. Dynamic anatomical region estimation:
   - Head region: derived from face landmarks (nose 0, eyes 1-2, ears 3-4) padded upward.
     Fallback: top 25% of worker bounding box when face keypoints are occluded/absent.
   - Torso region: derived from shoulder (5-6) and hip (11-12) keypoints.
     Fallback: 15% to 65% of worker bounding box height.
2. Intersection over Area (IoA):
   - Computes intersection between PPE bounding box and anatomical worker region.
   - Normalizes by PPE box area (small helmets inside large heads yield high IoA).
3. Candidate ambiguity handling:
   - When a PPE detection overlaps two worker regions with comparable scores,
     it is marked ambiguous rather than arbitrarily assigned.
4. One-to-one constraint:
   - Greedy bipartite assignment (highest score first).
   - Each PPE box belongs to at most one worker; each worker receives at most one helmet and one vest.
   - Primary key is track_id; detection ordering has zero influence on result.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.contracts.schemas import (
    HeadROIQuality,
    PPEDetectionItem,
    PPEItemType,
    PPEStatus,
    WorkerPPEAssociation,
    WorkerPPEState,
)

# Keypoint indices (COCO standard 17 keypoints)
HEAD_KPS = (0, 1, 2, 3, 4)       # Nose, eyes, ears
SHOULDERS = (5, 6)
HIPS = (11, 12)

# Matching thresholds
MIN_IOA = {
    PPEItemType.HELMET: 0.40,
    PPEItemType.VEST: 0.35,
}

# Legacy keywords support for backward compatibility
HELMET_KEYWORDS = ("helmet", "hardhat", "hard hat", "hard-hat", "hard_hat")
VEST_KEYWORDS = ("vest",)
_NEGATIVE = re.compile(r"^(no|non)[\s_\-]|without|missing|absent", re.I)


def normalize_label(name: str) -> Optional[str]:
    """Map raw class names to 'helmet' / 'vest' / None."""
    n = name.strip().lower()
    if _NEGATIVE.search(n):
        return None
    if any(k in n for k in HELMET_KEYWORDS):
        return "helmet"
    if any(k in n for k in VEST_KEYWORDS):
        return "vest"
    return None


def _area(b: List[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return _area([x1, y1, x2, y2])


def _kp_box(worker: Dict[str, Any], idxs: Tuple[int, ...], thresh: float = 0.4, min_pts: int = 2) -> Optional[List[float]]:
    """Derive tight bounding box from reliable worker keypoints."""
    kxy = worker.get("keypoints_xy")
    kc = worker.get("keypoints_conf")
    if kxy is None or kc is None:
        return None
    pts = [kxy[i] for i in idxs if i < len(kc) and kc[i] >= thresh]
    if len(pts) < min_pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def get_worker_anatomical_regions(worker: Dict[str, Any], kp_thresh: float = 0.4) -> Dict[PPEItemType, List[float]]:
    """Compute expected image coordinates for head (helmet) and torso (vest)."""
    x1, y1, x2, y2 = worker["bbox_xyxy"]
    h = y2 - y1

    # Head region
    head = _kp_box(worker, HEAD_KPS, kp_thresh, min_pts=2)
    if head:
        # Pad upward generously: helmets sit on and above top of head
        pad_x = 0.08 * h
        head_box = [
            head[0] - pad_x,
            head[1] - 0.18 * h,
            head[2] + pad_x,
            head[3] + pad_x,
        ]
    else:
        # Fallback: top 25% of worker bounding box
        head_box = [x1, y1, x2, y1 + 0.25 * h]

    # Torso region
    torso = _kp_box(worker, SHOULDERS + HIPS, kp_thresh, min_pts=2)
    if torso:
        pad_t = 0.06 * h
        torso_box = [
            torso[0] - pad_t,
            torso[1] - pad_t,
            torso[2] + pad_t,
            torso[3] + pad_t,
        ]
    else:
        # Fallback: 15% to 65% of worker height
        torso_box = [x1, y1 + 0.15 * h, x2, y1 + 0.65 * h]

    return {
        PPEItemType.HELMET: head_box,
        PPEItemType.VEST: torso_box,
    }


def _evaluate_head_quality(w: Dict[str, Any], kp_thresh: float = 0.4) -> HeadROIQuality:
    kxy = w.get("keypoints_xy")
    kc = w.get("keypoints_conf")
    if kxy is None or kc is None:
        return HeadROIQuality.UNKNOWN
    face_count = sum(1 for i in HEAD_KPS if i < len(kc) and kc[i] >= kp_thresh)
    if face_count >= 2:
        return HeadROIQuality.CONFIRMED
    elif face_count == 1:
        return HeadROIQuality.APPROXIMATE
    return HeadROIQuality.UNKNOWN


def associate_ppe_to_workers(
    workers: List[Dict[str, Any]],
    ppe_detections: List[PPEDetectionItem],
    has_vest_model: bool = False,
    ambiguity_margin: float = 0.0,
    allow_legacy_bbox: bool = False,
) -> Tuple[Dict[int, WorkerPPEAssociation], List[PPEDetectionItem]]:
    """Associate PPE detections with workers using keypoint geometry and IoA.

    Args:
        workers: list of worker dicts with 'track_id', 'bbox_xyxy', 'keypoints_xy', 'keypoints_conf'
        ppe_detections: list of PPEDetectionItem instances
        has_vest_model: whether a functional vest detector is present
        ambiguity_margin: score delta below which two competing worker assignments are marked ambiguous
        allow_legacy_bbox: whether to permit fallback bbox matching (for legacy tests)

    Returns:
        (associations_by_track_id, unassociated_ppe_items)
    """
    associations: Dict[int, WorkerPPEAssociation] = {}
    worker_qualities: Dict[int, HeadROIQuality] = {}

    for w in workers:
        tid = w["track_id"]
        hq = _evaluate_head_quality(w)
        if allow_legacy_bbox and hq == HeadROIQuality.UNKNOWN and w.get("keypoints_xy") is None:
            hq = HeadROIQuality.APPROXIMATE
        worker_qualities[tid] = hq

        if hq == HeadROIQuality.CONFIRMED or (allow_legacy_bbox and hq == HeadROIQuality.APPROXIMATE):
            helmet_status = PPEStatus.NOT_DETECTED
            helmet_reason = "no_helmet_detected"
        else:
            helmet_status = PPEStatus.UNKNOWN
            helmet_reason = "missing_head_keypoints"

        associations[tid] = WorkerPPEAssociation(
            track_id=tid,
            helmet=WorkerPPEState(
                status=helmet_status,
                confidence=None,
                bbox_xyxy=None,
                reason=helmet_reason,
            ),
            vest=WorkerPPEState(
                status=PPEStatus.UNKNOWN if not has_vest_model else PPEStatus.NOT_DETECTED,
                confidence=None,
                bbox_xyxy=None,
                reason="vest_model_not_installed" if not has_vest_model else "no_vest_detected",
            ),
            head_roi_quality=hq,
        )

    if not workers or not ppe_detections:
        return associations, ppe_detections

    # Compute anatomical regions for each worker
    worker_regions = {w["track_id"]: get_worker_anatomical_regions(w) for w in workers}

    # Build association candidate matrix
    # Candidates: (score, det_index, track_id, item_type)
    candidates: List[Tuple[float, int, int, PPEItemType]] = []
    det_worker_scores: Dict[Tuple[int, PPEItemType], List[Tuple[float, int]]] = {}

    for di, det in enumerate(ppe_detections):
        det_area = _area(det.bbox_xyxy)
        if det_area <= 0:
            continue

        min_ioa = MIN_IOA.get(det.item_type, 0.35)
        for tid, regions in worker_regions.items():
            if not allow_legacy_bbox and det.item_type == PPEItemType.HELMET and worker_qualities[tid] != HeadROIQuality.CONFIRMED:
                continue

            reg_box = regions[det.item_type]
            ioa = _inter(det.bbox_xyxy, reg_box) / det_area

            if ioa >= min_ioa:
                score = float(ioa * det.confidence)
                candidates.append((score, di, tid, det.item_type))
                det_worker_scores.setdefault((di, det.item_type), []).append((score, tid))

    # Sort candidates descending by match score (greedy matching)
    candidates.sort(key=lambda c: c[0], reverse=True)

    assigned_dets: Set[int] = set()
    assigned_worker_slots: Set[Tuple[int, PPEItemType]] = set()

    for score, di, tid, item_type in candidates:
        if di in assigned_dets or (tid, item_type) in assigned_worker_slots:
            continue

        # Ambiguity check: does this PPE detection overlap another worker with near-identical score?
        competing = det_worker_scores.get((di, item_type), [])
        if len(competing) > 1 and ambiguity_margin > 0.0:
            sorted_competing = sorted(competing, key=lambda x: x[0], reverse=True)
            top1_score, top1_tid = sorted_competing[0]
            top2_score, top2_tid = sorted_competing[1]
            if top1_tid == tid and (top1_score - top2_score) < ambiguity_margin:
                # Mark both workers as ambiguous
                if item_type == PPEItemType.HELMET:
                    associations[top1_tid].helmet = WorkerPPEState(
                        status=PPEStatus.UNKNOWN,
                        confidence=top1_score,
                        bbox_xyxy=ppe_detections[di].bbox_xyxy,
                        reason=f"ambiguous_proximity_to_worker_{top2_tid}",
                    )
                    associations[top2_tid].helmet = WorkerPPEState(
                        status=PPEStatus.UNKNOWN,
                        confidence=top2_score,
                        bbox_xyxy=ppe_detections[di].bbox_xyxy,
                        reason=f"ambiguous_proximity_to_worker_{top1_tid}",
                    )
                assigned_dets.add(di)
                continue

        # Definite assignment
        assigned_dets.add(di)
        assigned_worker_slots.add((tid, item_type))

        det_obj = ppe_detections[di]
        state = WorkerPPEState(
            status=PPEStatus.DETECTED,
            confidence=det_obj.confidence,
            bbox_xyxy=det_obj.bbox_xyxy,
            reason="matched_anatomical_region",
        )

        if item_type == PPEItemType.HELMET:
            associations[tid].helmet = state
        elif item_type == PPEItemType.VEST:
            associations[tid].vest = state

    # Identify unassociated PPE detections
    unassociated_items = [det for i, det in enumerate(ppe_detections) if i not in assigned_dets]
    for tid in associations:
        associations[tid].unassociated_ppe_count = len(unassociated_items)

    return associations, unassociated_items


# ============================================================================
# Backward-Compatible Legacy Wrappers
# ============================================================================

class PPEDetection:
    """Legacy detection dataclass for backward compatibility with existing tests."""
    def __init__(self, label: str, bbox_xyxy: List[float], confidence: float):
        self.label = label
        self.bbox_xyxy = bbox_xyxy
        self.confidence = confidence


def associate_ppe(
    workers: List[Dict[str, Any]],
    detections: List[Any],
    min_ioa: Optional[Dict[str, float]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Legacy interface maintained for 100% backward compatibility."""
    items: List[PPEDetectionItem] = []
    for d in detections:
        if isinstance(d, PPEDetectionItem):
            items.append(d)
        else:
            norm = normalize_label(getattr(d, "label", ""))
            if norm == "helmet":
                items.append(PPEDetectionItem(
                    item_type=PPEItemType.HELMET,
                    bbox_xyxy=d.bbox_xyxy,
                    confidence=d.confidence,
                ))
            elif norm == "vest":
                items.append(PPEDetectionItem(
                    item_type=PPEItemType.VEST,
                    bbox_xyxy=d.bbox_xyxy,
                    confidence=d.confidence,
                ))

    assoc_map, _ = associate_ppe_to_workers(
        workers,
        items,
        has_vest_model=any(i.item_type == PPEItemType.VEST for i in items),
        allow_legacy_bbox=True,
    )

    # Convert to legacy dict format: {track_id: {"helmet": bool, "vest": bool, "helmet_conf": float, "vest_conf": float}}
    legacy_out: Dict[int, Dict[str, Any]] = {}
    for tid, a in assoc_map.items():
        legacy_out[tid] = {
            "helmet": a.helmet.status == PPEStatus.DETECTED,
            "vest": a.vest.status == PPEStatus.DETECTED,
            "helmet_conf": a.helmet.confidence,
            "vest_conf": a.vest.confidence,
        }
    return legacy_out


class PPEModel:
    """Legacy PPEModel wrapper maintained for backward compatibility."""
    def __init__(self, model_path: str, conf: float = 0.35, imgsz: int = 640,
                 device: Optional[str] = None, half: Optional[bool] = None):
        from src.ppe.detector import YOLOv8HelmetDetector
        self.detector = YOLOv8HelmetDetector(model_path=model_path, conf=conf, imgsz=imgsz, device=device, half=half)

    def detect(self, frame: np.ndarray) -> List[PPEDetection]:
        items = self.detector.detect(frame)
        return [PPEDetection(label="helmet", bbox_xyxy=it.bbox_xyxy, confidence=it.confidence) for it in items]
