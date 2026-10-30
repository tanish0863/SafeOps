from src.association.ppe_association import (
    HeadROIQuality,
    PPEDetection,
    associate_ppe,
    associate_ppe_to_workers,
    get_worker_anatomical_regions,
    normalize_label,
)
from src.contracts.schemas import PPEDetectionItem, PPEItemType, PPEStatus


def worker(tid, x1):  # bbox-only worker (keypoint fallback regions)
    return {"track_id": tid, "bbox_xyxy": [x1, 100, x1 + 100, 400],
            "keypoints_xy": None, "keypoints_conf": None}


def make_worker_with_kps(tid, bbox, head_kps=True):
    x1, y1, x2, y2 = bbox
    kxy = [[0.0, 0.0] for _ in range(17)]
    kc = [0.0 for _ in range(17)]
    if head_kps:
        cx = (x1 + x2) / 2.0
        head_y = y1 + 30.0
        kxy[0] = [cx, head_y]          # nose
        kxy[1] = [cx - 10.0, head_y - 5.0]  # left eye
        kxy[2] = [cx + 10.0, head_y - 5.0]  # right eye
        kc[0] = 0.95
        kc[1] = 0.95
        kc[2] = 0.95
    return {
        "track_id": tid,
        "bbox_xyxy": bbox,
        "keypoints_xy": kxy if head_kps else None,
        "keypoints_conf": kc if head_kps else None,
    }


def test_two_workers_independent_of_order():
    ws = [worker(7, 0), worker(3, 200)]
    dets = [PPEDetection("helmet", [210, 100, 280, 150], 0.9),   # worker 3's head
            PPEDetection("vest", [10, 160, 90, 280], 0.8)]        # worker 7's torso
    r = associate_ppe(ws, dets)
    assert r[3]["helmet"] and not r[3]["vest"]
    assert r[7]["vest"] and not r[7]["helmet"]


def test_one_detection_goes_to_one_worker():
    ws = [worker(1, 0), worker(2, 20)]  # heavily overlapping
    dets = [PPEDetection("helmet", [30, 100, 90, 150], 0.9)]
    r = associate_ppe(ws, dets)
    assert sum(v["helmet"] for v in r.values()) == 1


def test_negative_labels_ignored():
    assert normalize_label("NO-Hardhat") is None
    assert normalize_label("no_vest") is None
    assert normalize_label("Hardhat") == "helmet"
    assert normalize_label("Safety Vest") == "vest"
    assert normalize_label("person") is None


# ============================================================================
# Phase 1 Critical Remediation Regression Tests (P0)
# ============================================================================

def test_worker_b_without_keypoints_cannot_steal_helmet_from_worker_a():
    """Worker A has visible head keypoints and helmet. Adjacent Worker B lacks facial keypoints.
    Worker B's fallback ROI must not steal Worker A's helmet.
    """
    w_a = make_worker_with_kps(1, [100.0, 100.0, 200.0, 400.0], head_kps=True)
    w_b = make_worker_with_kps(2, [140.0, 100.0, 240.0, 400.0], head_kps=False)
    helmet = PPEDetectionItem(
        item_type=PPEItemType.HELMET,
        bbox_xyxy=[130.0, 95.0, 170.0, 135.0],
        confidence=0.95,
    )

    r, unassigned = associate_ppe_to_workers([w_a, w_b], [helmet])

    assert r[1].helmet.status == PPEStatus.DETECTED
    assert r[1].head_roi_quality == HeadROIQuality.CONFIRMED

    # Worker B must NOT steal helmet, and must NOT be marked NOT_DETECTED
    assert r[2].helmet.status == PPEStatus.UNKNOWN
    assert r[2].helmet.reason == "missing_head_keypoints"
    assert r[2].head_roi_quality == HeadROIQuality.UNKNOWN
    assert len(unassigned) == 0


def test_both_workers_missing_facial_keypoints():
    """When both workers lack keypoints, fallback ROIs cannot establish compliance."""
    w1 = make_worker_with_kps(1, [100.0, 100.0, 200.0, 400.0], head_kps=False)
    w2 = make_worker_with_kps(2, [300.0, 100.0, 400.0, 400.0], head_kps=False)
    helmet = PPEDetectionItem(
        item_type=PPEItemType.HELMET,
        bbox_xyxy=[130.0, 95.0, 170.0, 135.0],
        confidence=0.95,
    )

    r, unassigned = associate_ppe_to_workers([w1, w2], [helmet])

    assert r[1].helmet.status == PPEStatus.UNKNOWN
    assert r[2].helmet.status == PPEStatus.UNKNOWN
    assert len(unassigned) == 1


def test_two_workers_overlap_and_compete_for_one_helmet():
    """When two confirmed workers overlap and compete for one helmet with close scores,
    both receive UNKNOWN rather than an arbitrary assignment.
    """
    w1 = make_worker_with_kps(1, [100.0, 100.0, 200.0, 400.0], head_kps=True)
    w2 = make_worker_with_kps(2, [105.0, 100.0, 205.0, 400.0], head_kps=True)
    helmet = PPEDetectionItem(
        item_type=PPEItemType.HELMET,
        bbox_xyxy=[135.0, 95.0, 175.0, 135.0],
        confidence=0.90,
    )

    r, unassigned = associate_ppe_to_workers([w1, w2], [helmet], ambiguity_margin=0.15)

    assert r[1].helmet.status == PPEStatus.UNKNOWN
    assert "ambiguous_proximity" in r[1].helmet.reason
    assert r[2].helmet.status == PPEStatus.UNKNOWN
    assert "ambiguous_proximity" in r[2].helmet.reason
    assert len(unassigned) == 0


def test_helmet_clearly_associated_with_one_worker():
    """When helmet clearly belongs to one worker, it is assigned correctly."""
    w1 = make_worker_with_kps(1, [100.0, 100.0, 200.0, 400.0], head_kps=True)
    w2 = make_worker_with_kps(2, [400.0, 100.0, 500.0, 400.0], head_kps=True)
    helmet = PPEDetectionItem(
        item_type=PPEItemType.HELMET,
        bbox_xyxy=[130.0, 95.0, 170.0, 135.0],
        confidence=0.95,
    )

    r, unassigned = associate_ppe_to_workers([w1, w2], [helmet])

    assert r[1].helmet.status == PPEStatus.DETECTED
    assert r[2].helmet.status == PPEStatus.NOT_DETECTED
    assert len(unassigned) == 0


def test_one_helmet_cannot_be_assigned_twice():
    """One helmet detection cannot be assigned to multiple workers."""
    w1 = make_worker_with_kps(1, [100.0, 100.0, 200.0, 400.0], head_kps=True)
    w2 = make_worker_with_kps(2, [130.0, 100.0, 230.0, 400.0], head_kps=True)
    helmet = PPEDetectionItem(
        item_type=PPEItemType.HELMET,
        bbox_xyxy=[140.0, 95.0, 180.0, 135.0],
        confidence=0.95,
    )

    r, _ = associate_ppe_to_workers([w1, w2], [helmet])
    detected_count = sum(1 for a in r.values() if a.helmet.status == PPEStatus.DETECTED)
    assert detected_count <= 1


def test_missing_keypoints_produce_unknown_rather_than_false_non_compliance():
    """Worker with missing keypoints must produce UNKNOWN, not NOT_DETECTED."""
    w1 = make_worker_with_kps(1, [100.0, 100.0, 200.0, 400.0], head_kps=False)
    r, _ = associate_ppe_to_workers([w1], [])

    assert r[1].helmet.status == PPEStatus.UNKNOWN
    assert r[1].helmet.status != PPEStatus.NOT_DETECTED

