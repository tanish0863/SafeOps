"""PPE association layer: assigns helmet/vest detections to tracked workers.

Keyed by track_id (never by detection order). Geometry only: no face or
appearance recognition. Each PPE detection goes to at most one worker, and
each worker receives at most one helmet and one vest (greedy, best score first).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# COCO keypoint indices
HEAD_KPS = (0, 1, 2, 3, 4)       # nose, eyes, ears
SHOULDERS = (5, 6)
HIPS = (11, 12)

HELMET_KEYWORDS = ("helmet", "hardhat", "hard hat", "hard-hat", "hard_hat")
VEST_KEYWORDS = ("vest",)
# Labels like "no_helmet", "NO-Hardhat", "without vest" must not count as PPE present.
_NEGATIVE = re.compile(r"^(no|non)[\s_\-]|without|missing|absent", re.I)

MIN_IOA = {"helmet": 0.5, "vest": 0.4}  # intersection / PPE-box area


@dataclass
class PPEDetection:
    label: str                    # "helmet" or "vest"
    bbox_xyxy: list[float]
    confidence: float


def normalize_label(name: str) -> str | None:
    """Map a raw class name from your PPE model to 'helmet' / 'vest' / None."""
    n = name.strip().lower()
    if _NEGATIVE.search(n):
        return None
    if any(k in n for k in HELMET_KEYWORDS):
        return "helmet"
    if any(k in n for k in VEST_KEYWORDS):
        return "vest"
    return None


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b) -> float:
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def _kp_box(worker, idxs, thresh, min_pts):
    kxy, kc = worker.get("keypoints_xy"), worker.get("keypoints_conf")
    if kxy is None or kc is None:
        return None
    pts = [kxy[i] for i in idxs if kc[i] >= thresh]
    if len(pts) < min_pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def worker_regions(worker: dict, kp_thresh: float = 0.5) -> dict[str, list[float]]:
    """Expected image regions for a helmet (head) and a vest (torso)."""
    x1, y1, x2, y2 = worker["bbox_xyxy"]
    h = y2 - y1

    head = _kp_box(worker, HEAD_KPS, kp_thresh, min_pts=2)
    if head:  # pad generously upward: helmets sit above the eyes
        pad = 0.07 * h
        head = [head[0] - pad, head[1] - 0.16 * h, head[2] + pad, head[3] + pad]
    else:
        head = [x1, y1, x2, y1 + 0.25 * h]

    torso = _kp_box(worker, SHOULDERS + HIPS, kp_thresh, min_pts=3)
    if torso:
        pad = 0.05 * h
        torso = [torso[0] - pad, torso[1] - pad, torso[2] + pad, torso[3] + pad]
    else:
        torso = [x1, y1 + 0.15 * h, x2, y1 + 0.65 * h]
    return {"helmet": head, "vest": torso}


def associate_ppe(
    workers: list[dict],
    detections: list[PPEDetection],
    min_ioa: dict[str, float] | None = None,
) -> dict[int, dict]:
    """Return {track_id: {"helmet": bool, "vest": bool, "helmet_conf", "vest_conf"}}."""
    min_ioa = {**MIN_IOA, **(min_ioa or {})}
    result = {
        w["track_id"]: {"helmet": False, "vest": False, "helmet_conf": None, "vest_conf": None}
        for w in workers
    }
    regions = {w["track_id"]: worker_regions(w) for w in workers}

    candidates = []
    for di, det in enumerate(detections):
        det_area = _area(det.bbox_xyxy)
        if det.label not in min_ioa or det_area <= 0:
            continue
        for tid, reg in regions.items():
            ioa = _inter(det.bbox_xyxy, reg[det.label]) / det_area
            if ioa >= min_ioa[det.label]:
                candidates.append((ioa * det.confidence, di, tid))

    used_dets, used_slots = set(), set()
    for score, di, tid in sorted(candidates, reverse=True):
        label = detections[di].label
        if di in used_dets or (tid, label) in used_slots:
            continue
        used_dets.add(di)
        used_slots.add((tid, label))
        result[tid][label] = True
        result[tid][f"{label}_conf"] = float(detections[di].confidence)
    return result


class PPEModel:
    """Thin wrapper around your custom YOLO PPE weights.

    Class names are mapped by keyword (see normalize_label). Check
    `model.model.names` on YOUR weights and adjust the keyword lists if needed.
    """

    def __init__(self, model_path: str, conf: float = 0.35, imgsz: int = 640,
                 device: str | None = None, half: bool | None = None):
        import torch
        from ultralytics import YOLO

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.half = str(self.device).startswith("cuda") if half is None else half
        self.model = YOLO(model_path)
        self.conf, self.imgsz = conf, imgsz
        self.label_map = {i: normalize_label(n) for i, n in self.model.names.items()}
        print("PPE classes:", {self.model.names[i]: l for i, l in self.label_map.items()})

    def detect(self, frame) -> list[PPEDetection]:
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                               device=self.device, half=self.half, verbose=False)[0]
        out = []
        if r.boxes is None:
            return out
        for box, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                             r.boxes.cls.int().cpu().numpy()):
            label = self.label_map.get(int(k))
            if label:
                out.append(PPEDetection(label, [float(v) for v in box], float(c)))
        return out
