import json

import numpy as np
import pytest

from src.head_region import estimate_head_region
from src.inference import RawInference
from src.pipeline import Phase2Pipeline
from src.schemas import KEYPOINT_NAMES, FrameRecord


def kp(x, y, c=0.9):
    return {"x": x, "y": y, "confidence": c}


def empty():
    return {n: None for n in KEYPOINT_NAMES}


FRONTAL = empty()
FRONTAL.update(nose=kp(175, 65), left_eye=kp(160, 55), right_eye=kp(190, 55),
               left_ear=kp(150, 60), right_ear=kp(200, 60),
               left_shoulder=kp(130, 180), right_shoulder=kp(240, 180))


def test_visible_all_five():
    r = estimate_head_region(FRONTAL, 1280, 720, [100, 50, 300, 450])
    assert r["head_visibility"] == "VISIBLE" and r["head_region"] is not None


def test_partial_two_points():
    k = empty()
    k.update(nose=kp(175, 65), left_eye=kp(160, 55))
    assert estimate_head_region(k, 1280, 720)["head_visibility"] == "PARTIAL"


@pytest.mark.parametrize("names", [[], ["nose"]])
def test_uncertain_returns_null(names):
    k = empty()
    for n in names:
        k[n] = kp(100, 100)
    k.update(left_shoulder=kp(50, 200), right_shoulder=kp(150, 200))  # shoulders alone never suffice
    assert estimate_head_region(k, 640, 480) == {"head_region": None, "head_visibility": "UNCERTAIN"}


def test_head_clipped_to_frame():
    k = empty()
    k.update(left_eye=kp(5, 5), right_eye=kp(25, 5))
    x1, y1, x2, y2 = estimate_head_region(k, 640, 480)["head_region"]
    assert x1 >= 0 and y1 >= 0 and x2 <= 640 and y2 <= 480


class FakeEngine:
    device = "cpu"

    def infer(self, frame):
        boxes = np.array([[10, 10, 110, 300], [400, 20, 520, 320]], float)
        scores = np.array([0.9, 0.8])
        k = np.zeros((2, 17, 3))
        for i, (x0, y0) in enumerate([(60, 40), (460, 50)]):  # person 0 left, person 1 right
            k[i, 0] = [x0, y0 + 10, 0.95]
            k[i, 1] = [x0 - 10, y0, 0.9]
            k[i, 2] = [x0 + 10, y0, 0.9]
            k[i, 3] = [x0 - 22, y0 + 5, 0.2]  # low confidence -> must become None
        return RawInference(boxes, scores, k)


def test_pipeline_matching_and_contract():
    pipe = Phase2Pipeline(engine=FakeEngine())
    rec = pipe.process_frame(np.zeros((480, 640, 3), np.uint8), frame_number=7, timestamp_s=0.23)
    assert (rec.frame_width, rec.frame_height) == (640, 480)
    assert len(rec.persons) == 2
    p0, p1 = rec.persons
    assert p0.keypoints["nose"].x < 120 < p1.keypoints["nose"].x  # box/keypoint correspondence
    assert p0.bbox[0] < 120 < p1.bbox[0]
    for p in rec.persons:
        assert set(p.keypoints) == set(KEYPOINT_NAMES)            # all 17 represented
        assert p.keypoints["left_ear"] is None                    # low confidence rejected
        assert p.keypoints["left_ankle"] is None                  # missing -> null
        assert p.head_visibility == "PARTIAL" and p.head_region is not None
    FrameRecord.model_validate(json.loads(rec.model_dump_json()))


def test_no_people():
    class Empty:
        device = "cpu"

        def infer(self, f):
            return RawInference(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0, 17, 3)))

    rec = Phase2Pipeline(engine=Empty()).process_frame(np.zeros((100, 100, 3), np.uint8))
    assert rec.persons == []
