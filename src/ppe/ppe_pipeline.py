"""
src/ppe/ppe_pipeline.py — Integrated SafeOps PPE Detection & Association Pipeline.

Orchestrates:
  1. PersonDetector (finds human bodies)
  2. PPEHelmetAdapter (detects with_helmet / without_helmet)
  3. HelmetAssociator (pairs helmets to persons using spatial geometry)

Exposes high-level predict() and predict_and_annotate() APIs yielding
standardized JSON-serializable structured results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from src.ppe.helmet_adapter import PPEHelmetAdapter
from src.ppe.helmet_associator import HelmetAssociator
from src.ppe.person_detector import PersonDetector

logger = logging.getLogger(__name__)


class PPEPipeline:
    """
    End-to-end PPE Compliance Pipeline.

    Parameters
    ----------
    person_model_path : Union[str, Path]
        Path to pretrained person detector (e.g. 'yolov8n.pt').
    helmet_model_path : Union[str, Path]
        Path to trained helmet detector checkpoint (e.g. 'runs/detect/train/weights/best.pt').
    device : str
        Inference device ('cuda' or 'cpu').
    person_conf : float
        Confidence threshold for person detector (default 0.40).
    person_iou : float
        IoU threshold for person detector NMS (default 0.45).
    helmet_conf : Union[float, dict[str, float]]
        Confidence threshold(s) for helmet detector (default 0.25).
    helmet_iou : float
        IoU threshold for helmet detector NMS (default 0.45).
    input_size : int
        Inference image size (default 640).
    upper_body_ratio : float
        Fraction of person height for helmet search zone (default 0.45).
    headroom_ratio : float
        Allowed headroom above person box for helmet crowns (default 0.15).
    min_ioha : float
        Minimum Intersection-over-Helmet-Area (default 0.25).
    min_match_score : float
        Minimum association score to accept pairing (default 0.35).
    ambiguity_threshold : float
        Score difference below which pairings are marked ambiguous (default 0.08).
    """

    def __init__(
        self,
        person_model_path: Union[str, Path] = "yolov8n.pt",
        helmet_model_path: Union[str, Path] = "runs/detect/train/weights/best.pt",
        device: str = "cuda",
        person_conf: float = 0.40,
        person_iou: float = 0.45,
        helmet_conf: Union[float, dict[str, float]] = 0.25,
        helmet_iou: float = 0.45,
        input_size: int = 640,
        upper_body_ratio: float = 0.45,
        headroom_ratio: float = 0.15,
        min_ioha: float = 0.25,
        min_match_score: float = 0.35,
        ambiguity_threshold: float = 0.08,
    ) -> None:
        self.device = device

        logger.info(
            "[PPEPipeline] Initializing PersonDetector (model=%s, device=%s)",
            person_model_path,
            device,
        )
        self.person_detector = PersonDetector(
            model_path=person_model_path,
            device=device,
            confidence_threshold=person_conf,
            iou_threshold=person_iou,
            input_size=input_size,
        )

        logger.info(
            "[PPEPipeline] Initializing PPEHelmetAdapter (model=%s, device=%s)",
            helmet_model_path,
            device,
        )
        self.helmet_detector = PPEHelmetAdapter(
            model_path=helmet_model_path,
            device=device,
            confidence_threshold=helmet_conf,
            iou_threshold=helmet_iou,
        )

        self.associator = HelmetAssociator(
            upper_body_ratio=upper_body_ratio,
            headroom_ratio=headroom_ratio,
            min_ioha=min_ioha,
            min_match_score=min_match_score,
            ambiguity_threshold=ambiguity_threshold,
        )
        logger.info("[PPEPipeline] Initialized successfully.")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def predict(self, image: Union[str, Path, np.ndarray]) -> dict:
        """
        Run person detection, helmet detection, and association on *image*.

        Parameters
        ----------
        image : Union[str, Path, np.ndarray]
            Image path or OpenCV BGR NumPy array.

        Returns
        -------
        dict
            Structured result containing image identifier, persons list with
            associated helmet details, unmatched helmets, and summary metrics.
        """
        image_name = Path(image).name if isinstance(image, (str, Path)) else "frame"
        frame = self._load_image(image)

        # Stage 1: Person detection
        persons = self.person_detector.predict(frame)

        # Stage 2: Helmet detection
        helmets = self.helmet_detector.predict(frame)

        # Stage 3: Spatial association
        association_result = self.associator.associate(persons, helmets)

        return {
            "image": image_name,
            "persons": association_result.persons,
            "unmatched_helmets": association_result.unmatched_helmets,
            "summary": association_result.summary,
        }

    def predict_and_annotate(
        self, image: Union[str, Path, np.ndarray]
    ) -> tuple[dict, np.ndarray]:
        """
        Run the full pipeline and generate an annotated visualization.

        Returns
        -------
        result : dict
            Structured prediction dictionary.
        annotated_image : np.ndarray
            OpenCV BGR image with bounding boxes, labels, and status badges.
        """
        image_name = Path(image).name if isinstance(image, (str, Path)) else "frame"
        frame = self._load_image(image)
        result = self.predict(frame)
        result["image"] = image_name
        annotated_frame = self.annotate_frame(frame, result)
        return result, annotated_frame

    # -----------------------------------------------------------------------
    # Visualization helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        result: dict,
    ) -> np.ndarray:
        """
        Draw person detections, helmet detections, and association labels
        on a copy of the input frame.
        """
        annotated = frame.copy()

        # Colors (BGR)
        COLOR_COMPLIANT = (0, 200, 0)      # Green: with_helmet
        COLOR_VIOLATION = (0, 0, 220)      # Red: without_helmet
        COLOR_AMBIGUOUS = (0, 165, 255)    # Orange: ambiguous association
        COLOR_UNKNOWN   = (180, 180, 180)  # Gray: no helmet detected
        COLOR_UNMATCHED = (255, 200, 0)    # Cyan/Yellow: unmatched helmet

        # Draw persons
        for person in result.get("persons", []):
            px1, py1, px2, py2 = [int(v) for v in person["bbox"]]
            status = person.get("association_status", "no_helmet_detection")
            helmet = person.get("helmet")

            if status == "associated" and helmet:
                cname = helmet.get("class_name", "")
                if cname == "WITH_HELMET":
                    color = COLOR_COMPLIANT
                    label_text = f"P#{person['person_id']} [WITH HELMET {helmet['confidence']:.2f}]"
                elif cname == "WITHOUT_HELMET":
                    color = COLOR_VIOLATION
                    label_text = f"P#{person['person_id']} [NO HELMET {helmet['confidence']:.2f}]"
                else:
                    color = COLOR_UNKNOWN
                    label_text = f"P#{person['person_id']} [{cname}]"
            elif status == "ambiguous":
                color = COLOR_AMBIGUOUS
                label_text = f"P#{person['person_id']} [AMBIGUOUS]"
            else:
                color = COLOR_UNKNOWN
                label_text = f"P#{person['person_id']} [NO HELMET DETECTED]"

            # Person box
            cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 2)

            # Label banner
            (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (px1, max(0, py1 - lh - 6)),
                (px1 + lw + 6, py1),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label_text,
                (px1 + 3, max(lh + 2, py1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Draw associated helmet box
            if helmet:
                hx1, hy1, hx2, hy2 = [int(v) for v in helmet["bbox"]]
                cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), color, 2)

        # Draw unmatched helmets
        for u_helmet in result.get("unmatched_helmets", []):
            hx1, hy1, hx2, hy2 = [int(v) for v in u_helmet["bbox"]]
            cname = u_helmet.get("class_name", "HELMET")
            conf = u_helmet.get("confidence", 0.0)
            label = f"Unmatched: {cname} {conf:.2f}"

            cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), COLOR_UNMATCHED, 1, cv2.LINE_AA)
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(
                annotated,
                (hx1, max(0, hy1 - lh - 4)),
                (hx1 + lw + 4, hy1),
                COLOR_UNMATCHED,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (hx1 + 2, max(lh + 2, hy1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # Summary badge in top-left corner
        summary = result.get("summary", {})
        badge_lines = [
            f"Persons: {summary.get('persons_detected', 0)}",
            f"Associated Helmets: {summary.get('helmets_associated', 0)}",
            f"Unmatched Helmets: {summary.get('unmatched_helmets', 0)}",
            f"Ambiguous: {summary.get('ambiguous_associations', 0)}",
        ]
        y_offset = 20
        for line in badge_lines:
            cv2.putText(
                annotated,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y_offset += 20

        return annotated

    # -----------------------------------------------------------------------
    # Helper
    # -----------------------------------------------------------------------

    @staticmethod
    def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"[PPEPipeline] Image not found: {path}")
            img = cv2.imread(str(path))
            if img is None:
                raise RuntimeError(f"[PPEPipeline] cv2.imread failed for: {path}")
            return img
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"[PPEPipeline] NumPy image must be HxWx3 (BGR); got shape {image.shape}"
                )
            return image
        raise ValueError(
            f"[PPEPipeline] image must be a file path or np.ndarray; got {type(image).__name__!r}"
        )
