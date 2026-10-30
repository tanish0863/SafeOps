"""
src/ppe/helmet_associator.py — Spatial Geometry Helmet-to-Person Associator.

Associates helmet detections (from PPEHelmetAdapter) with person detections
(from PersonDetector) using geometric containment, upper-body proximity,
intersection-over-area (IoA), and horizontal alignment.

Design Principles:
  - Helmets are worn on the upper portion of a human body (head region).
  - Standard IoU is inadequate because a helmet is tiny (~5-15% of person area);
    we use Intersection-over-Helmet-Area (IoHA) and relative vertical position.
  - 1-to-1 assignment: One helmet cannot belong to multiple people; one person
    is assigned at most one primary helmet.
  - Ambiguity detection: When two people have nearly identical high match scores
    for a helmet in crowded/overlapping scenes, the association is flagged as
    'ambiguous' rather than arbitrarily guessed.
  - Unmatched helmets (e.g., helmets on tables, distant riders, or false positives)
    are returned in a separate list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class AssociationResult:
    """
    Result of associating a set of persons with detected helmets.
    """
    persons: list[dict]
    unmatched_helmets: list[dict]
    summary: dict[str, int]


class HelmetAssociator:
    """
    Deterministic geometric matcher for pairing person and helmet detections.

    Parameters
    ----------
    upper_body_ratio : float
        Maximum fraction of person bounding box height from top (y1) considered
        valid for helmet placement (default 0.45).
    headroom_ratio : float
        Allowed headroom above person box top (y1) as a fraction of person height
        to account for helmet crowns extending above person detection (default 0.15).
    min_ioha : float
        Minimum Intersection-over-Helmet-Area (IoHA) required for association
        (default 0.25).
    min_match_score : float
        Minimum composite match score in [0.0, 1.0] to accept an association
        (default 0.35).
    ambiguity_threshold : float
        If top two candidate persons for a helmet have scores within this margin,
        flag as ambiguous (default 0.08).
    """

    def __init__(
        self,
        upper_body_ratio: float = 0.45,
        headroom_ratio: float = 0.15,
        min_ioha: float = 0.25,
        min_match_score: float = 0.35,
        ambiguity_threshold: float = 0.08,
    ) -> None:
        self.upper_body_ratio = upper_body_ratio
        self.headroom_ratio = headroom_ratio
        self.min_ioha = min_ioha
        self.min_match_score = min_match_score
        self.ambiguity_threshold = ambiguity_threshold

        self._validate_config()

    def _validate_config(self) -> None:
        if not (0.0 < self.upper_body_ratio <= 1.0):
            raise ValueError(
                f"[HelmetAssociator] upper_body_ratio must be in (0.0, 1.0]; got {self.upper_body_ratio}"
            )
        if not (0.0 <= self.headroom_ratio <= 1.0):
            raise ValueError(
                f"[HelmetAssociator] headroom_ratio must be in [0.0, 1.0]; got {self.headroom_ratio}"
            )
        if not (0.0 <= self.min_ioha <= 1.0):
            raise ValueError(
                f"[HelmetAssociator] min_ioha must be in [0.0, 1.0]; got {self.min_ioha}"
            )
        if not (0.0 <= self.min_match_score <= 1.0):
            raise ValueError(
                f"[HelmetAssociator] min_match_score must be in [0.0, 1.0]; got {self.min_match_score}"
            )
        if not (0.0 <= self.ambiguity_threshold <= 1.0):
            raise ValueError(
                f"[HelmetAssociator] ambiguity_threshold must be in [0.0, 1.0]; got {self.ambiguity_threshold}"
            )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def associate(
        self,
        persons: list[dict],
        helmets: list[dict],
    ) -> AssociationResult:
        """
        Associate helmets to persons.

        Parameters
        ----------
        persons : list[dict]
            List of person detection dictionaries with keys ``person_id``,
            ``bbox`` ([x1, y1, x2, y2]), ``confidence``.
        helmets : list[dict]
            List of helmet detection dictionaries with keys ``class_name``,
            ``bbox`` ([x1, y1, x2, y2]), ``confidence``.

        Returns
        -------
        AssociationResult
            Contains ``persons`` (with added ``helmet`` and ``association_status``),
            ``unmatched_helmets``, and ``summary``.
        """
        if not persons:
            return AssociationResult(
                persons=[],
                unmatched_helmets=list(helmets),
                summary={
                    "persons_detected": 0,
                    "helmets_associated": 0,
                    "unmatched_helmets": len(helmets),
                    "ambiguous_associations": 0,
                },
            )

        if not helmets:
            unassociated_persons = [
                {
                    "person_id": p.get("person_id", idx),
                    "bbox": list(p["bbox"]),
                    "confidence": float(p["confidence"]),
                    "helmet": None,
                    "association_status": "no_helmet_detection",
                }
                for idx, p in enumerate(persons)
            ]
            return AssociationResult(
                persons=unassociated_persons,
                unmatched_helmets=[],
                summary={
                    "persons_detected": len(persons),
                    "helmets_associated": 0,
                    "unmatched_helmets": 0,
                    "ambiguous_associations": 0,
                },
            )

        # ── Compute pairwise scores between all (person, helmet) pairs ────
        num_persons = len(persons)
        num_helmets = len(helmets)

        # candidate_matches: list of (score, person_idx, helmet_idx)
        candidate_matches: list[tuple[float, int, int]] = []
        scores_matrix = [[0.0] * num_helmets for _ in range(num_persons)]

        for p_idx, person in enumerate(persons):
            for h_idx, helmet in enumerate(helmets):
                score = self._compute_match_score(person["bbox"], helmet["bbox"])
                scores_matrix[p_idx][h_idx] = score
                if score >= self.min_match_score:
                    candidate_matches.append((score, p_idx, h_idx))

        # Sort descending by score for greedy 1-to-1 matching
        candidate_matches.sort(key=lambda x: x[0], reverse=True)

        assigned_persons: dict[int, dict[str, Any]] = {}
        assigned_helmets: set[int] = set()
        ambiguous_count = 0

        # Check for ambiguity: if a helmet has two strong candidate persons
        # with scores very close to each other, flag as ambiguous.
        ambiguous_helmets: set[int] = set()
        for h_idx in range(num_helmets):
            h_scores = [scores_matrix[p_idx][h_idx] for p_idx in range(num_persons)]
            h_scores_sorted = sorted(h_scores, reverse=True)
            if len(h_scores_sorted) >= 2:
                top1, top2 = h_scores_sorted[0], h_scores_sorted[1]
                if (
                    top1 >= self.min_match_score
                    and top2 >= self.min_match_score
                    and (top1 - top2) < self.ambiguity_threshold
                ):
                    ambiguous_helmets.add(h_idx)

        # Match greedy 1-to-1
        for score, p_idx, h_idx in candidate_matches:
            if p_idx in assigned_persons or h_idx in assigned_helmets:
                continue

            helmet_obj = helmets[h_idx]

            if h_idx in ambiguous_helmets:
                # Flag as ambiguous association
                assigned_persons[p_idx] = {
                    "helmet": helmet_obj,
                    "status": "ambiguous",
                }
                assigned_helmets.add(h_idx)
                ambiguous_count += 1
            else:
                assigned_persons[p_idx] = {
                    "helmet": helmet_obj,
                    "status": "associated",
                }
                assigned_helmets.add(h_idx)

        # Assemble final persons list
        output_persons: list[dict] = []
        associated_count = 0

        for p_idx, person in enumerate(persons):
            p_id = person.get("person_id", p_idx)
            assignment = assigned_persons.get(p_idx)

            if assignment is not None:
                status = assignment["status"]
                helmet_data = assignment["helmet"]
                if status == "associated":
                    associated_count += 1
                output_persons.append(
                    {
                        "person_id": p_id,
                        "bbox": list(person["bbox"]),
                        "confidence": float(person["confidence"]),
                        "helmet": helmet_data,
                        "association_status": status,
                    }
                )
            else:
                output_persons.append(
                    {
                        "person_id": p_id,
                        "bbox": list(person["bbox"]),
                        "confidence": float(person["confidence"]),
                        "helmet": None,
                        "association_status": "no_helmet_detection",
                    }
                )

        # Unmatched helmets
        unmatched_helmets = [
            helmets[h_idx]
            for h_idx in range(num_helmets)
            if h_idx not in assigned_helmets
        ]

        summary = {
            "persons_detected": len(persons),
            "helmets_associated": associated_count,
            "unmatched_helmets": len(unmatched_helmets),
            "ambiguous_associations": ambiguous_count,
        }

        return AssociationResult(
            persons=output_persons,
            unmatched_helmets=unmatched_helmets,
            summary=summary,
        )

    # -----------------------------------------------------------------------
    # Internal geometric scoring
    # -----------------------------------------------------------------------

    def _compute_match_score(
        self,
        person_bbox: list[float],
        helmet_bbox: list[float],
    ) -> float:
        """
        Compute compatibility score in [0.0, 1.0] between a person and helmet.

        Parameters
        ----------
        person_bbox : [px1, py1, px2, py2]
        helmet_bbox : [hx1, hy1, hx2, hy2]
        """
        px1, py1, px2, py2 = person_bbox
        hx1, hy1, hx2, hy2 = helmet_bbox

        pw = max(1.0, px2 - px1)
        ph = max(1.0, py2 - py1)
        hw = max(1.0, hx2 - hx1)
        hh = max(1.0, hy2 - hy1)
        h_area = hw * hh

        # Helmet center
        hcx = (hx1 + hx2) / 2.0
        hcy = (hy1 + hy2) / 2.0

        # Person head region bounds
        head_top = py1 - (ph * self.headroom_ratio)
        head_bottom = py1 + (ph * self.upper_body_ratio)

        # 1. Vertical feasibility check:
        # If helmet center is completely below head_bottom or above head_top, score = 0.
        if hcy < head_top or hcy > head_bottom:
            return 0.0

        # 2. Horizontal feasibility check:
        # Helmet center should not be far outside person's horizontal span.
        h_margin = pw * 0.25
        if hcx < (px1 - h_margin) or hcx > (px2 + h_margin):
            return 0.0

        # 3. Intersection-over-Helmet-Area (IoHA)
        ix1 = max(px1, hx1)
        iy1 = max(py1 - (ph * self.headroom_ratio), hy1)
        ix2 = min(px2, hx2)
        iy2 = min(py2, hy2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection_area = iw * ih
        ioha = intersection_area / h_area

        if ioha < self.min_ioha:
            return 0.0

        # 4. Vertical alignment score:
        # Best position is around py1 to py1 + 0.15*ph.
        ideal_y = py1 + (ph * 0.08)
        norm_vert_dist = abs(hcy - ideal_y) / (ph * self.upper_body_ratio)
        vertical_score = max(0.0, 1.0 - norm_vert_dist)

        # 5. Horizontal alignment score:
        pcx = (px1 + px2) / 2.0
        norm_horiz_dist = abs(hcx - pcx) / (pw / 2.0 + h_margin)
        horizontal_score = max(0.0, 1.0 - norm_horiz_dist)

        # 6. Area reasonableness:
        # A helmet should be noticeably smaller than the person (e.g. area < 0.35 of person).
        p_area = pw * ph
        area_ratio = h_area / p_area
        if area_ratio > 0.45:
            # Impossibly large helmet compared to person body
            return 0.0

        # Composite weighted score
        composite_score = (
            0.40 * ioha
            + 0.35 * vertical_score
            + 0.25 * horizontal_score
        )

        return min(1.0, max(0.0, composite_score))
