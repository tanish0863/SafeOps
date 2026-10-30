"""
tests/test_helmet_associator.py — Tests for HelmetAssociator geometric matcher.

Runs with both `unittest` and `pytest`:
    python -m unittest tests/test_helmet_associator.py -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ppe.helmet_associator import HelmetAssociator


class TestHelmetAssociator(unittest.TestCase):
    def setUp(self):
        self.associator = HelmetAssociator(
            upper_body_ratio=0.45,
            headroom_ratio=0.15,
            min_ioha=0.25,
            min_match_score=0.35,
            ambiguity_threshold=0.08,
        )

    def test_helmet_fully_inside_upper_body(self):
        """Helmet located in upper 20% of person box should associate cleanly."""
        persons = [{"person_id": 0, "bbox": [100.0, 50.0, 200.0, 300.0], "confidence": 0.95}]
        # Person height is 250px. Head is roughly y=50 to y=100.
        helmet = [{"class_name": "WITH_HELMET", "confidence": 0.92, "bbox": [130.0, 55.0, 170.0, 95.0]}]

        result = self.associator.associate(persons, helmet)

        self.assertEqual(len(result.persons), 1)
        self.assertEqual(result.persons[0]["association_status"], "associated")
        self.assertIsNotNone(result.persons[0]["helmet"])
        self.assertEqual(result.persons[0]["helmet"]["class_name"], "WITH_HELMET")
        self.assertEqual(len(result.unmatched_helmets), 0)
        self.assertEqual(result.summary["helmets_associated"], 1)

    def test_helmet_centered_with_headroom(self):
        """Helmet top slightly extends above person detection (headroom)."""
        # Person from y=100 to y=400 (h=300). Headroom is 15% of 300 = 45px.
        persons = [{"person_id": 0, "bbox": [100.0, 100.0, 200.0, 400.0], "confidence": 0.90}]
        # Helmet from y=85 to y=140 (crosses person y1=100)
        helmet = [{"class_name": "WITH_HELMET", "confidence": 0.88, "bbox": [125.0, 85.0, 175.0, 140.0]}]

        result = self.associator.associate(persons, helmet)

        self.assertEqual(result.persons[0]["association_status"], "associated")
        self.assertEqual(result.summary["helmets_associated"], 1)

    def test_helmet_outside_all_person_boxes(self):
        """A helmet lying on a table far from any person remains unmatched."""
        persons = [{"person_id": 0, "bbox": [100.0, 100.0, 200.0, 400.0], "confidence": 0.90}]
        # Helmet far away at x=500
        helmet = [{"class_name": "WITH_HELMET", "confidence": 0.85, "bbox": [500.0, 200.0, 550.0, 250.0]}]

        result = self.associator.associate(persons, helmet)

        self.assertEqual(result.persons[0]["association_status"], "no_helmet_detection")
        self.assertIsNone(result.persons[0]["helmet"])
        self.assertEqual(len(result.unmatched_helmets), 1)
        self.assertEqual(result.summary["unmatched_helmets"], 1)
        self.assertEqual(result.summary["helmets_associated"], 0)

    def test_helmet_at_person_feet_is_not_associated(self):
        """A helmet near the feet (lower body) should NOT be associated as a head helmet."""
        persons = [{"person_id": 0, "bbox": [100.0, 100.0, 200.0, 400.0], "confidence": 0.90}]
        # Helmet placed at y=350 to y=390 (feet region of person)
        helmet = [{"class_name": "WITH_HELMET", "confidence": 0.85, "bbox": [120.0, 350.0, 160.0, 390.0]}]

        result = self.associator.associate(persons, helmet)

        self.assertEqual(result.persons[0]["association_status"], "no_helmet_detection")
        self.assertEqual(len(result.unmatched_helmets), 1)

    def test_two_people_with_two_distinct_helmets(self):
        """Two separated workers each get their own respective helmet."""
        persons = [
            {"person_id": 0, "bbox": [50.0, 50.0, 150.0, 300.0], "confidence": 0.92},
            {"person_id": 1, "bbox": [300.0, 60.0, 400.0, 310.0], "confidence": 0.89},
        ]
        helmets = [
            {"class_name": "WITH_HELMET", "confidence": 0.95, "bbox": [75.0, 55.0, 125.0, 105.0]},
            {"class_name": "WITHOUT_HELMET", "confidence": 0.87, "bbox": [325.0, 65.0, 375.0, 115.0]},
        ]

        result = self.associator.associate(persons, helmets)

        self.assertEqual(result.summary["persons_detected"], 2)
        self.assertEqual(result.summary["helmets_associated"], 2)
        self.assertEqual(result.persons[0]["helmet"]["class_name"], "WITH_HELMET")
        self.assertEqual(result.persons[1]["helmet"]["class_name"], "WITHOUT_HELMET")

    def test_one_helmet_cannot_be_assigned_to_two_people(self):
        """A single helmet must be assigned to at most one person (1-to-1)."""
        # Two people standing close
        persons = [
            {"person_id": 0, "bbox": [100.0, 50.0, 200.0, 300.0], "confidence": 0.92},
            {"person_id": 1, "bbox": [120.0, 50.0, 220.0, 300.0], "confidence": 0.90},
        ]
        # Only one helmet detection
        helmets = [
            {"class_name": "WITH_HELMET", "confidence": 0.94, "bbox": [130.0, 55.0, 170.0, 95.0]}
        ]

        result = self.associator.associate(persons, helmets)

        # Exactly one person gets the helmet; other has no helmet
        statuses = [p["association_status"] for p in result.persons]
        self.assertEqual(statuses.count("associated") + statuses.count("ambiguous"), 1)
        self.assertEqual(statuses.count("no_helmet_detection"), 1)
        self.assertEqual(result.summary["unmatched_helmets"], 0)

    def test_ambiguous_association_flagged(self):
        """Identically overlapping persons with equal proximity flag ambiguity."""
        persons = [
            {"person_id": 0, "bbox": [100.0, 50.0, 200.0, 300.0], "confidence": 0.90},
            {"person_id": 1, "bbox": [101.0, 50.0, 201.0, 300.0], "confidence": 0.90},
        ]
        helmets = [
            {"class_name": "WITH_HELMET", "confidence": 0.94, "bbox": [130.0, 55.0, 170.0, 95.0]}
        ]

        result = self.associator.associate(persons, helmets)
        # Scores are virtually identical -> should trigger ambiguous flag
        self.assertGreaterEqual(result.summary["ambiguous_associations"], 1)

    def test_empty_persons_returns_all_unmatched_helmets(self):
        helmets = [{"class_name": "WITH_HELMET", "confidence": 0.90, "bbox": [10.0, 10.0, 50.0, 50.0]}]
        result = self.associator.associate([], helmets)
        self.assertEqual(result.summary["persons_detected"], 0)
        self.assertEqual(result.summary["unmatched_helmets"], 1)

    def test_empty_helmets_returns_all_unassociated_persons(self):
        persons = [{"person_id": 0, "bbox": [10.0, 10.0, 100.0, 200.0], "confidence": 0.88}]
        result = self.associator.associate(persons, [])
        self.assertEqual(result.summary["helmets_associated"], 0)
        self.assertEqual(result.persons[0]["association_status"], "no_helmet_detection")

    def test_json_serializability(self):
        """The output structure must be cleanly JSON serializable."""
        persons = [{"person_id": 0, "bbox": [100.0, 50.0, 200.0, 300.0], "confidence": 0.95}]
        helmet = [{"class_name": "WITH_HELMET", "confidence": 0.92, "bbox": [130.0, 55.0, 170.0, 95.0]}]

        result = self.associator.associate(persons, helmet)
        output_payload = {
            "image": "test.jpg",
            "persons": result.persons,
            "unmatched_helmets": result.unmatched_helmets,
            "summary": result.summary,
        }
        serialized = json.dumps(output_payload)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized["summary"]["persons_detected"], 1)


if __name__ == "__main__":
    unittest.main()
