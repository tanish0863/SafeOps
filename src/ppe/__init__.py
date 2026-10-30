"""
src.ppe — SafeOps PPE Compliance Detection Package.

Modules:
    helmet_adapter     — Phase 1: Reusable YOLOv8 helmet detection adapter
    person_detector    — Phase 2: Reusable YOLOv8 person detection module
    helmet_associator  — Phase 2: Spatial geometry helmet-to-person matcher
    ppe_pipeline       — Phase 2: End-to-end detection and association pipeline
"""

from src.ppe.helmet_adapter import PPEHelmetAdapter
from src.ppe.person_detector import PersonDetector
from src.ppe.helmet_associator import HelmetAssociator, AssociationResult
from src.ppe.ppe_pipeline import PPEPipeline

__all__ = [
    "PPEHelmetAdapter",
    "PersonDetector",
    "HelmetAssociator",
    "AssociationResult",
    "PPEPipeline",
]
