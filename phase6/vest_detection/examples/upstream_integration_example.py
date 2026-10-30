"""
upstream_integration_example.py
-------------------------------
Integration example demonstrating how an upstream PPE Guard pipeline
(e.g., person detection / tracking / pose estimation) should invoke
the Safety Vest Detection module.

IMPORTANT ARCHITECTURAL BOUNDARIES:
- This script does NOT generate worker IDs or maintain tracking state.
- This script does NOT perform person detection or pose estimation.
- This script does NOT make overall worker compliance decisions (e.g., combining helmet and vest).
- The torso bounding box below is a clearly labelled demonstration input.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union
import numpy as np
import cv2

# Ensure vest_detector package is accessible
current_dir = Path(__file__).resolve().parent
package_src = current_dir.parent / "src"
if str(package_src) not in sys.path:
    sys.path.insert(0, str(package_src))

from vest_detector import VestDetector, TorsoVestAssessment, VestAssessmentState


def process_upstream_worker_torso(
    image: Union[np.ndarray, str, Path],
    torso_bbox: Sequence[Union[int, float]],
    detector: Optional[VestDetector] = None,
) -> Dict[str, Any]:
    """
    Simulates an upstream worker evaluation step.

    Accepts an image and a worker torso bounding box from an upstream caller,
    evaluates vest presence via VestDetector, logs the result, and returns
    the canonical assessment dictionary.

    Parameters
    ----------
    image : np.ndarray or file path
        Full-frame camera image (BGR format).
    torso_bbox : Sequence of 4 floats [x1, y1, x2, y2]
        Worker torso bounding box in original-image pixel space.
        (Upstream responsibility: determined by person detection or pose estimation).
    detector : Optional[VestDetector]
        Existing VestDetector instance to reuse across frames.

    Returns
    -------
    Dict[str, Any]
        Structured dictionary containing:
        - status: "PRESENT" | "ABSENT" | "UNKNOWN"
        - confidence: float (0.0 to 1.0)
        - vest_detected: bool
        - detections: list of vest bounding boxes in full-frame pixels
        - evidence: dictionary with overlap IoU, intersection ratio, decision reasons
    """
    if detector is None:
        model_path = current_dir.parent / "models" / "yolov8n_vest_best.pt"
        detector = VestDetector(model_path=model_path, device="cuda")

    # Call the three-state torso assessment engine
    assessment: TorsoVestAssessment = detector.assess_torso(
        image=image,
        torso_bbox=torso_bbox,
    )

    # Print assessment details for logging/auditing
    print("=" * 60)
    print("UPSTREAM WORKER VEST ASSESSMENT RESULT")
    print("=" * 60)
    print(f"Torso Input Box:  {list(torso_bbox)}")
    print(f"Assessment State: {assessment.status.value}")
    print(f"Confidence:       {assessment.confidence:.4f}")
    print(f"Vest Detected:    {assessment.vest_detected}")
    print(f"Overlap IoU:      {assessment.evidence.overlap_iou:.4f}")
    print(f"Quality Passed:   {assessment.evidence.quality_checks_passed}")
    print("Decision Reasons:")
    for reason in assessment.evidence.decision_reasons:
        print(f"  - {reason}")

    if assessment.detections:
        print("Associated Full-Frame Vest Bounding Boxes:")
        for det in assessment.detections:
            print(f"  * {det.class_name} ({det.confidence:.2f}): {det.bbox}")
    print("=" * 60)

    # Return pure assessment dictionary without worker IDs or overall compliance logic
    return assessment.to_dict()


def main():
    print("Initializing Safety Vest Detection integration example...")

    # Check for model weights
    model_path = current_dir.parent / "models" / "yolov8n_vest_best.pt"
    if not model_path.exists():
        print(f"Error: Pretrained model weights not found at {model_path}")
        return

    detector = VestDetector(model_path=model_path, device="cuda")
    print(f"VestDetector initialized on device: {detector.config.device}")

    # Select an evaluation image
    cctv_path = (
        current_dir.parent.parent.parent
        / "camera-quality-guardian"
        / "reports"
        / "annotated_construction_cctv_20261030_192942.jpg"
    )
    fallback_sample = current_dir.parent / "samples" / "worker_with_vest.jpg"

    if cctv_path.exists():
        test_image_path = cctv_path
        # DEMONSTRATION BOUNDING BOX:
        # NOTE: This is a clearly labelled demonstration bounding box matching
        # the torso region of a worker in the construction CCTV image.
        # It is provided here as simulated input from an upstream caller.
        demo_torso_bbox = [1357.0, 670.0, 1457.0, 810.0]
        image_label = "Genuine Construction CCTV Report Image"
    else:
        test_image_path = fallback_sample
        demo_torso_bbox = [150.0, 100.0, 490.0, 450.0]
        image_label = "Unit-Test Sample Image"

    print(f"\nProcessing test image ({image_label}):\n{test_image_path}")

    # Execute simulation
    result = process_upstream_worker_torso(
        image=test_image_path,
        torso_bbox=demo_torso_bbox,
        detector=detector,
    )

    print("\nIntegration execution completed successfully.")
    print(f"Output state returned to caller: {result['status']}")


if __name__ == "__main__":
    main()
