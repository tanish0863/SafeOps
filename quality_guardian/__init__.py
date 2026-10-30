"""
Intelligent Camera Quality Guardian
------------------------------------
A standalone module for assessing camera and video quality
before trusting PPE detection results.

Modules:
    checks          - Individual quality check functions (blur, darkness, glare, resolution)
    obstruction     - Frame obstruction/blockage heuristics
    assessment      - Rule-based quality scoring engine
    image_processor - Single image analysis pipeline
    video_processor - Video / webcam analysis pipeline
    report_generator- JSON report and annotated image output
"""

__version__ = "1.0.0"
__author__ = "Camera Quality Guardian"
