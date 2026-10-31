"""
Initial database seeder. Populates baseline SafeOps tables if empty.
"""

import json
from sqlalchemy.orm import Session
from backend.models import (
    WorkerModel,
    ViolationModel,
    ActivityModel,
    CameraDeviceModel,
    SystemSettingModel,
)


def seed_database(db: Session):
    # Workers, violations, and activity events start empty so only real data added by users is present.
    # No mock workers or mock violations are seeded.


    # 4. Seed Camera Devices
    if db.query(CameraDeviceModel).count() == 0:
        initial_cameras = [
            {"id": "CAM-01", "name": "Assembly Line A Primary", "zone": "Assembly Line A", "rtsp_url": "rtsp://10.10.12.101:554/ch01", "target_fps": 30, "enabled": True, "status": "online", "last_seen": "10s ago"},
            {"id": "CAM-02", "name": "Assembly Line B Overhead", "zone": "Assembly Line B", "rtsp_url": "rtsp://10.10.12.102:554/ch01", "target_fps": 30, "enabled": True, "status": "online", "last_seen": "8s ago"},
            {"id": "CAM-03", "name": "Welding Station South", "zone": "Welding Bay", "rtsp_url": "rtsp://10.10.12.103:554/ch01", "target_fps": 30, "enabled": True, "status": "online", "last_seen": "12s ago"},
            {"id": "CAM-04", "name": "Foundry Furnace Access", "zone": "Foundry Floor", "rtsp_url": "rtsp://10.10.12.104:554/ch01", "target_fps": 30, "enabled": True, "status": "online", "last_seen": "5s ago"},
            {"id": "CAM-05", "name": "Loading Dock Gate 02", "zone": "Loading Dock", "rtsp_url": "rtsp://10.10.12.105:554/ch01", "target_fps": 30, "enabled": True, "status": "online", "last_seen": "15s ago"},
        ]
        for c in initial_cameras:
            db.add(CameraDeviceModel(**c))
        db.commit()

    # 5. Seed System Settings
    if db.query(SystemSettingModel).count() == 0:
        default_settings = {
            "personConfidenceThreshold": 0.50,
            "ppeConfidenceThreshold": 0.40,
            "torsoOverlapThreshold": 0.30,
            "alertCriticalViolations": True,
            "alertReviewRequired": False,
            "enableAudioAlerts": True,
            "retentionDays": 30,
            "webhookUrl": "https://hooks.safeops.local/alerts",
            "notificationEmail": "safety-ops@safeops.local",
        }
        db.add(SystemSettingModel(key="general", value_json=json.dumps(default_settings)))
        db.commit()
