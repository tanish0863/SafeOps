import datetime
import json
from sqlalchemy import Column, String, Integer, Float, Boolean, Text, DateTime
from backend.database import Base


class WorkerModel(Base):
    __tablename__ = "workers"

    id = Column(String(50), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    zone = Column(String(100), nullable=False)
    shift = Column(String(10), default="A")
    last_scan = Column(String(50), default="Never")
    scans = Column(Integer, default=0)
    helmet = Column(Boolean, default=False)
    vest = Column(Boolean, default=False)
    status = Column(String(20), default="compliant")  # compliant | violation | review
    needs_review = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "zone": self.zone,
            "shift": self.shift,
            "lastScan": self.last_scan,
            "scans": self.scans,
            "helmet": self.helmet,
            "vest": self.vest,
            "status": self.status,
            "needsReview": self.needs_review,
        }


class ViolationModel(Base):
    __tablename__ = "violations"

    id = Column(String(50), primary_key=True, index=True)
    worker_id = Column(String(50), nullable=False)
    worker_name = Column(String(100), nullable=False)
    type = Column(String(100), nullable=False)
    zone = Column(String(100), nullable=False)
    timestamp = Column(String(50), nullable=False)
    severity = Column(String(20), default="medium")  # critical | high | medium | low
    review_status = Column(String(20), default="pending")  # confirmed | pending | resolved
    note = Column(Text, default="")
    evidence_crop_url = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "workerId": self.worker_id,
            "workerName": self.worker_name,
            "type": self.type,
            "zone": self.zone,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "reviewStatus": self.review_status,
            "note": self.note,
            "evidenceCropUrl": self.evidence_crop_url,
        }


class DetectionJobModel(Base):
    __tablename__ = "detection_jobs"

    id = Column(String(50), primary_key=True, index=True)
    media_id = Column(String(50), nullable=True)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=True)
    rules_json = Column(Text, default="{}")
    status = Column(String(20), default="complete")  # queued | processing | complete | failed
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    result_json = Column(Text, nullable=True)

    def get_result(self):
        if self.result_json:
            try:
                return json.loads(self.result_json)
            except Exception:
                return None
        return None


class MediaFileModel(Base):
    __tablename__ = "media_files"

    id = Column(String(50), primary_key=True, index=True)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class SystemSettingModel(Base):
    __tablename__ = "system_settings"

    key = Column(String(100), primary_key=True)
    value_json = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class CameraDeviceModel(Base):
    __tablename__ = "camera_devices"

    id = Column(String(50), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    zone = Column(String(100), nullable=False)
    rtsp_url = Column(String(255), nullable=False)
    target_fps = Column(Integer, default=30)
    enabled = Column(Boolean, default=True)
    status = Column(String(20), default="online")
    last_seen = Column(String(50), default="Now")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "zone": self.zone,
            "rtspUrl": self.rtsp_url,
            "targetFps": self.target_fps,
            "enabled": self.enabled,
            "status": self.status,
            "lastSeen": self.last_seen,
        }


class ActivityModel(Base):
    __tablename__ = "activity_logs"

    id = Column(String(50), primary_key=True, index=True)
    time_str = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    kind = Column(String(20), default="scan")  # scan | violation | review
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "time": self.time_str,
            "message": self.message,
            "kind": self.kind,
        }
