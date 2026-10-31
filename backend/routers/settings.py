import json
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import SystemSettingModel, CameraDeviceModel
from backend.schemas import SystemSettingsDTO, CameraDeviceCreate, CameraDeviceResponse

router = APIRouter(prefix="", tags=["Settings & Cameras"])


@router.get("/settings", response_model=SystemSettingsDTO)
def get_system_settings(db: Session = Depends(get_db)):
    setting = db.query(SystemSettingModel).filter(SystemSettingModel.key == "general").first()
    if not setting:
        return SystemSettingsDTO()
    try:
        data = json.loads(setting.value_json)
        return SystemSettingsDTO(**data)
    except Exception:
        return SystemSettingsDTO()


@router.put("/settings", response_model=SystemSettingsDTO)
def update_system_settings(payload: SystemSettingsDTO, db: Session = Depends(get_db)):
    setting = db.query(SystemSettingModel).filter(SystemSettingModel.key == "general").first()
    json_val = payload.model_dump_json()

    if not setting:
        setting = SystemSettingModel(key="general", value_json=json_val)
        db.add(setting)
    else:
        setting.value_json = json_val

    db.commit()
    return payload


@router.get("/cameras", response_model=list[CameraDeviceResponse])
def list_cameras(db: Session = Depends(get_db)):
    cameras = db.query(CameraDeviceModel).all()
    return [CameraDeviceResponse(**c.to_dict()) for c in cameras]


@router.post("/cameras", response_model=CameraDeviceResponse, status_code=201)
def add_camera(payload: CameraDeviceCreate, db: Session = Depends(get_db)):
    count = db.query(CameraDeviceModel).count()
    cam_id = f"CAM-{count + 1:02d}"

    new_cam = CameraDeviceModel(
        id=cam_id,
        name=payload.name,
        zone=payload.zone,
        rtsp_url=payload.rtspUrl,
        target_fps=payload.targetFps,
        enabled=payload.enabled,
        status="online",
        last_seen="Just now",
    )
    db.add(new_cam)
    db.commit()
    db.refresh(new_cam)
    return CameraDeviceResponse(**new_cam.to_dict())


@router.delete("/cameras/{camera_id}")
def delete_camera(camera_id: str, db: Session = Depends(get_db)):
    cam = db.query(CameraDeviceModel).filter(CameraDeviceModel.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found.")
    db.delete(cam)
    db.commit()
    return {"message": f"Camera {camera_id} deleted successfully"}
