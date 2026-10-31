import datetime
import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import ViolationModel, ActivityModel
from backend.schemas import ViolationCreate, ViolationUpdate, ViolationResponse

router = APIRouter(prefix="/violations", tags=["Violations Log"])


@router.get("", response_model=list[ViolationResponse])
def list_violations(
    search: Optional[str] = None,
    severity: Optional[str] = None,
    reviewStatus: Optional[str] = None,
    db: Session = Depends(get_db),
):
    query = db.query(ViolationModel)

    if search:
        s = f"%{search.strip()}%"
        query = query.filter(
            (ViolationModel.id.ilike(s))
            | (ViolationModel.worker_id.ilike(s))
            | (ViolationModel.worker_name.ilike(s))
            | (ViolationModel.type.ilike(s))
            | (ViolationModel.zone.ilike(s))
        )

    if severity and severity != "all":
        query = query.filter(ViolationModel.severity == severity)

    if reviewStatus and reviewStatus != "all":
        query = query.filter(ViolationModel.review_status == reviewStatus)

    violations = query.order_by(ViolationModel.created_at.desc()).all()
    return [ViolationResponse(**v.to_dict()) for v in violations]


@router.get("/{violation_id}", response_model=ViolationResponse)
def get_violation(violation_id: str, db: Session = Depends(get_db)):
    v = db.query(ViolationModel).filter(ViolationModel.id == violation_id).first()
    if not v:
        raise HTTPException(status_code=404, detail=f"Violation {violation_id} not found.")
    return ViolationResponse(**v.to_dict())


@router.post("", response_model=ViolationResponse, status_code=status.HTTP_201_CREATED)
def create_violation(payload: ViolationCreate, db: Session = Depends(get_db)):
    v_id = payload.id
    if not v_id:
        count = db.query(ViolationModel).count()
        v_id = f"V-{1025 + count}"

    timestamp = payload.timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_v = ViolationModel(
        id=v_id,
        worker_id=payload.workerId,
        worker_name=payload.workerName,
        type=payload.type,
        zone=payload.zone,
        timestamp=timestamp,
        severity=payload.severity,
        review_status=payload.reviewStatus,
        note=payload.note,
        evidence_crop_url=payload.evidenceCropUrl,
    )
    db.add(new_v)

    # Log in activity feed
    activity = ActivityModel(
        id=f"A-{uuid.uuid4().hex[:8]}",
        time_str=datetime.datetime.now().strftime("%H:%M"),
        message=f"{payload.severity.capitalize()} violation raised — {payload.workerId} {payload.type} ({payload.zone})",
        kind="violation",
    )
    db.add(activity)
    db.commit()
    db.refresh(new_v)
    return ViolationResponse(**new_v.to_dict())


@router.patch("/{violation_id}", response_model=ViolationResponse)
def update_violation(violation_id: str, payload: ViolationUpdate, db: Session = Depends(get_db)):
    v = db.query(ViolationModel).filter(ViolationModel.id == violation_id).first()
    if not v:
        raise HTTPException(status_code=404, detail=f"Violation {violation_id} not found.")

    update_data = payload.model_dump(exclude_unset=True)
    if "reviewStatus" in update_data:
        v.review_status = update_data["reviewStatus"]
    if "severity" in update_data:
        v.severity = update_data["severity"]
    if "note" in update_data:
        v.note = update_data["note"]

    db.commit()
    db.refresh(v)
    return ViolationResponse(**v.to_dict())


@router.post("/{violation_id}/resolve", response_model=ViolationResponse)
def resolve_violation(violation_id: str, db: Session = Depends(get_db)):
    v = db.query(ViolationModel).filter(ViolationModel.id == violation_id).first()
    if not v:
        raise HTTPException(status_code=404, detail=f"Violation {violation_id} not found.")

    v.review_status = "resolved"
    v.note = (v.note or "") + f" [Resolved by supervisor on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}]"
    
    # Add activity
    act = ActivityModel(
        id=f"A-{uuid.uuid4().hex[:8]}",
        time_str=datetime.datetime.now().strftime("%H:%M"),
        message=f"Violation resolved — {v.id} {v.type} ({v.worker_name})",
        kind="review",
    )
    db.add(act)
    db.commit()
    db.refresh(v)
    return ViolationResponse(**v.to_dict())


@router.delete("/{violation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_violation(violation_id: str, db: Session = Depends(get_db)):
    v = db.query(ViolationModel).filter(ViolationModel.id == violation_id).first()
    if v:
        db.delete(v)
        db.commit()
    return None
