import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import WorkerModel
from backend.schemas import WorkerCreate, WorkerUpdate, WorkerResponse

router = APIRouter(prefix="/workers", tags=["Workers Registry"])


@router.get("", response_model=list[WorkerResponse])
def list_workers(
    search: Optional[str] = None,
    status: Optional[str] = None,
    shift: Optional[str] = None,
    db: Session = Depends(get_db),
):
    query = db.query(WorkerModel)

    if search:
        s = f"%{search.strip()}%"
        query = query.filter(
            (WorkerModel.id.ilike(s)) | (WorkerModel.name.ilike(s)) | (WorkerModel.zone.ilike(s))
        )

    if status and status != "all":
        query = query.filter(WorkerModel.status == status)

    if shift and shift in ("A", "B", "C"):
        query = query.filter(WorkerModel.shift == shift)

    workers = query.order_by(WorkerModel.created_at.desc()).all()
    return [WorkerResponse(**w.to_dict()) for w in workers]


@router.post("", response_model=WorkerResponse, status_code=status.HTTP_201_CREATED)
def create_worker(payload: WorkerCreate, db: Session = Depends(get_db)):
    # Generate automatic ID if not provided
    worker_id = payload.id
    if not worker_id:
        count = db.query(WorkerModel).count()
        worker_id = f"W-{count + 101:03d}"
        while db.query(WorkerModel).filter(WorkerModel.id == worker_id).first():
            count += 1
            worker_id = f"W-{count + 101:03d}"

    existing = db.query(WorkerModel).filter(WorkerModel.id == worker_id).first()
    if existing:
        existing.scans += 1
        existing.last_scan = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        existing.helmet = payload.helmet
        existing.vest = payload.vest
        existing.status = payload.status
        existing.needs_review = payload.needsReview
        if payload.name and payload.name != "Worker":
            existing.name = payload.name
        if payload.zone:
            existing.zone = payload.zone
        db.commit()
        db.refresh(existing)
        return WorkerResponse(**existing.to_dict())

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    new_worker = WorkerModel(
        id=worker_id,
        name=payload.name or f"Worker {worker_id}",
        zone=payload.zone or "Vision Intake Corridor",
        shift=payload.shift or "A",
        last_scan=now_str,
        scans=1,
        helmet=payload.helmet,
        vest=payload.vest,
        status=payload.status,
        needs_review=payload.needsReview,
    )
    db.add(new_worker)
    db.commit()
    db.refresh(new_worker)
    return WorkerResponse(**new_worker.to_dict())


@router.get("/{worker_id}", response_model=WorkerResponse)
def get_worker(worker_id: str, db: Session = Depends(get_db)):
    worker = db.query(WorkerModel).filter(WorkerModel.id == worker_id).first()
    if not worker:
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found.")
    return WorkerResponse(**worker.to_dict())


@router.patch("/{worker_id}", response_model=WorkerResponse)
def update_worker(worker_id: str, payload: WorkerUpdate, db: Session = Depends(get_db)):
    worker = db.query(WorkerModel).filter(WorkerModel.id == worker_id).first()
    if not worker:
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found.")

    update_data = payload.model_dump(exclude_unset=True)
    if "name" in update_data:
        worker.name = update_data["name"]
    if "zone" in update_data:
        worker.zone = update_data["zone"]
    if "shift" in update_data:
        worker.shift = update_data["shift"]
    if "helmet" in update_data:
        worker.helmet = update_data["helmet"]
    if "vest" in update_data:
        worker.vest = update_data["vest"]
    if "status" in update_data:
        worker.status = update_data["status"]
    if "needsReview" in update_data:
        worker.needs_review = update_data["needsReview"]
    if "lastScan" in update_data:
        worker.last_scan = update_data["lastScan"]
    if "scans" in update_data:
        worker.scans = update_data["scans"]

    db.commit()
    db.refresh(worker)
    return WorkerResponse(**worker.to_dict())


@router.delete("/{worker_id}")
def delete_worker(worker_id: str, db: Session = Depends(get_db)):
    worker = db.query(WorkerModel).filter(WorkerModel.id == worker_id).first()
    if not worker:
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found.")
    db.delete(worker)
    db.commit()
    return {"message": f"Worker {worker_id} removed successfully", "id": worker_id}


@router.delete("")
def clear_all_workers(db: Session = Depends(get_db)):
    db.query(WorkerModel).delete()
    db.commit()
    return {"message": "All workers cleared successfully"}
