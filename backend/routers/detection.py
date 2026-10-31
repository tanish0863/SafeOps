import json
import time
import datetime
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import DetectionJobModel, MediaFileModel
from backend.schemas import (
    DetectionJobCreate,
    DetectionJobResponse,
    WorkspaceAnalysis,
    WorkspaceWorker,
    NormalizedBox,
    PpeDetection,
    PpeRules,
)

router = APIRouter(prefix="/detection", tags=["Detection Workspace"])


def generate_structured_detection_result(job_id: str, file_name: str, rules: PpeRules) -> WorkspaceAnalysis:
    """
    ML INTEGRATION BOUNDARY:
    Returns the canonical WorkspaceAnalysis data structure with clearly identified
    isDemo=True / integration stub model tag, strictly respecting the PPE rules.
    When the separate ML service is connected, this function dispatches to the CV pipeline.
    """
    # 3 canonical workers matching the bundled demonstration and investigation canvas
    raw_workers = [
        {
            "workerId": "WORKER 01",
            "box": NormalizedBox(x=0.295, y=0.12, w=0.27, h=0.82),
            "personConfidence": 0.96,
            "helmet": PpeDetection(
                kind="helmet",
                status="detected",
                box=NormalizedBox(x=0.345, y=0.13, w=0.13, h=0.14),
                confidence=0.97,
                evaluated=rules.helmetRequired,
            ),
            "vest": PpeDetection(
                kind="vest",
                status="detected",
                box=NormalizedBox(x=0.325, y=0.36, w=0.21, h=0.40),
                confidence=0.94,
                evaluated=rules.vestRequired,
            ),
            "associationConfidence": 0.93,
            "reviewReason": None,
        },
        {
            "workerId": "WORKER 02",
            "box": NormalizedBox(x=0.615, y=0.20, w=0.17, h=0.66),
            "personConfidence": 0.91,
            "helmet": PpeDetection(
                kind="helmet",
                status="missing",
                box=NormalizedBox(x=0.655, y=0.20, w=0.09, h=0.10),
                confidence=0.88,
                evaluated=rules.helmetRequired,
            ),
            "vest": PpeDetection(
                kind="vest",
                status="detected",
                box=NormalizedBox(x=0.635, y=0.38, w=0.14, h=0.36),
                confidence=0.90,
                evaluated=rules.vestRequired,
            ),
            "associationConfidence": 0.90,
            "reviewReason": None,
        },
        {
            "workerId": "WORKER 03",
            "box": NormalizedBox(x=0.055, y=0.24, w=0.16, h=0.62),
            "personConfidence": 0.84,
            "helmet": PpeDetection(
                kind="helmet",
                status="uncertain",
                box=None,
                confidence=0.54,
                evaluated=rules.helmetRequired,
            ),
            "vest": PpeDetection(
                kind="vest",
                status="detected",
                box=NormalizedBox(x=0.075, y=0.40, w=0.12, h=0.34),
                confidence=0.86,
                evaluated=rules.vestRequired,
            ),
            "associationConfidence": 0.61,
            "reviewReason": "Helmet partially occluded — low-confidence optical match routed to review.",
        },
    ]

    evaluated_workers = []
    for w in raw_workers:
        helmet = w["helmet"]
        vest = w["vest"]
        evaluated_items = [item for item in [helmet, vest] if item.evaluated]

        if not evaluated_items:
            compliance = "review"
            review_reason = "No PPE rules enabled for this run — worker routed to review."
        else:
            statuses = [item.status for item in evaluated_items]
            if "missing" in statuses:
                compliance = "violation"
                review_reason = None
            elif "uncertain" in statuses:
                compliance = "review"
                review_reason = w["reviewReason"] or "Ambiguous PPE detection requires manual review."
            else:
                compliance = "compliant"
                review_reason = None

        evaluated_workers.append(
            WorkspaceWorker(
                workerId=w["workerId"],
                box=w["box"],
                personConfidence=w["personConfidence"],
                helmet=helmet,
                vest=vest,
                compliance=compliance,
                associationConfidence=w["associationConfidence"],
                reviewReason=review_reason,
            )
        )

    return WorkspaceAnalysis(
        jobId=job_id,
        fileName=file_name,
        analyzedAt=datetime.datetime.utcnow().isoformat() + "Z",
        model="SafeOps-Vision-v2.1 (Integration Stub · ML Ready)",
        isDemo=True,
        workers=evaluated_workers,
    )


@router.post("/jobs", response_model=DetectionJobResponse)
def create_detection_job(payload: DetectionJobCreate, db: Session = Depends(get_db)):
    job_id = f"job_{int(time.time() * 1000)}"
    file_name = payload.fileName or "detection_frame.jpg"
    file_path = None

    if payload.mediaId:
        media = db.query(MediaFileModel).filter(MediaFileModel.id == payload.mediaId).first()
        if media:
            file_name = media.file_name
            file_path = media.file_path

    # Compute structured detection result
    result = generate_structured_detection_result(job_id, file_name, payload.rules)

    db_job = DetectionJobModel(
        id=job_id,
        media_id=payload.mediaId,
        file_name=file_name,
        file_path=file_path,
        rules_json=payload.rules.model_dump_json(),
        status="complete",
        result_json=result.model_dump_json(),
        completed_at=datetime.datetime.utcnow(),
    )
    db.add(db_job)
    db.commit()

    return DetectionJobResponse(
        jobId=job_id,
        mediaId=payload.mediaId,
        status="complete",
        createdAt=datetime.datetime.utcnow().isoformat() + "Z",
    )


@router.get("/jobs/{job_id}/result", response_model=WorkspaceAnalysis)
def get_detection_result(job_id: str, db: Session = Depends(get_db)):
    job = db.query(DetectionJobModel).filter(DetectionJobModel.id == job_id).first()
    if not job:
        # Generate default response if job not yet in DB
        return generate_structured_detection_result(job_id, "inspection_frame.jpg", PpeRules())

    res = job.get_result()
    if res:
        return WorkspaceAnalysis(**res)

    return generate_structured_detection_result(job.id, job.file_name, PpeRules())


@router.post("/analyze", response_model=WorkspaceAnalysis)
def analyze_frame(payload: DetectionJobCreate, db: Session = Depends(get_db)):
    """
    Direct analysis endpoint used by runImageAnalysis() in the frontend.
    """
    job_res = create_detection_job(payload, db)
    return get_detection_result(job_res.jobId, db)


@router.get("/history")
def get_detection_history(limit: int = 15, db: Session = Depends(get_db)):
    jobs = db.query(DetectionJobModel).order_by(DetectionJobModel.created_at.desc()).limit(limit).all()
    return [
        {
            "jobId": j.id,
            "mediaId": j.media_id,
            "fileName": j.file_name,
            "status": j.status,
            "createdAt": j.created_at.isoformat() if j.created_at else None,
        }
        for j in jobs
    ]
