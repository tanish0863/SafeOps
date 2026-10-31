import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import WorkerModel, ViolationModel, ActivityModel
from backend.schemas import OverviewStats, ActivityItem, TelemetryResponse, MetricDelta

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/stats", response_model=OverviewStats)
def get_dashboard_stats(db: Session = Depends(get_db)):
    total_workers = db.query(WorkerModel).count()
    compliant_workers = db.query(WorkerModel).filter(WorkerModel.status == "compliant").count()
    violation_workers = db.query(WorkerModel).filter(WorkerModel.status == "violation").count()
    review_workers = db.query(WorkerModel).filter(WorkerModel.status == "review").count()
    critical_violations = db.query(ViolationModel).filter(ViolationModel.severity == "critical").count()

    compliance_rate = round((compliant_workers / total_workers) * 100, 1) if total_workers > 0 else 100.0

    return OverviewStats(
        complianceRate=compliance_rate,
        totalWorkers=total_workers,
        compliantCount=compliant_workers,
        violationCount=violation_workers,
        reviewCount=review_workers,
        criticalCount=critical_violations,
        workersDetected=MetricDelta(value=str(total_workers), delta="Live registered"),
        compliant=MetricDelta(value=str(compliant_workers), delta=f"{compliance_rate}% compliance"),
        violations=MetricDelta(value=str(violation_workers), delta=f"{critical_violations} critical"),
        needsReview=MetricDelta(value=str(review_workers), delta="Requires check"),
    )


@router.get("/activity", response_model=list[ActivityItem])
def get_recent_activity(limit: int = 10, db: Session = Depends(get_db)):
    logs = db.query(ActivityModel).order_by(ActivityModel.created_at.desc()).limit(limit).all()
    return [ActivityItem(id=l.id, time=l.time_str, message=l.message, kind=l.kind) for l in logs]


@router.get("/telemetry", response_model=TelemetryResponse)
def get_edge_telemetry():
    return TelemetryResponse(
        activeSensors=8,
        totalSensors=8,
        fps=30.4,
        latencyMs=38.2,
        trackedPersonnel=148,
        cudaActive=True,
        pipelineVersion="v2.4 (Edge-Optimized)",
        activePlant="Facility Unit 01",
    )


@router.post("/simulate-scan")
def simulate_sensor_scan(db: Session = Depends(get_db)):
    # Increment scans for compliant workers
    workers = db.query(WorkerModel).all()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    for w in workers:
        w.scans += 1
        w.last_scan = now_str
    
    # Add an activity log entry
    new_act = ActivityModel(
        id=f"A-{int(datetime.datetime.now().timestamp()) % 1000:03d}",
        time_str=datetime.datetime.now().strftime("%H:%M"),
        message=f"Sensor frame ingested — {len(workers)} worker vectors verified",
        kind="scan"
    )
    db.add(new_act)
    db.commit()
    return {"message": "Sensor scan simulated successfully", "timestamp": now_str}
