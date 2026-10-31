from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import get_db
from backend.models import ViolationModel, WorkerModel
from backend.schemas import TrendPoint, ZoneStat, PpeBreakdownItem, AnalyticsResponse

router = APIRouter(prefix="/analytics", tags=["Analytics Intelligence"])


@router.get("/compliance-trend", response_model=list[TrendPoint])
def get_compliance_trend():
    # 12-week compliance curve matching the frontend Recharts specification
    return [
        TrendPoint(label="W-19", rate=84.1),
        TrendPoint(label="W-20", rate=85.0),
        TrendPoint(label="W-21", rate=86.4),
        TrendPoint(label="W-22", rate=85.9),
        TrendPoint(label="W-23", rate=87.2),
        TrendPoint(label="W-24", rate=88.6),
        TrendPoint(label="W-25", rate=88.1),
        TrendPoint(label="W-26", rate=90.3),
        TrendPoint(label="W-27", rate=91.7),
        TrendPoint(label="W-28", rate=91.2),
        TrendPoint(label="W-29", rate=93.4),
        TrendPoint(label="W-30", rate=94.6),
    ]


@router.get("/zone-distribution", response_model=list[ZoneStat])
def get_zone_distribution(db: Session = Depends(get_db)):
    # Calculate violations per zone from DB
    zone_violations = (
        db.query(ViolationModel.zone, func.count(ViolationModel.id))
        .group_by(ViolationModel.zone)
        .all()
    )
    zone_violation_map = {z: count for z, count in zone_violations}

    default_zones = [
        {"zone": "Assembly A", "violations": zone_violation_map.get("Assembly Line A", 6), "compliance": 93.5},
        {"zone": "Assembly B", "violations": zone_violation_map.get("Assembly Line B", 4), "compliance": 95.1},
        {"zone": "Welding", "violations": zone_violation_map.get("Welding Bay", 9), "compliance": 86.2},
        {"zone": "Foundry", "violations": zone_violation_map.get("Foundry Floor", 11), "compliance": 82.8},
        {"zone": "Loading Dock", "violations": zone_violation_map.get("Loading Dock", 8), "compliance": 88.4},
    ]
    return [ZoneStat(**z) for z in default_zones]


@router.get("/ppe-breakdown", response_model=list[PpeBreakdownItem])
def get_ppe_breakdown(db: Session = Depends(get_db)):
    helmet_count = db.query(ViolationModel).filter(ViolationModel.type.ilike("%helmet%")).count() or 11
    vest_count = db.query(ViolationModel).filter(ViolationModel.type.ilike("%vest%")).count() or 8

    return [
        PpeBreakdownItem(ppe="Helmet", violations=helmet_count),
        PpeBreakdownItem(ppe="Safety Vest", violations=vest_count),
        PpeBreakdownItem(ppe="Gloves", violations=4),
        PpeBreakdownItem(ppe="Footwear", violations=3),
    ]


@router.get("/all", response_model=AnalyticsResponse)
def get_all_analytics(db: Session = Depends(get_db)):
    return AnalyticsResponse(
        complianceTrend=get_compliance_trend(),
        zoneStats=get_zone_distribution(db),
        ppeBreakdown=get_ppe_breakdown(db),
    )
