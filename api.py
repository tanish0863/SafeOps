from __future__ import annotations

import base64
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import cv2
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from quality_guardian.assessment import AssessmentConfig, assess_frame
from quality_guardian.report_generator import create_annotated_image


app = FastAPI(
    title="Camera Quality Guardian API",
    description="REST API for industrial camera quality assessment and PPE detection pre-flight checks.",
    version="1.0.0",
)

# Allow React dev server and local clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory application state
active_config = AssessmentConfig()
assessment_history: List[Dict[str, Any]] = []
assessment_cache: Dict[str, Any] = {}

SAMPLES_DIR = Path("samples")


def make_json_safe(value: Any) -> Any:
    """Convert dataclasses, enums and NumPy types into JSON-serialisable primitives."""
    if is_dataclass(value) and not isinstance(value, type):
        return make_json_safe(asdict(value))

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    import numpy as np
    if isinstance(value, np.generic):
        return value.item()

    return value


def _encode_bgr_to_base64_jpeg(image_bgr: Any) -> str:
    """Encode an OpenCV BGR image array into a base64 Data URL string."""
    success, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not success:
        return ""
    b64_str = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def _build_assessment_payload(
    image_bgr: Any,
    filename: str,
    source_type: str = "upload",
) -> Dict[str, Any]:
    """Execute quality evaluation and build complete response payload."""
    height, width = image_bgr.shape[:2]

    # Run core rule-based assessment engine
    result = assess_frame(image_bgr, active_config)

    # Generate annotated visualization
    try:
        annotated_bgr = create_annotated_image(image_bgr, result)
        annotated_b64 = _encode_bgr_to_base64_jpeg(annotated_bgr)
    except Exception:
        annotated_b64 = ""

    original_b64 = _encode_bgr_to_base64_jpeg(image_bgr)

    assessment_id = str(uuid.uuid4())
    processed_at = datetime.now().isoformat()

    # Determine PPE detection readiness verdict
    if result.overall_status == "GOOD":
        ppe_readiness = {
            "suitable": True,
            "badge": "SUITABLE",
            "message": "Camera meets all quality standards. Footage is suitable for reliable PPE detection.",
            "color": "#B6FF2E",
        }
    elif result.overall_status == "WARNING":
        ppe_readiness = {
            "suitable": False,
            "badge": "DEGRADED",
            "message": "Quality degradation detected. Downstream PPE detection accuracy may be impaired.",
            "color": "#FBBF24",
        }
    else:
        ppe_readiness = {
            "suitable": False,
            "badge": "UNSUITABLE",
            "message": "Severe quality issues detected. Camera footage is not suitable for PPE compliance checks.",
            "color": "#EF4444",
        }

    passed_count = sum(1 for c in result.checks.values() if c.passed) + (1 if result.obstruction.status.value == "CLEAR" else 0)
    uncertain_count = sum(1 for c in result.checks.values() if getattr(c, "uncertain", False)) + (1 if result.obstruction.status.value == "UNCERTAIN" else 0)
    failed_count = sum(1 for c in result.checks.values() if not c.passed and not getattr(c, "uncertain", False)) + (1 if result.obstruction.status.value == "OBSTRUCTED" else 0)

    checks_summary = {
        "passed": passed_count,
        "failed": failed_count,
        "uncertain": uncertain_count,
        "total": 5,
    }

    response_data = {
        "id": assessment_id,
        "processed_at": processed_at,
        "filename": filename,
        "source_type": source_type,
        "image_width": int(width),
        "image_height": int(height),
        "overall_score": result.overall_score,
        "overall_status": result.overall_status,
        "ppe_readiness": ppe_readiness,
        "checks_summary": checks_summary,
        "checks": result.checks,
        "obstruction": result.obstruction,
        "all_issues": result.all_issues,
        "all_recommendations": result.all_recommendations,
        "score_breakdown": result.score_breakdown,
        "disclaimer": result.disclaimer,
        "annotated_image": annotated_b64,
        "original_image": original_b64,
    }

    json_safe = make_json_safe(response_data)

    # Record to history (keep most recent 50)
    history_entry = {
        "id": assessment_id,
        "processed_at": processed_at,
        "filename": filename,
        "source_type": source_type,
        "width": int(width),
        "height": int(height),
        "overall_score": result.overall_score,
        "overall_status": result.overall_status,
        "issues_count": len(result.all_issues),
        "ppe_suitable": ppe_readiness["suitable"],
        "passed_count": passed_count,
        "failed_count": failed_count,
        "uncertain_count": uncertain_count,
    }
    assessment_cache[assessment_id] = json_safe
    assessment_history.insert(0, history_entry)
    if len(assessment_history) > 50:
        assessment_history.pop()

    return json_safe


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Camera Quality Guardian",
        "version": "1.0.0",
        "engine": "OpenCV Rule-Based CV",
    }


@app.post("/api/assess/image")
async def assess_image(file: UploadFile = File(...)):
    """Assess an uploaded image file."""
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    filename = file.filename or "uploaded_image.jpg"
    extension = filename.lower().rsplit(".", 1)

    if len(extension) != 2 or f".{extension[1]}" not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Please upload JPG, JPEG, or PNG.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    import numpy as np
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the image. The file may be corrupted.",
        )

    return _build_assessment_payload(image_bgr, filename, source_type="upload")


@app.get("/api/samples")
def list_samples():
    """List built-in sample images available for demonstration."""
    sample_definitions = [
        {
            "id": "normal.jpg",
            "name": "Standard Reference",
            "category": "Normal Lighting",
            "description": "Balanced lighting, clear geometry, and optimal focus. Expected result: GOOD (100).",
            "expected_status": "GOOD",
        },
        {
            "id": "dark.jpg",
            "name": "Underexposed Scene",
            "category": "Lighting Defect",
            "description": "Simulates night-time or poorly lit plant area. Darkness flagged without false obstruction.",
            "expected_status": "WARNING",
        },
        {
            "id": "blurry.jpg",
            "name": "Out-of-Focus Lens",
            "category": "Optical Defect",
            "description": "Simulates lens defocus or machinery vibration blur. Blur flagged without false obstruction.",
            "expected_status": "WARNING",
        },
        {
            "id": "bright.jpg",
            "name": "Overexposed / Glare",
            "category": "Sensor Saturation",
            "description": "Simulates direct spotlight or harsh sun reflection. Glare flagged without false obstruction.",
            "expected_status": "GOOD",
        },
    ]
    return {"samples": sample_definitions}


@app.get("/api/samples/{name}")
def get_sample_image(name: str):
    """Serve a sample image file directly."""
    file_path = SAMPLES_DIR / name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(file_path, media_type="image/jpeg")


@app.post("/api/assess/sample/{name}")
def assess_sample(name: str):
    """Assess a built-in sample image without requiring manual re-upload."""
    file_path = SAMPLES_DIR / name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Sample '{name}' not found.")

    image_bgr = cv2.imread(str(file_path))
    if image_bgr is None:
        raise HTTPException(status_code=500, detail=f"Failed to read sample '{name}'.")

    return _build_assessment_payload(image_bgr, name, source_type="sample")


@app.get("/api/history")
def get_history():
    """Return recent assessment records."""
    return {"history": assessment_history, "total": len(assessment_history)}


@app.delete("/api/history")
def clear_history():
    """Clear recorded assessment history."""
    assessment_history.clear()
    return {"message": "Assessment history cleared."}


@app.get("/api/config")
def get_config():
    """Return the current threshold and penalty configuration."""
    return {
        "darkness_mean_threshold": active_config.darkness_mean_threshold,
        "darkness_dark_pixel_fraction": active_config.darkness_dark_pixel_fraction,
        "darkness_dark_pixel_value": active_config.darkness_dark_pixel_value,
        "darkness_penalty": active_config.darkness_penalty,
        "blur_laplacian_variance_threshold": active_config.blur_laplacian_variance_threshold,
        "blur_penalty": active_config.blur_penalty,
        "resolution_min_width": active_config.resolution_min_width,
        "resolution_min_height": active_config.resolution_min_height,
        "resolution_penalty": active_config.resolution_penalty,
        "glare_bright_value_threshold": active_config.glare_bright_value_threshold,
        "glare_bright_saturation_threshold": active_config.glare_bright_saturation_threshold,
        "glare_bright_pixel_fraction_threshold": active_config.glare_bright_pixel_fraction_threshold,
        "glare_penalty": active_config.glare_penalty,
        "obstruction_edge_density_low": active_config.obstruction_edge_density_low,
        "obstruction_edge_density_uncertain": active_config.obstruction_edge_density_uncertain,
        "obstruction_std_low": active_config.obstruction_std_low,
        "obstruction_std_uncertain": active_config.obstruction_std_uncertain,
        "obstruction_penalty": active_config.obstruction_penalty,
        "score_good_threshold": active_config.score_good_threshold,
        "score_warning_threshold": active_config.score_warning_threshold,
    }


@app.post("/api/config")
def update_config(payload: Dict[str, Any] = Body(...)):
    """Update assessment thresholds in memory."""
    global active_config
    for key, value in payload.items():
        if hasattr(active_config, key):
            # Preserve type
            orig_type = type(getattr(active_config, key))
            setattr(active_config, key, orig_type(value))
    return {"status": "updated", "config": get_config()}


@app.post("/api/config/reset")
def reset_config():
    """Reset configuration to system default thresholds."""
    global active_config
    active_config = AssessmentConfig()
    return {"status": "reset", "config": get_config()}


@app.get("/api/cameras")
def get_monitored_cameras():
    """Return simulated live industrial camera feeds for the Camera Monitor view."""
    cameras = [
        {
            "id": "cam-01",
            "name": "Main Entrance Gate",
            "location": "North Access Road",
            "ip": "192.168.1.101",
            "status": "OPERATIONAL",
            "quality_status": "GOOD",
            "last_score": 96.0,
            "fps": 30.0,
            "resolution": "1920x1080",
            "stream_protocol": "RTSP / H.264",
            "sample_ref": "normal.jpg",
            "ppe_active": True,
        },
        {
            "id": "cam-02",
            "name": "Heavy Fabrication Bay 3",
            "location": "Sector C - Industrial Hall",
            "ip": "192.168.1.104",
            "status": "DEGRADED",
            "quality_status": "WARNING",
            "last_score": 75.0,
            "fps": 24.5,
            "resolution": "1280x720",
            "stream_protocol": "RTSP / H.264",
            "sample_ref": "blurry.jpg",
            "ppe_active": False,
            "active_alert": "Optical focus degraded - lens service suggested",
        },
        {
            "id": "cam-03",
            "name": "Chemical Storage Yard",
            "location": "Hazardous Zone 2",
            "ip": "192.168.1.112",
            "status": "WARNING",
            "quality_status": "WARNING",
            "last_score": 70.0,
            "fps": 28.0,
            "resolution": "1920x1080",
            "stream_protocol": "RTSP / H.265",
            "sample_ref": "dark.jpg",
            "ppe_active": False,
            "active_alert": "Insufficient ambient lighting - auxiliary spotlight needed",
        },
        {
            "id": "cam-04",
            "name": "Loading Dock Ramp 2",
            "location": "East Warehouse Logis",
            "ip": "192.168.1.118",
            "status": "OPERATIONAL",
            "quality_status": "GOOD",
            "last_score": 85.0,
            "fps": 30.0,
            "resolution": "1920x1080",
            "stream_protocol": "RTSP / H.264",
            "sample_ref": "bright.jpg",
            "ppe_active": True,
            "active_alert": "Minor glare from bay lights - operational",
        },
    ]
    return {"cameras": cameras}


@app.get("/api/assess/{assessment_id}")
def get_cached_assessment(assessment_id: str):
    """Retrieve full assessment by ID."""
    if assessment_id in assessment_cache:
        return assessment_cache[assessment_id]
    raise HTTPException(status_code=404, detail="Assessment not found.")
