import cv2
import numpy as np
import base64
import json
import asyncio
import importlib.util
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import socket
import uvicorn
import logging
import tempfile
from collections import deque
import csv
import io
import mimetypes
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO)
from fastapi.responses import HTMLResponse, Response, FileResponse
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from input_handler import VideoSource
logger = logging.getLogger(__name__)


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UI_DIR = Path(__file__).resolve().parent
UI_HTML = UI_DIR / "camera ui 2.0.html"
ROOT_DIR = Path(__file__).resolve().parents[1]
MEDIA_DIR = ROOT_DIR / "videos"


@app.get("/", response_class=HTMLResponse)
def serve_ui() -> HTMLResponse:
    if not UI_HTML.exists():
        return HTMLResponse("<h1>UI file not found</h1>", status_code=404)
    return HTMLResponse(UI_HTML.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
def api_status() -> dict:
    return _dashboard_payload()


@app.get("/api/dashboard")
def api_dashboard() -> dict:
    return _dashboard_payload()


@app.get("/api/report")
def api_report() -> dict:
    return _dashboard_payload()


@app.get("/api/report-csv")
def api_report_csv() -> Response:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["frame_number", "status", "compliance", "total_persons", "alerts"],
    )
    writer.writeheader()
    for row in timeline_buffer:
        writer.writerow({
            "frame_number": row.get("frame_number", ""),
            "status": row.get("status", ""),
            "compliance": row.get("compliance", ""),
            "total_persons": row.get("total_persons", ""),
            "alerts": len(row.get("alerts", []) or []),
        })

    filename = "safeguard-report.csv"
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/media/{filename}")
def serve_media(filename: str):
    safe_name = Path(filename).name
    media_path = (MEDIA_DIR / safe_name).resolve()
    if MEDIA_DIR.resolve() not in media_path.parents and media_path != MEDIA_DIR.resolve():
        return Response(content="Invalid media path", status_code=400)
    if not media_path.exists() or not media_path.is_file():
        return Response(content="Media file not found", status_code=404)
    media_type = mimetypes.guess_type(str(media_path))[0] or "application/octet-stream"
    return FileResponse(str(media_path), media_type=media_type, filename=safe_name)


def find_free_port(start_port: int) -> int:
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                port += 1


def _load_root_pipeline(source="0", profile="accuracy"):
    spec = importlib.util.spec_from_file_location("_root_safety_main", ROOT_DIR / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load root safety pipeline")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SafetyPipeline(source=source, display=False, headless=True, profile=profile)


def _format_detection(track) -> dict:
    x1, y1, x2, y2 = [int(v) for v in track.bbox_xyxy]
    label = "person"
    if track.has_helmet is True:
        ppe = "helmet"
    elif track.has_helmet is False:
        ppe = "no_helmet"
    else:
        ppe = "unknown"
    if track.in_restricted_zone:
        zone = "zone_breach"
    else:
        zone = "clear"
    return {
        "track_id": track.track_id,
        "label": label,
        "ppe": ppe,
        "zone": zone,
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "confidence": float(track.confidence),
        "ppe_confidence": float(track.ppe_confidence or 0.0),
        "detection_state": getattr(track, "detection_state", "SAFE"),
    }


def _summarize_tracks(tracks, alerts, frame_number: int, source: str) -> dict:
    total_persons = len(tracks)
    helmet_count = sum(1 for t in tracks if t.has_helmet is True)
    no_helmet_count = sum(1 for t in tracks if t.has_helmet is False)
    unknown_count = sum(1 for t in tracks if t.has_helmet is None)
    uncertain_count = sum(1 for t in tracks if getattr(t, "detection_state", "SAFE") == "UNCERTAIN")
    zone_breaches = sum(1 for t in tracks if t.in_restricted_zone)
    known_count = helmet_count + no_helmet_count
    compliance = int(round((helmet_count / known_count) * 100)) if known_count else 0

    if total_persons == 0:
        status = "NO PERSON"
    elif zone_breaches > 0 and no_helmet_count > 0:
        status = "NO HELMET + ZONE"
    elif zone_breaches > 0:
        status = "ZONE BREACH"
    elif no_helmet_count > 0:
        status = "NO HELMET"
    elif helmet_count > 0:
        status = "HELMET DETECTED"
    else:
        status = "UNKNOWN"

    summary = {
        "total_persons": total_persons,
        "helmet_count": helmet_count,
        "no_helmet_count": no_helmet_count,
        "unknown_count": unknown_count,
        "uncertain_count": uncertain_count,
        "known_count": known_count,
        "zone_breaches": zone_breaches,
        "compliance": compliance,
        "status": status,
        "frame_number": frame_number,
        "source": source,
    }
    return summary


def _serialize_alert(alert) -> dict:
    return {
        "track_id": alert.track_id,
        "violation_type": alert.violation_type.name,
        "zone_ids": alert.zone_ids,
        "frame_number": alert.frame_number,
        "message": str(alert),
        "timestamp": alert.timestamp,
        "severity": "critical",
    }


def _most_common_int(values, default=0) -> int:
    filtered = [int(v) for v in values if v is not None]
    if not filtered:
        return default
    return Counter(filtered).most_common(1)[0][0]


def _range_label(values, default="0") -> str:
    filtered = [int(v) for v in values if v is not None]
    if not filtered:
        return default
    low = min(filtered)
    high = max(filtered)
    return str(low) if low == high else f"{low}-{high}"


def _status_rank(status: str) -> int:
    order = {
        "NO HELMET + ZONE": 5,
        "ZONE BREACH": 4,
        "NO HELMET": 3,
        "HELMET DETECTED": 2,
        "NO PERSON": 1,
        "UNKNOWN": 0,
    }
    return order.get(status, 0)


try:
    safety_pipeline = _load_root_pipeline(source="0", profile="accuracy")
    logger.info("[UI] Root safety pipeline loaded successfully.")
except Exception as exc:
    logger.exception("[UI] Failed to load root safety pipeline: %s", exc)
    safety_pipeline = None

status_buffer = deque(maxlen=1)
timeline_buffer = deque(maxlen=60)
alert_buffer = deque(maxlen=25)
AVAILABLE_CAMERAS: list[int] | None = None
latest_summary = {
    "total_persons": 0,
    "helmet_count": 0,
    "no_helmet_count": 0,
    "unknown_count": 0,
    "uncertain_count": 0,
    "known_count": 0,
    "zone_breaches": 0,
    "compliance": 0,
    "status": "UNKNOWN",
    "frame_number": 0,
    "source": "idle",
}
latest_detections: list[dict] = []


def get_stable_status(new_status):
    status_buffer.clear()
    status_buffer.append(new_status)
    return new_status


def _process_frame_with_pipeline(frame, frame_number: int, source: str) -> tuple[list[dict], dict, list[dict]]:
    if safety_pipeline is None:
        raise RuntimeError("Safety pipeline is not available")

    _, tracks, alerts = safety_pipeline.process_single_frame(frame, frame_number)
    detections = [_format_detection(track) for track in tracks]
    summary = _summarize_tracks(tracks, alerts, frame_number, source)
    summary["frame_width"] = int(frame.shape[1])
    summary["frame_height"] = int(frame.shape[0])
    summary["status"] = get_stable_status(summary["status"])
    if alerts:
        for alert in alerts:
            alert_data = _serialize_alert(alert)
            alert_buffer.appendleft(alert_data)
    return detections, summary, [_serialize_alert(a) for a in alerts]


def _dashboard_payload() -> dict:
    global AVAILABLE_CAMERAS
    zones = []
    if safety_pipeline is not None and safety_pipeline.zone_checker is not None:
        zones = [
            {
                "id": zone.zone_id,
                "name": zone.name,
                "points": zone.points.tolist(),
            }
            for zone in safety_pipeline.zone_checker.zones
        ]

    if AVAILABLE_CAMERAS is None:
        AVAILABLE_CAMERAS = VideoSource.list_webcams()

    return {
        "status": latest_summary.get("status", "UNKNOWN"),
        "summary": latest_summary,
        "history": list(timeline_buffer),
        "recent_alerts": list(alert_buffer)[:10],
        "available_cameras": AVAILABLE_CAMERAS,
        "model_info": {
            "person_model": config.PERSON_MODEL_PATH,
            "ppe_model": config.PPE_MODEL_PATH,
        },
        "zones": zones,
        "backend": {
            "source_mode": latest_summary.get("source", "idle"),
            "frames_processed": latest_summary.get("frame_number", 0),
        },
    }


def analyze_video_file(video_path: str, frame_stride: int = 1, max_sampled_frames: int = 300):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Failed to open uploaded video")

    upload_pipeline = _load_root_pipeline(source=video_path, profile="accuracy")

    sampled_frames = 0
    frame_index = 0
    best_detections: list[dict] = []
    best_person_count = -1
    best_frame_number = 0
    best_frame_width = 0
    best_frame_height = 0
    timeline_buffer.clear()

    aggregated = {
        "total_persons": 0,
        "helmet_count": 0,
        "no_helmet_count": 0,
        "unknown_count": 0,
        "uncertain_count": 0,
        "known_count": 0,
        "zone_breaches": 0,
        "compliance": 0,
        "status": "UNKNOWN",
        "source": "upload",
    }
    person_counts: list[int] = []
    helmet_counts: list[int] = []
    no_helmet_counts: list[int] = []
    unknown_counts: list[int] = []
    uncertain_counts: list[int] = []
    known_counts: list[int] = []
    zone_breach_counts: list[int] = []
    compliance_scores: list[int] = []
    statuses: list[str] = []

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            if frame_index % frame_stride == 0:
                detections, frame_summary, alerts = _process_frame_with_pipeline_with(upload_pipeline, frame, frame_index, "upload")
                sampled_frames += 1
                total_persons = int(frame_summary.get("total_persons", 0))
                helmet_count = int(frame_summary.get("helmet_count", 0))
                no_helmet_count = int(frame_summary.get("no_helmet_count", 0))
                unknown_count = int(frame_summary.get("unknown_count", 0))
                uncertain_count = int(frame_summary.get("uncertain_count", 0))
                zone_breaches = int(frame_summary.get("zone_breaches", 0))

                person_counts.append(total_persons)
                helmet_counts.append(helmet_count)
                no_helmet_counts.append(no_helmet_count)
                unknown_counts.append(unknown_count)
                uncertain_counts.append(uncertain_count)
                known_counts.append(int(frame_summary.get("known_count", helmet_count + no_helmet_count)))
                zone_breach_counts.append(zone_breaches)
                compliance_scores.append(int(frame_summary.get("compliance", 0)))
                statuses.append(str(frame_summary.get("status", "UNKNOWN")))

                if total_persons > best_person_count:
                    best_person_count = total_persons
                    best_detections = detections
                    best_frame_number = frame_summary.get("frame_number", frame_index)
                    best_frame_width = int(frame.shape[1])
                    best_frame_height = int(frame.shape[0])

                timeline_buffer.append({
                    "frame_number": frame_index,
                    "status": frame_summary["status"],
                    "compliance": frame_summary["compliance"],
                    "total_persons": frame_summary["total_persons"],
                    "alerts": alerts,
                })

                if sampled_frames >= max_sampled_frames:
                    break

            frame_index += 1
    finally:
        capture.release()

    person_counts_for_summary = [count for count in person_counts if count > 0] or person_counts
    helmet_counts_for_summary = [count for count in helmet_counts if count >= 0]
    no_helmet_counts_for_summary = [count for count in no_helmet_counts if count >= 0]

    aggregated["total_persons"] = _most_common_int(person_counts_for_summary, default=0)
    aggregated["helmet_count"] = _most_common_int(helmet_counts_for_summary, default=0)
    aggregated["no_helmet_count"] = _most_common_int(no_helmet_counts_for_summary, default=0)
    aggregated["unknown_count"] = _most_common_int(unknown_counts, default=0)
    aggregated["uncertain_count"] = _most_common_int(uncertain_counts, default=0)
    aggregated["known_count"] = _most_common_int(known_counts, default=0)
    aggregated["zone_breaches"] = _most_common_int(zone_breach_counts, default=0)
    aggregated["compliance"] = (
        int(round(sum(compliance_scores) / len(compliance_scores)))
        if compliance_scores
        else 0
    )
    aggregated["status"] = max(statuses or ["UNKNOWN"], key=_status_rank)
    aggregated["total_persons_range"] = _range_label(person_counts_for_summary, default="0")
    aggregated["helmet_count_range"] = _range_label(helmet_counts_for_summary, default="0")
    aggregated["no_helmet_count_range"] = _range_label(no_helmet_counts_for_summary, default="0")
    aggregated["zone_breaches_range"] = _range_label(zone_breach_counts, default="0")
    aggregated["sampled_frame_count"] = sampled_frames
    aggregated["representative_frame_number"] = best_frame_number or frame_index
    aggregated["representative_total_persons"] = best_person_count if best_person_count >= 0 else 0
    aggregated["representative_frame_width"] = best_frame_width or 0
    aggregated["representative_frame_height"] = best_frame_height or 0
    aggregated["frame_width"] = best_frame_width or 0
    aggregated["frame_height"] = best_frame_height or 0

    aggregated["frame_number"] = best_frame_number or frame_index
    latest_summary.update(aggregated)

    return {
        "detections": best_detections,
        "summary": aggregated,
        "frames_processed": frame_index,
        "sampled_frames": sampled_frames,
        "status": aggregated.get("status", "UNKNOWN"),
        "history": list(timeline_buffer),
        "recent_alerts": list(alert_buffer)[:10],
    }


def _process_frame_with_pipeline_with(pipeline, frame, frame_number: int, source: str) -> tuple[list[dict], dict, list[dict]]:
    if pipeline is None:
        raise RuntimeError("Safety pipeline is not available")

    _, tracks, alerts = pipeline.process_single_frame(frame, frame_number)
    detections = [_format_detection(track) for track in tracks]
    summary = _summarize_tracks(tracks, alerts, frame_number, source)
    summary["frame_width"] = int(frame.shape[1])
    summary["frame_height"] = int(frame.shape[0])
    summary["status"] = get_stable_status(summary["status"])
    if alerts:
        for alert in alerts:
            alert_data = _serialize_alert(alert)
            alert_buffer.appendleft(alert_data)
    return detections, summary, [_serialize_alert(a) for a in alerts]


@app.post("/api/analyze-video")
async def analyze_uploaded_video(request: Request):
    body = await request.body()
    if not body:
        return Response(
            content=json.dumps({"error": "No video data received"}),
            media_type="application/json",
            status_code=400,
        )

    filename = request.query_params.get("filename", "upload.mp4")
    suffix = Path(filename).suffix if Path(filename).suffix else ".mp4"
    frame_stride = max(1, int(request.query_params.get("frame_stride", "4")))
    max_sampled_frames = max(1, int(request.query_params.get("max_frames", "120")))

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(body)
            temp_path = temp_file.name

        result = analyze_video_file(
            temp_path,
            frame_stride=frame_stride,
            max_sampled_frames=max_sampled_frames,
        )
        return result
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    last_frame_id = -1
    try:
        while True:
            # Receive base64 frame from frontend
            data = await websocket.receive_text()
            if not data:
                continue

            try:
                frame_id = None
                raw_payload = data
                if raw_payload.lstrip().startswith("{"):
                    payload = json.loads(raw_payload)
                    frame_id = payload.get("frame_id")
                    raw_payload = payload.get("image", "")

                if frame_id is not None and int(frame_id) < last_frame_id:
                    continue

                # Remove header if present (e.g., "data:image/jpeg;base64,")
                if "," in raw_payload:
                    _, raw_payload = raw_payload.split(",", 1)
                
                # Add padding if missing
                missing_padding = len(raw_payload) % 4
                if missing_padding:
                    raw_payload += '=' * (4 - missing_padding)

                img_bytes = base64.b64decode(raw_payload)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    logger.warning("Failed to decode frame")
                    continue

                frame_number = latest_summary.get("frame_number", 0) + 1
                _, tracks, alerts = safety_pipeline.process_single_frame(frame, frame_number)
                detections = [_format_detection(track) for track in tracks]
                summary = _summarize_tracks(tracks, alerts, frame_number, "live")
                summary["frame_width"] = int(frame.shape[1])
                summary["frame_height"] = int(frame.shape[0])
                summary["status"] = get_stable_status(summary["status"])
                latest_summary.update(summary)
                latest_detections[:] = detections
                if frame_id is not None:
                    last_frame_id = int(frame_id)
                for alert in alerts:
                    alert_buffer.appendleft(_serialize_alert(alert))
                timeline_buffer.append({
                    "frame_number": frame_number,
                    "status": summary["status"],
                    "compliance": summary["compliance"],
                    "total_persons": summary["total_persons"],
                    "alerts": [_serialize_alert(a) for a in alerts],
                })

                # Send results back
                await websocket.send_json({
                    "frame_id": frame_id,
                    "frame_width": int(frame.shape[1]),
                    "frame_height": int(frame.shape[0]),
                    "detections": detections,
                    "status": summary["status"],
                    "summary": summary,
                    "alerts": [_serialize_alert(a) for a in alerts],
                })

            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    start_port = int(os.getenv("UI_PORT", "8001"))
    ui_port = find_free_port(start_port)
    print(f"UI backend available at http://127.0.0.1:{ui_port}")
    uvicorn.run(app, host="0.0.0.0", port=ui_port)
