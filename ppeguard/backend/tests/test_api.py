import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "safeops-api"


def test_dashboard_stats():
    response = client.get("/api/v1/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    assert "complianceRate" in data
    assert "totalWorkers" in data
    assert "compliantCount" in data
    assert "violationCount" in data
    assert "criticalCount" in data


def test_dashboard_telemetry():
    response = client.get("/api/v1/dashboard/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert data["activeSensors"] == 8
    assert data["fps"] > 0
    assert data["cudaActive"] is True


def test_workers_crud():
    # 1. List workers
    list_res = client.get("/api/v1/workers")
    assert list_res.status_code == 200

    # 2. Create worker
    new_worker = {
        "name": "Integration Test Worker",
        "zone": "Robotics Cell 4",
        "shift": "B",
        "helmet": True,
        "vest": False,
        "status": "violation",
        "needsReview": False,
    }
    create_res = client.post("/api/v1/workers", json=new_worker)
    assert create_res.status_code == 201
    created = create_res.json()
    assert created["name"] == "Integration Test Worker"
    worker_id = created["id"]

    # 3. Get single worker
    get_res = client.get(f"/api/v1/workers/{worker_id}")
    assert get_res.status_code == 200
    assert get_res.json()["zone"] == "Robotics Cell 4"

    # 4. Update worker
    update_res = client.patch(f"/api/v1/workers/{worker_id}", json={"vest": True, "status": "compliant"})
    assert update_res.status_code == 200
    assert update_res.json()["vest"] is True
    assert update_res.json()["status"] == "compliant"

    # 5. Delete worker
    del_res = client.delete(f"/api/v1/workers/{worker_id}")
    assert del_res.status_code == 200

    # Verify deleted
    get_del = client.get(f"/api/v1/workers/{worker_id}")
    assert get_del.status_code == 404


def test_violations_lifecycle():
    # 1. Create a violation
    new_v = {
        "workerId": "W-999",
        "workerName": "Audit User",
        "type": "Missing Helmet",
        "zone": "Loading Dock",
        "severity": "critical",
        "reviewStatus": "pending",
        "note": "Test violation event",
    }
    create_res = client.post("/api/v1/violations", json=new_v)
    assert create_res.status_code == 201
    v_id = create_res.json()["id"]

    # 2. Filter violations by severity
    res_crit = client.get("/api/v1/violations?severity=critical")
    assert res_crit.status_code == 200
    crit_list = res_crit.json()
    assert any(v["id"] == v_id for v in crit_list)

    # 3. Resolve violation
    resolve_res = client.post(f"/api/v1/violations/{v_id}/resolve")
    assert resolve_res.status_code == 200
    assert resolve_res.json()["reviewStatus"] == "resolved"

    # 4. Clean up violation
    del_res = client.delete(f"/api/v1/violations/{v_id}")
    assert del_res.status_code == 204


def test_media_upload_and_detection():
    # 1. Upload mock image
    file_content = b"\xFF\xD8\xFF\xE0\x00\x10JFIF" + b"\x00" * 200  # Minimal JPEG header
    files = {"file": ("test_frame.jpg", file_content, "image/jpeg")}
    upload_res = client.post("/api/v1/media/image", files=files)
    assert upload_res.status_code == 200
    media_data = upload_res.json()
    assert "mediaId" in media_data
    media_id = media_data["mediaId"]

    # 2. Create detection job
    job_payload = {
        "mediaId": media_id,
        "fileName": "test_frame.jpg",
        "rules": {"helmetRequired": True, "vestRequired": True},
    }
    job_res = client.post("/api/v1/detection/jobs", json=job_payload)
    assert job_res.status_code == 200
    job_data = job_res.json()
    assert job_data["status"] == "complete"
    job_id = job_data["jobId"]

    # 3. Fetch detection result
    result_res = client.get(f"/api/v1/detection/jobs/{job_id}/result")
    assert result_res.status_code == 200
    analysis = result_res.json()
    assert analysis["jobId"] == job_id
    assert analysis["isDemo"] is True
    assert len(analysis["workers"]) == 3
    # Check normalized bounding box coordinates
    for w in analysis["workers"]:
        assert 0.0 <= w["box"]["x"] <= 1.0
        assert 0.0 <= w["box"]["y"] <= 1.0
        assert 0.0 <= w["box"]["w"] <= 1.0
        assert 0.0 <= w["box"]["h"] <= 1.0
        assert w["compliance"] in ("compliant", "violation", "review")


def test_analytics_endpoints():
    res_trend = client.get("/api/v1/analytics/compliance-trend")
    assert res_trend.status_code == 200
    assert len(res_trend.json()) == 12

    res_zones = client.get("/api/v1/analytics/zone-distribution")
    assert res_zones.status_code == 200
    assert len(res_zones.json()) >= 5

    res_ppe = client.get("/api/v1/analytics/ppe-breakdown")
    assert res_ppe.status_code == 200
    assert len(res_ppe.json()) >= 2


def test_settings_persistence():
    # 1. Fetch current settings
    res = client.get("/api/v1/settings")
    assert res.status_code == 200
    settings = res.json()
    assert "personConfidenceThreshold" in settings

    # 2. Update settings
    updated = {**settings, "personConfidenceThreshold": 0.65, "alertCriticalViolations": False}
    put_res = client.put("/api/v1/settings", json=updated)
    assert put_res.status_code == 200
    assert put_res.json()["personConfidenceThreshold"] == 0.65

    # 3. Verify persisted
    check_res = client.get("/api/v1/settings")
    assert check_res.json()["personConfidenceThreshold"] == 0.65
    assert check_res.json()["alertCriticalViolations"] is False
