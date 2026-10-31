import httpx
import sys

def main():
    base_url = "http://127.0.0.1:8000/api/v1"
    print(f"Connecting to live backend at {base_url}...")

    # 1. Health
    r = httpx.get(f"{base_url}/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    print(f"[OK] Health check passed: {r.json()['status']}")

    # 2. Dashboard Stats
    r = httpx.get(f"{base_url}/dashboard/stats")
    assert r.status_code == 200
    stats = r.json()
    print(f"[OK] Dashboard stats retrieved: {stats['totalWorkers']} workers, {stats['complianceRate']}% compliance")

    # 3. Workers
    r = httpx.get(f"{base_url}/workers")
    assert r.status_code == 200
    workers = r.json()
    initial_count = len(workers)
    print(f"[OK] Fetched {initial_count} workers from SQLite database")

    # 4. Create Worker
    new_worker = {
        "name": "Integration Test Worker",
        "zone": "Welding Bay C",
        "shift": "B",
        "helmet": True,
        "vest": False,
        "status": "violation"
    }
    r = httpx.post(f"{base_url}/workers", json=new_worker)
    assert r.status_code == 201
    created = r.json()
    created_id = created["id"]
    print(f"[OK] Created worker ID: {created_id} ({created['name']})")

    # Verify count increased
    r = httpx.get(f"{base_url}/workers")
    assert len(r.json()) == initial_count + 1
    print(f"[OK] Worker persistence verified in SQLite (count: {len(r.json())})")

    # Clean up test worker
    httpx.delete(f"{base_url}/workers/{created_id}")
    print(f"[OK] Cleaned up test worker {created_id}")

    # 5. Violations: Create a real test violation
    new_violation = {
        "workerId": "W-102",
        "workerName": "Test Personnel",
        "type": "Missing Safety Vest",
        "zone": "Welding Bay",
        "severity": "critical",
        "reviewStatus": "pending",
        "note": "Live optical test violation",
    }
    r = httpx.post(f"{base_url}/violations", json=new_violation)
    assert r.status_code == 201
    created_v = r.json()
    first_v_id = created_v["id"]
    print(f"[OK] Created live violation {first_v_id}")

    # 6. Resolve Violation
    r = httpx.post(f"{base_url}/violations/{first_v_id}/resolve")
    if r.status_code != 200:
        print(f"Resolve failed: code={r.status_code}, response={r.text}")
    assert r.status_code == 200
    assert r.json()["reviewStatus"] == "resolved"
    print(f"[OK] Successfully resolved violation {first_v_id}")

    # Clean up test violation
    httpx.delete(f"{base_url}/violations/{first_v_id}")
    print(f"[OK] Cleaned up test violation {first_v_id}")

    # 7. Detection Analysis
    r = httpx.post(
        f"{base_url}/detection/analyze",
        json={"fileName": "test_security_frame.jpg", "rules": {"helmetRequired": True, "vestRequired": True}}
    )
    assert r.status_code == 200
    analysis = r.json()
    print(f"[OK] Detection analysis executed: {len(analysis['workers'])} workers evaluated (isDemo: {analysis['isDemo']})")

    # 8. Settings
    r = httpx.get(f"{base_url}/settings")
    assert r.status_code == 200
    settings = r.json()
    print(f"[OK] System settings retrieved (retention: {settings['retentionDays']} days)")

    # 9. Analytics
    r = httpx.get(f"{base_url}/analytics/all")
    assert r.status_code == 200
    print("[OK] Aggregated analytics retrieved (trends, zones, and PPE breakdown)")

    print("\n>>> ALL LIVE BACKEND ENDPOINTS AND PERSISTENCE VERIFIED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    main()
