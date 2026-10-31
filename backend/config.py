import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR.parent
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(WORKSPACE_ROOT / "uploads"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{WORKSPACE_ROOT / 'safeops.db'}")
CORS_ORIGINS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*",
]
API_V1_PREFIX = "/api/v1"

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)
