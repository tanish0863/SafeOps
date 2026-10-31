import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add parent directory to sys.path so backend modules import cleanly
current_dir = Path(__file__).resolve().parent
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

from backend.config import CORS_ORIGINS, API_V1_PREFIX, UPLOAD_DIR
from backend.database import engine, Base, SessionLocal
from backend.seed import seed_database
from backend.routers import dashboard, media, detection, workers, violations, analytics, settings

# Ensure tables and seed data exist immediately
Base.metadata.create_all(bind=engine)
with SessionLocal() as _db:
    seed_database(_db)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure tables and seed data exist on server start
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        seed_database(db)
    yield


app = FastAPI(
    title="SafeOps Security Suite API",
    description="Unified Computer Vision & Workplace PPE Safety Operations API",
    version="2.4.0",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static File Uploads Directory
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# Health Checks
@app.get("/")
def root_status():
    return {
        "service": "SafeOps Security Suite API",
        "version": "2.4.0",
        "status": "online",
        "docsUrl": "/docs",
        "apiPrefix": API_V1_PREFIX,
    }


@app.get("/health")
@app.get(f"{API_V1_PREFIX}/health")
def health_check():
    return {
        "status": "healthy",
        "service": "safeops-api",
        "database": "connected",
        "timestamp": os.getenv("APP_ENV", "local"),
    }


# Include Routers under /api/v1
app.include_router(dashboard.router, prefix=API_V1_PREFIX)
app.include_router(media.router, prefix=API_V1_PREFIX)
app.include_router(detection.router, prefix=API_V1_PREFIX)
app.include_router(workers.router, prefix=API_V1_PREFIX)
app.include_router(violations.router, prefix=API_V1_PREFIX)
app.include_router(analytics.router, prefix=API_V1_PREFIX)
app.include_router(settings.router, prefix=API_V1_PREFIX)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
