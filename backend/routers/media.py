import os
import time
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.config import UPLOAD_DIR
from backend.database import get_db
from backend.models import MediaFileModel
from backend.schemas import UploadedMediaResponse

router = APIRouter(prefix="/media", tags=["Media Storage"])


def save_uploaded_file(file: UploadFile, allowed_prefixes: tuple[str, ...], file_kind: str, db: Session) -> UploadedMediaResponse:
    content_type = file.content_type or ""
    filename = file.filename or f"upload_{int(time.time())}"
    
    # Check allowed formats
    valid = any(content_type.startswith(p) for p in allowed_prefixes) or any(
        filename.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".mp4", ".webm", ".mov", ".mkv")
    )
    if not valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format for {file_kind}. Allowed: JPG, PNG, WebP for images; MP4, WebM, MOV for videos.",
        )

    media_id = f"med_{int(time.time() * 1000)}_{os.urandom(4).hex()}"
    ext = os.path.splitext(filename)[1] or (".mp4" if "video" in content_type else ".jpg")
    safe_filename = f"{media_id}{ext}"
    dest_path = os.path.join(UPLOAD_DIR, safe_filename)

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size = os.path.getsize(dest_path)

    # Persist in DB
    db_media = MediaFileModel(
        id=media_id,
        file_name=filename,
        file_type="video" if "video" in content_type or ext in (".mp4", ".webm", ".mov", ".mkv") else "image",
        file_path=dest_path,
        file_size_bytes=file_size,
    )
    db.add(db_media)
    db.commit()

    return UploadedMediaResponse(
        mediaId=media_id,
        fileName=filename,
        mediaType="video" if db_media.file_type == "video" else "image",
        fileSizeBytes=file_size,
        storageUrl=f"/uploads/{safe_filename}",
        uploadedAt=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


@router.post("/image")
def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    res = save_uploaded_file(file, ("image/",), "image", db)
    # Return contract matching UploadedMedia in detectionService.ts
    return {"mediaId": res.mediaId, "fileName": res.fileName}


@router.post("/video")
def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    res = save_uploaded_file(file, ("video/",), "video", db)
    return {"mediaId": res.mediaId, "fileName": res.fileName}


@router.post("/upload", response_model=UploadedMediaResponse)
def upload_generic_media(file: UploadFile = File(...), db: Session = Depends(get_db)):
    return save_uploaded_file(file, ("image/", "video/"), "media", db)
