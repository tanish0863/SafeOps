"""
input_handler.py — Unified video input abstraction.

Supports:
  - Webcam (index or device path)
  - Video file (mp4, avi, mkv, etc.)

The VideoSource class provides a consistent generator interface
regardless of input type, ensuring main.py has zero branching logic
for source type.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoSourceError(RuntimeError):
    pass


class VideoSource:
    """
    Context-manager-safe video source that normalises webcam and file inputs.

    Usage:
        with VideoSource(source=0) as src:
            for frame, frame_number, timestamp in src.stream():
                process(frame)

        with VideoSource(source="path/to/video.mp4") as src:
            for frame, frame_number, timestamp in src.stream():
                process(frame)
    """

    def __init__(
        self,
        source: Union[int, str, Path] = 0,
        target_fps: Optional[float]   = None,   # None = as fast as possible
        loop_video: bool              = False,   # Re-loop when file ends
        buffer_size: int              = 1,       # CV_CAP_PROP_BUFFERSIZE (reduce latency)
    ) -> None:
        self.source      = source
        self.target_fps  = target_fps
        self.loop_video  = loop_video
        self.buffer_size = buffer_size

        self._cap:          Optional[cv2.VideoCapture] = None
        self._frame_count:  int   = 0
        self._source_fps:   float = 0.0
        self._is_file:      bool  = isinstance(source, (str, Path)) and str(source) != "0"
        self._is_image:     bool  = False
        self._image_frame:  Optional[np.ndarray] = None
        if self._is_file:
            ext = Path(str(source)).suffix.lower()
            if ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                self._is_image = True

    def open(self) -> None:
        if self._is_image:
            img = cv2.imread(str(self.source))
            if img is None:
                raise VideoSourceError(f"Cannot open image source: {self.source!r}")
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            self._image_frame = img
            self._source_fps = 30.0
            h, w = img.shape[:2]
            logger.info(f"[VideoSource] Opened image: {self.source} | {w}×{h} (looping for live test)")
            return

        src = int(self.source) if str(self.source).isdigit() else str(self.source)
        self._cap = cv2.VideoCapture(src)

        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise VideoSourceError(f"Cannot open video source: {self.source!r}")

        # Reduce latency for webcam sources
        if not self._is_file:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self._source_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration     = total_frames / self._source_fps if total_frames > 0 else 0

        logger.info(
            f"[VideoSource] Opened {'file' if self._is_file else 'webcam'}: {self.source} "
            f"| {self.width}×{self.height} @ {self._source_fps:.1f}fps "
            f"{'| ' + f'{duration:.1f}s total' if self._is_file else ''}"
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._image_frame = None
        logger.info("[VideoSource] Capture released.")

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def width(self) -> int:
        if self._is_image and self._image_frame is not None:
            return int(self._image_frame.shape[1])
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def height(self) -> int:
        if self._is_image and self._image_frame is not None:
            return int(self._image_frame.shape[0])
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0

    @property
    def source_fps(self) -> float:
        return self._source_fps

    def stream(self) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        """
        Generator that yields (frame, frame_number, timestamp_seconds).

        For webcam: always reads latest frame (drops stale buffer frames).
        For file: reads sequentially, respects target_fps throttling.
        """
        if self._is_image:
            if self._image_frame is None:
                raise VideoSourceError("Image source not opened. Use as context manager.")
            while True:
                self._frame_count += 1
                now = time.monotonic()
                yield self._image_frame.copy(), self._frame_count, self._frame_count / 30.0
                time.sleep(1.0 / 30.0)

        if self._cap is None:
            raise VideoSourceError("VideoSource not opened. Use as context manager.")

        frame_interval = 1.0 / self.target_fps if self.target_fps else 0.0
        last_yield_time = 0.0

        while True:
            read_ok, frame = self._cap.read()

            if not read_ok:
                if self._is_file and self.loop_video:
                    logger.info("[VideoSource] End of file — looping.")
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._frame_count = 0
                    continue
                else:
                    logger.info("[VideoSource] Stream ended.")
                    break

            now = time.monotonic()
            if frame_interval > 0 and (now - last_yield_time) < frame_interval:
                # Throttle to target FPS (mainly useful for file sources)
                continue

            last_yield_time = now
            self._frame_count += 1

            ts = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if self._is_file else now
            yield frame, self._frame_count, ts

    def seek(self, frame_number: int) -> None:
        """Seek to a specific frame (file sources only)."""
        if self._is_file and self._cap:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self._frame_count = frame_number

    @staticmethod
    def list_webcams(max_index: int = 5) -> list:
        """Scan for available webcam indices."""
        available = []
        if os.name == "nt":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        for i in range(max_index):
            for backend in backends:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    available.append(i)
                    cap.release()
                    break
                cap.release()
        return available
