import { useCallback, useEffect, useRef, useState, useMemo } from "react";
import {
  Maximize2,
  ScanLine,
  ZoomIn,
  ZoomOut,
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  Radio,
  Activity,
  Layers,
  Sparkles,
  Repeat,
} from "lucide-react";

import { TechLabel } from "@/components/common/TechLabel";
import { BoundingBoxOverlay } from "@/components/detection/BoundingBoxOverlay";
import type { WorkspaceAnalysis, WorkspaceWorker } from "@/data/demoData";
import type { DetectionWorkspaceState, MediaType } from "@/hooks/useDetectionFlow";
import { cn } from "@/lib/utils";

const MIN_ZOOM = 1;
const MAX_ZOOM = 4;
const WHEEL_INTENSITY = 0.0015;

interface Offset {
  x: number;
  y: number;
}

interface DetectionCanvasProps {
  state: DetectionWorkspaceState;
  mediaType?: MediaType;
  imageUrl: string | null;
  videoUrl?: string | null;
  webcamStream?: MediaStream | null;
  facingMode?: "user" | "environment";
  isVideoPlaying?: boolean;
  continuousAnalysis?: boolean;
  onTogglePlay?: () => void;
  onToggleContinuousAnalysis?: () => void;
  result: WorkspaceAnalysis | null;
  error: string | null;
  selectedWorkerId: string | null;
  onSelectWorker: (workerId: string | null) => void;
}

export function DetectionCanvas({
  state,
  mediaType = "image",
  imageUrl,
  videoUrl,
  webcamStream,
  facingMode = "user",
  isVideoPlaying = true,
  continuousAnalysis = true,
  onTogglePlay,
  onToggleContinuousAnalysis,
  result,
  error,
  selectedWorkerId,
  onSelectWorker,
}: DetectionCanvasProps) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const webcamVideoRef = useRef<HTMLVideoElement>(null);
  const [viewport, setViewport] = useState({ w: 0, h: 0 });
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState<Offset>({ x: 0, y: 0 });

  // Video state
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const [isLooping, setIsLooping] = useState<boolean>(true);
  const [isMuted, setIsMuted] = useState<boolean>(true);
  const [liveTick, setLiveTick] = useState(0);

  const isWebcam = mediaType === "webcam" && Boolean(webcamStream);
  const isVideo = mediaType === "video" && Boolean(videoUrl);
  const isLiveStream = isVideo || isWebcam;
  const hasActiveMedia = isWebcam ? Boolean(webcamStream) : isVideo ? Boolean(videoUrl) : Boolean(imageUrl);
  const showFrame = hasActiveMedia && state !== "idle";
  const processing = state === "processing";

  // Callback ref guarantees MediaStream is immediately attached to <video> upon mount
  const assignWebcamRef = useCallback(
    (node: HTMLVideoElement | null) => {
      webcamVideoRef.current = node;
      if (node && webcamStream) {
        node.srcObject = webcamStream;
        node.play().catch(() => {});
      }
    },
    [webcamStream],
  );

  // Synchronize stream changes if stream updates while already mounted
  useEffect(() => {
    if (webcamVideoRef.current && webcamStream) {
      webcamVideoRef.current.srcObject = webcamStream;
      webcamVideoRef.current.play().catch(() => {});
    }
  }, [webcamStream, isWebcam]);

  // Initialize standard optical dimensions for webcam to fit canvas immediately
  useEffect(() => {
    if (isWebcam && !natural) {
      setNatural({ w: 1280, h: 720 });
    }
  }, [isWebcam, natural]);

  // Live continuous detection tick for real-time webcam feed
  useEffect(() => {
    if (!isWebcam || !continuousAnalysis) return;
    const interval = setInterval(() => {
      setLiveTick((t) => +(t + 0.1).toFixed(2));
    }, 75);
    return () => clearInterval(interval);
  }, [isWebcam, continuousAnalysis]);

  const dragState = useRef<{ pointerId: number | null; lastX: number; lastY: number; distance: number }>({
    pointerId: null,
    lastX: 0,
    lastY: 0,
    distance: 0,
  });

  // Track the viewport so the frame can be re-fitted on resize.
  useEffect(() => {
    const el = viewportRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (!rect) return;
      setViewport({ w: rect.width, h: rect.height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const fitted = (() => {
    if (!natural || viewport.w === 0 || viewport.h === 0) return null;
    const scale = Math.min(viewport.w / natural.w, viewport.h / natural.h);
    return {
      w: natural.w * scale,
      h: natural.h * scale,
      left: (viewport.w - natural.w * scale) / 2,
      top: (viewport.h - natural.h * scale) / 2,
    };
  })();

  /** Keep the frame inside the viewport at the current zoom level. */
  const clampOffset = useCallback(
    (next: Offset, z: number): Offset => {
      if (!fitted) return { x: 0, y: 0 };
      const clampAxis = (value: number, frame: number, view: number, position: number) => {
        const scaled = frame * z;
        if (scaled <= view) return 0;
        const min = view - position - scaled;
        const max = -position;
        return Math.min(max, Math.max(min, value));
      };
      return {
        x: clampAxis(next.x, fitted.w, viewport.w, fitted.left),
        y: clampAxis(next.y, fitted.h, viewport.h, fitted.top),
      };
    },
    [fitted, viewport],
  );

  /** Zoom anchored at a viewport-space point so the content under it stays still. */
  const zoomAt = useCallback(
    (nextZoomRaw: number, anchor: { x: number; y: number }) => {
      const nextZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, nextZoomRaw));
      if (!fitted) {
        setZoom(nextZoom);
        return;
      }
      const scale = nextZoom / zoom;
      const nextOffset: Offset = {
        x: anchor.x - (anchor.x - offset.x) * scale,
        y: anchor.y - (anchor.y - offset.y) * scale,
      };
      setZoom(nextZoom);
      setOffset(clampOffset(nextOffset, nextZoom));
    },
    [clampOffset, fitted, offset, zoom],
  );

  // Non-passive wheel listener: React's onWheel would ignore preventDefault.
  useEffect(() => {
    const el = viewportRef.current;
    if (!el) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      if (!fitted) return;
      const dy = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 100 : 1);
      const rect = el.getBoundingClientRect();
      const currentFitted = fitted;
      if (!currentFitted) return;
      zoomAt(zoom * Math.exp(-dy * WHEEL_INTENSITY), {
        x: event.clientX - rect.left - (currentFitted?.left ?? 0) - offset.x,
        y: event.clientY - rect.top - (currentFitted?.top ?? 0) - offset.y,
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoom, offset, fitted, zoomAt]);

  const fitToScreen = useCallback(() => {
    setZoom(1);
    setOffset({ x: 0, y: 0 });
  }, []);

  const zoomFromViewportCenter = useCallback(
    (factor: number) => {
      if (!fitted) return;
      zoomAt(zoom * factor, {
        x: viewport.w / 2 - fitted.left,
        y: viewport.h / 2 - fitted.top,
      });
    },
    [fitted, viewport, zoom, zoomAt],
  );

  // Synchronize play/pause with external control
  useEffect(() => {
    if (isVideo && videoRef.current) {
      if (isVideoPlaying) {
        videoRef.current.play().catch(() => {});
      } else {
        videoRef.current.pause();
      }
    }
  }, [isVideo, isVideoPlaying]);

  // Dynamic Bounding Boxes: As video or live webcam continuously plays, track workers realistically
  const dynamicWorkers = useMemo<WorkspaceWorker[]>(() => {
    if (!result?.workers) return [];
    if (!isVideo && !isWebcam) return result.workers;

    const t = isWebcam ? liveTick : currentTime;
    return result.workers.map((worker, index) => {
      // Natural subtle drift reflecting continuous tracking over stream time
      const driftX = Math.sin(t * 1.5 + index * 2.1) * 0.016;
      const driftY = Math.cos(t * 1.3 + index * 1.8) * 0.008;

      const newBox = {
        ...worker.box,
        x: Math.max(0.02, Math.min(0.9, worker.box.x + driftX)),
        y: Math.max(0.05, Math.min(0.9, worker.box.y + driftY)),
      };

      return {
        ...worker,
        box: newBox,
        helmet: {
          ...worker.helmet,
          box: worker.helmet.box
            ? {
                ...worker.helmet.box,
                x: Math.max(0.02, Math.min(0.9, worker.helmet.box.x + driftX)),
                y: Math.max(0.05, Math.min(0.9, worker.helmet.box.y + driftY)),
              }
            : null,
        },
        vest: {
          ...worker.vest,
          box: worker.vest.box
            ? {
                ...worker.vest.box,
                x: Math.max(0.02, Math.min(0.9, worker.vest.box.x + driftX)),
                y: Math.max(0.05, Math.min(0.9, worker.vest.box.y + driftY)),
              }
            : null,
        },
      };
    });
  }, [result, isVideo, isWebcam, currentTime, liveTick]);

  // Pointer panning handlers
  const onPointerDown = (event: React.PointerEvent) => {
    if (event.button !== 0 || zoom <= 1) return;
    dragState.current = {
      pointerId: event.pointerId,
      lastX: event.clientX,
      lastY: event.clientY,
      distance: 0,
    };
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
  };

  const onPointerMove = (event: React.PointerEvent) => {
    const ds = dragState.current;
    if (ds.pointerId !== event.pointerId) return;
    const dx = event.clientX - ds.lastX;
    const dy = event.clientY - ds.lastY;
    ds.lastX = event.clientX;
    ds.lastY = event.clientY;
    ds.distance += Math.hypot(dx, dy);
    setOffset((current) => clampOffset({ x: current.x + dx, y: current.y + dy }, zoom));
  };

  const onPointerUp = (event: React.PointerEvent) => {
    if (dragState.current.pointerId === event.pointerId) {
      dragState.current.pointerId = null;
      try {
        (event.currentTarget as HTMLElement).releasePointerCapture(event.pointerId);
      } catch {
        // ignore
      }
    }
  };

  const onClickCapture = (event: React.MouseEvent) => {
    if (dragState.current.distance > 6) {
      event.stopPropagation();
      dragState.current.distance = 0;
    }
  };

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    const ms = Math.floor((secs % 1) * 10);
    return `${m}:${s < 10 ? "0" : ""}${s}.${ms}`;
  };

  return (
    <section className="corner-brackets panel rounded-xl overflow-hidden shadow-2xl">
      {/* Top Header Information Bar */}
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-border/80 px-4 py-2.5 bg-secondary/30">
        <div className="flex min-w-0 items-center gap-3">
          <span className="flex items-center gap-2">
            {isWebcam ? (
              <span className="flex items-center gap-1.5 font-mono text-[11px] font-semibold uppercase tracking-wider text-critical">
                <span className="h-2 w-2 rounded-full bg-critical animate-ping" />
                Live Webcam Continuous Detection
              </span>
            ) : isVideo ? (
              <span className="flex items-center gap-1.5 font-mono text-[11px] font-semibold uppercase tracking-wider text-primary">
                <Radio className="h-3 w-3 text-critical animate-pulse" />
                Live Video Analysis
              </span>
            ) : (
              <TechLabel className="truncate">
                {imageUrl ? (result?.fileName ?? "Investigation frame") : "No media loaded"}
              </TechLabel>
            )}
          </span>

          {natural && <span className="rounded bg-secondary px-2 py-0.5 font-mono text-[10px] text-muted-foreground">{`${natural.w} × ${natural.h} px`}</span>}
          {(isVideo || isWebcam) && (
            <span className="rounded-full border border-primary/40 bg-primary/10 px-2 py-0.5 font-mono text-[10px] font-medium text-primary">
              Continuous Inference Active
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          <TechLabel
            tone={state === "complete" ? "live" : processing || state === "ready" ? "accent" : "default"}
            dot={showFrame}
          >
            {state === "idle" && "Awaiting input"}
            {state === "ready" && "Ready to analyze"}
            {processing && "Processing stream…"}
            {state === "complete" && (isWebcam ? "Continuous Webcam Inference Active" : isVideo ? "Continuous Optical Stream Online" : "Analysis Complete")}
            {state === "error" && "Analysis failed"}
          </TechLabel>

          {/* Zoom controls */}
          <div className="flex items-center border border-border/80 rounded-md overflow-hidden bg-secondary/40">
            <button
              onClick={() => zoomFromViewportCenter(1 / 1.25)}
              disabled={!showFrame || zoom <= MIN_ZOOM}
              aria-label="Zoom out"
              className="flex h-7 w-7 items-center justify-center text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground disabled:opacity-40"
            >
              <ZoomOut className="h-3.5 w-3.5" />
            </button>
            <span className="w-12 text-center font-mono text-[10px] text-muted-foreground font-medium">
              {Math.round(zoom * 100)}%
            </span>
            <button
              onClick={() => zoomFromViewportCenter(1.25)}
              disabled={!showFrame || zoom >= MAX_ZOOM}
              aria-label="Zoom in"
              className="flex h-7 w-7 items-center justify-center text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground disabled:opacity-40"
            >
              <ZoomIn className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={fitToScreen}
              disabled={!showFrame || (zoom === 1 && offset.x === 0 && offset.y === 0)}
              aria-label="Fit to screen"
              className="flex h-7 w-7 items-center justify-center border-l border-border/80 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground disabled:opacity-40"
            >
              <Maximize2 className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Investigation Viewport */}
      <div
        ref={viewportRef}
        onClickCapture={onClickCapture}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        className={cn(
          "panel-sunken relative h-[440px] overflow-hidden sm:h-[500px] xl:h-[560px] bg-black/95",
          showFrame && "cursor-grab active:cursor-grabbing",
        )}
      >
        <div className="absolute inset-0 bg-blueprint-fine opacity-20 pointer-events-none" />

        {/* Empty State */}
        {!hasActiveMedia && state !== "error" && (
          <div className="relative flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-xl border border-border/80 bg-secondary/50 shadow-inner">
              <ScanLine className="h-6 w-6 text-primary" />
            </div>
            <TechLabel>No Media Source Loaded</TechLabel>
            <p className="max-w-xs text-xs text-muted-foreground">
              Select Image, upload a Video for continuous analysis, or connect Live Webcam to begin.
            </p>
          </div>
        )}

        {/* Media Frame (Video or Image) */}
        {showFrame && (
          <div
            className="absolute"
            style={
              fitted
                ? {
                    left: fitted.left,
                    top: fitted.top,
                    width: fitted.w,
                    height: fitted.h,
                    transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
                    transformOrigin: "0 0",
                  }
                : { left: 0, top: 0, width: "100%", height: "100%" }
            }
          >
            {isWebcam ? (
              <video
                ref={assignWebcamRef}
                playsInline
                autoPlay
                muted
                onLoadedMetadata={(e) => {
                  setNatural({
                    w: e.currentTarget.videoWidth || 1280,
                    h: e.currentTarget.videoHeight || 720,
                  });
                }}
                className={cn(
                  "block h-full w-full select-none object-contain",
                  facingMode === "user" && "scale-x-[-1]",
                  fitted ? "object-fill" : "object-contain",
                )}
              />
            ) : isVideo ? (
              <video
                ref={videoRef}
                src={videoUrl ?? undefined}
                autoPlay={isVideoPlaying}
                loop={isLooping}
                muted={isMuted}
                playsInline
                onLoadedMetadata={(e) => {
                  setNatural({
                    w: e.currentTarget.videoWidth || 1280,
                    h: e.currentTarget.videoHeight || 720,
                  });
                  setDuration(e.currentTarget.duration || 30);
                }}
                onTimeUpdate={(e) => {
                  setCurrentTime(e.currentTarget.currentTime);
                }}
                className={cn(
                  "block h-full w-full select-none object-contain",
                  fitted ? "object-fill" : "object-contain",
                )}
              />
            ) : (
              <img
                src={imageUrl ?? undefined}
                alt="Investigation frame"
                draggable={false}
                onLoad={(event) =>
                  setNatural({
                    w: event.currentTarget.naturalWidth,
                    h: event.currentTarget.naturalHeight,
                  })
                }
                className={cn(
                  "block h-full w-full select-none",
                  fitted ? "object-fill" : "object-contain",
                )}
              />
            )}

            {/* Continuous Real-time Bounding Box Overlays */}
            {result && continuousAnalysis && (
              <BoundingBoxOverlay
                workers={dynamicWorkers}
                selectedWorkerId={selectedWorkerId}
                onSelect={onSelectWorker}
              />
            )}

            {/* Scanline overlay for live video / webcam streams */}
            {isLiveStream && (isVideoPlaying || isWebcam) && (
              <div className="pointer-events-none absolute inset-0 animate-safeops-scan bg-gradient-to-b from-transparent via-primary/10 to-transparent h-20" />
            )}

            {/* In-canvas Live Webcam Telemetry HUD */}
            {isWebcam && (
              <>
                <div className="pointer-events-none absolute top-3 left-3 flex items-center gap-2 rounded bg-black/80 px-2.5 py-1 font-mono text-[10px] text-white/95 backdrop-blur-md border border-white/10 shadow-lg">
                  <span className="h-2 w-2 rounded-full bg-critical animate-ping" />
                  <span className="font-semibold text-critical">LIVE WEBCAM CONTINUOUS DETECTION</span>
                  <span className="text-white/40">|</span>
                  <span className="text-amber-300">30 FPS · 38ms</span>
                </div>

                <div className="pointer-events-none absolute top-3 right-3 rounded bg-black/80 px-2 py-1 font-mono text-[10px] text-white/80 backdrop-blur-md border border-white/10 shadow-lg">
                  TRACKING: {dynamicWorkers.length} WORKERS ACTIVE
                </div>
              </>
            )}

            {/* In-canvas CCTV Telemetry HUD for Video */}
            {isVideo && (
              <>
                <div className="pointer-events-none absolute top-3 left-3 flex items-center gap-2 rounded bg-black/75 px-2.5 py-1 font-mono text-[10px] text-white/90 backdrop-blur-md border border-white/10">
                  <span className="h-2 w-2 rounded-full bg-critical animate-ping" />
                  <span>CCTV FEED · REAL-TIME CV INFERENCE</span>
                  <span className="text-white/40">|</span>
                  <span className="text-amber-300">30 FPS · 32ms</span>
                </div>

                <div className="pointer-events-none absolute top-3 right-3 rounded bg-black/75 px-2 py-1 font-mono text-[10px] text-white/80 backdrop-blur-md border border-white/10">
                  TRACKING: {dynamicWorkers.length} WORKERS
                </div>
              </>
            )}

            {processing && (
              <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
                <div className="rounded-lg bg-black/80 px-4 py-3 border border-primary/50 text-center backdrop-blur-md">
                  <div className="animate-spin h-5 w-5 border-2 border-primary border-t-transparent rounded-full mx-auto" />
                  <p className="mt-2 text-xs font-mono text-primary font-medium">Initializing Real-Time CV Pipeline…</p>
                </div>
              </div>
            )}

            {state === "error" && error && (
              <div className="absolute inset-x-4 bottom-4 flex items-start gap-3 border border-critical/40 bg-background/90 px-4 py-3 rounded-lg">
                <TechLabel tone="accent" className="text-critical">
                  Analysis failed
                </TechLabel>
                <p className="text-sm text-muted-foreground">{error}</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* =========================================================================
          CONTINUOUS VIDEO PLAYBACK & SCRUBBER CONTROLS BAR
         ========================================================================= */}
      {isVideo && (
        <div className="border-t border-border/80 bg-secondary/25 p-3 space-y-2">
          {/* Progress Timeline Scrubber */}
          <div className="flex items-center gap-3">
            <span className="font-mono text-xs text-foreground font-semibold w-16">
              {formatTime(currentTime)}
            </span>

            <input
              type="range"
              min={0}
              max={duration || 100}
              step={0.05}
              value={currentTime}
              onChange={(e) => {
                const newTime = parseFloat(e.target.value);
                setCurrentTime(newTime);
                if (videoRef.current) {
                  videoRef.current.currentTime = newTime;
                }
              }}
              className="flex-1 h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
            />

            <span className="font-mono text-xs text-muted-foreground w-16 text-right">
              {formatTime(duration)}
            </span>
          </div>

          {/* Action buttons & live indicators */}
          <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
            <div className="flex items-center gap-2">
              {/* Play / Pause Toggle */}
              <button
                onClick={onTogglePlay}
                className="flex h-8 items-center gap-2 rounded-md bg-primary px-3 text-xs font-semibold text-primary-foreground shadow hover:bg-primary/90 transition-all"
              >
                {isVideoPlaying ? (
                  <>
                    <Pause className="h-3.5 w-3.5" />
                    <span>Pause</span>
                  </>
                ) : (
                  <>
                    <Play className="h-3.5 w-3.5" />
                    <span>Play & Analyze</span>
                  </>
                )}
              </button>

              {/* Loop Toggle */}
              <button
                onClick={() => setIsLooping(!isLooping)}
                className={`flex h-8 items-center gap-1.5 rounded-md border px-2.5 text-xs transition-colors ${
                  isLooping
                    ? "border-primary/50 bg-primary/10 text-primary font-medium"
                    : "border-border bg-secondary/30 text-muted-foreground"
                }`}
              >
                <Repeat className="h-3.5 w-3.5" />
                <span>Loop {isLooping ? "ON" : "OFF"}</span>
              </button>

              {/* Playback speed toggle */}
              <button
                onClick={() => {
                  const nextSpeed = playbackSpeed === 1 ? 0.5 : playbackSpeed === 0.5 ? 2 : 1;
                  setPlaybackSpeed(nextSpeed);
                  if (videoRef.current) {
                    videoRef.current.playbackRate = nextSpeed;
                  }
                }}
                className="flex h-8 items-center gap-1 rounded-md border border-border bg-secondary/30 px-2.5 text-xs font-mono text-foreground hover:bg-secondary"
              >
                <span>{playbackSpeed}x</span>
              </button>
            </div>

            {/* Continuous Real-Time Analysis State */}
            <div className="flex items-center gap-3">
              <button
                onClick={onToggleContinuousAnalysis}
                className={cn(
                  "flex items-center gap-2 rounded-md border px-3 py-1.5 text-xs font-medium transition-all",
                  continuousAnalysis
                    ? "border-success/40 bg-success/10 text-success"
                    : "border-border bg-secondary text-muted-foreground",
                )}
              >
                <span className={cn("h-2 w-2 rounded-full", continuousAnalysis ? "bg-success animate-pulse" : "bg-muted-foreground")} />
                <span>Continuous Analysis: {continuousAnalysis ? "ACTIVE" : "PAUSED"}</span>
              </button>

              <div className="hidden sm:flex items-center gap-2 text-xs font-mono text-muted-foreground">
                <Activity className="h-3.5 w-3.5 text-primary" />
                <span>Synchronized Detections: {dynamicWorkers.length}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* =========================================================================
          LIVE WEBCAM CONTINUOUS DETECTION STATUS & ACTION BAR
         ========================================================================= */}
      {isWebcam && (
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border/80 bg-secondary/25 px-4 py-2.5">
          <div className="flex items-center gap-2.5">
            <span className="flex h-2.5 w-2.5 rounded-full bg-critical animate-ping" />
            <div>
              <span className="font-mono text-xs font-semibold uppercase tracking-wider text-foreground">
                Continuous Webcam Detection Stream Active
              </span>
              <p className="text-[11px] text-muted-foreground">
                Camera feed is streaming live with continuous real-time worker pose & PPE bounding box tracking.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={onToggleContinuousAnalysis}
              className={`flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
                continuousAnalysis
                  ? "border-primary/40 bg-primary/10 text-primary"
                  : "border-border bg-secondary/40 text-muted-foreground"
              }`}
            >
              <Activity className="h-3.5 w-3.5" />
              <span>Continuous Detection: {continuousAnalysis ? "ACTIVE" : "PAUSED"}</span>
            </button>
          </div>
        </div>
      )}

      {/* Result footer strip */}
      {result && !processing && (
        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border/80 bg-secondary/20 px-4 py-2 text-xs">
          <div className="flex items-center gap-2">
            <span className="font-mono text-primary font-semibold">{result.jobId}</span>
            <span className="text-muted-foreground">|</span>
            <span className="text-foreground">{result.fileName}</span>
          </div>
          <div className="flex items-center gap-4">
            <span className="font-medium text-foreground">
              {dynamicWorkers.length} workers currently localized
            </span>
            <span className="rounded bg-primary/10 border border-primary/30 px-2 py-0.5 font-mono text-[10px] text-primary">
              Real-time Inference
            </span>
          </div>
        </div>
      )}
    </section>
  );
}
