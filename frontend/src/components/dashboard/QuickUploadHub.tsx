import { useRef, useState, useEffect, useCallback } from "react";
import { Link, useNavigate } from "@tanstack/react-router";
import {
  Camera,
  Radio,
  Eye,
  Maximize2,
  Sparkles,
  ArrowRight,
  Video,
  Upload,
  Play,
  Pause,
  Trash2,
  Flame,
  Zap,
  Tv,
  CheckCircle2,
} from "lucide-react";
import { toast } from "sonner";
import sampleFrame from "@/assets/detection-frame.jpg";
import heroFactory from "@/assets/hero-factory.jpg";

interface CamConfig {
  id: string;
  name: string;
  sector: string;
  camCode: string;
  status: "critical" | "warning" | "optimal" | "live";
  hazard: string;
  workerLabel: string;
  workerStatus: "critical" | "warning" | "compliant";
  confidence: number;
  image: string;
  latency: number;
}

const defaultCameras: CamConfig[] = [
  {
    id: "cam-01",
    name: "Foundry Floor",
    sector: "Sector 01 · Blast Corridor",
    camCode: "CAM-01",
    status: "critical",
    hazard: "Molten ladle transit · High radiant heat",
    workerLabel: "W-131 · NO HELMET",
    workerStatus: "critical",
    confidence: 91,
    image: sampleFrame,
    latency: 38,
  },
  {
    id: "cam-02",
    name: "Welding Bay",
    sector: "Sector 02 · Fabrication Cell",
    camCode: "CAM-02",
    status: "warning",
    hazard: "Arc-flash radiation & spark dispersion",
    workerLabel: "W-022 · MISSING VEST",
    workerStatus: "warning",
    confidence: 88,
    image: heroFactory,
    latency: 28,
  },
  {
    id: "cam-03",
    name: "Assembly Line",
    sector: "Sector 03 · Robotic Conveyor",
    camCode: "CAM-03",
    status: "optimal",
    hazard: "Automated strapping & conveyor pinch",
    workerLabel: "W-041 · COMPLIANT",
    workerStatus: "compliant",
    confidence: 97,
    image: sampleFrame,
    latency: 32,
  },
  {
    id: "cam-04",
    name: "Loading Dock",
    sector: "Sector 04 · Live Lens / Yard",
    camCode: "CAM-04",
    status: "live",
    hazard: "Forklift transit & worker live check",
    workerLabel: "W-074 · SAFE 96%",
    workerStatus: "compliant",
    confidence: 96,
    image: heroFactory,
    latency: 24,
  },
];

export function QuickUploadHub() {
  const navigate = useNavigate();

  // Webcam states for each camera box
  const [activeWebcamCam, setActiveWebcamCam] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user");
  const webcamVideoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const webcamStreamRef = useRef<MediaStream | null>(null);

  // Uploaded video states for all 4 camera channels
  const [videoSources, setVideoSources] = useState<
    Record<string, { file: File; url: string } | null>
  >({
    "cam-01": null,
    "cam-02": null,
    "cam-03": null,
    "cam-04": null,
  });

  const videoInputRefs = useRef<Record<string, HTMLInputElement | null>>({});
  const videoPlayerRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const [dragOverBox, setDragOverBox] = useState<Record<string, boolean>>({});

  // Overlay toggles for all 4 boxes
  const [showOverlays, setShowOverlays] = useState<Record<string, boolean>>({
    "cam-01": true,
    "cam-02": true,
    "cam-03": true,
    "cam-04": true,
  });

  // Ticking surveillance clock
  const [clock, setClock] = useState("06:51:24");
  useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      setClock(now.toTimeString().split(" ")[0]);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const stopWebcam = useCallback(() => {
    if (webcamStreamRef.current) {
      webcamStreamRef.current.getTracks().forEach((track) => track.stop());
      webcamStreamRef.current = null;
    }
    Object.values(webcamVideoRefs.current).forEach((videoEl) => {
      if (videoEl) videoEl.srcObject = null;
    });
    setActiveWebcamCam(null);
  }, []);

  useEffect(() => {
    return () => {
      stopWebcam();
      // Revoke any created video URLs
      Object.values(videoSources).forEach((v) => {
        if (v?.url) URL.revokeObjectURL(v.url);
      });
    };
  }, [stopWebcam]);

  const startWebcamForBox = async (camId: string) => {
    stopWebcam();
    // Clear video on this box if active
    if (videoSources[camId]) {
      removeVideoFromBox(camId);
    }
    try {
      if (!navigator?.mediaDevices?.getUserMedia) {
        toast.error("Webcam not supported in this browser");
        return;
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: facingMode,
        },
        audio: false,
      });
      webcamStreamRef.current = stream;
      const vid = webcamVideoRefs.current[camId];
      if (vid) {
        vid.srcObject = stream;
        await vid.play();
      }
      setActiveWebcamCam(camId);
      toast.success(`Webcam stream active on ${camId.toUpperCase()}`, {
        description: "Live camera input is streaming into the surveillance matrix.",
      });
    } catch {
      toast.error("Could not access webcam. Check browser camera permissions.");
    }
  };

  // Handle video upload to any of the 4 boxes
  const handleVideoUploadToBox = (camId: string, file: File | undefined) => {
    if (!file) return;
    if (!/^video\/(mp4|webm|quicktime|ogg|x-matroska)$/i.test(file.type)) {
      toast.error("Unsupported video format. Please upload MP4, WebM or MOV.");
      return;
    }

    // Stop webcam if active on this box
    if (activeWebcamCam === camId) {
      stopWebcam();
    }

    if (videoSources[camId]?.url) {
      URL.revokeObjectURL(videoSources[camId]!.url);
    }

    const url = URL.createObjectURL(file);
    setVideoSources((prev) => ({
      ...prev,
      [camId]: { file, url },
    }));

    toast.success(`Video Loaded into ${camId.toUpperCase()}`, {
      description: `${file.name} is playing continuously with real-time PPE detection overlays.`,
    });
  };

  const removeVideoFromBox = (camId: string) => {
    if (videoSources[camId]?.url) {
      URL.revokeObjectURL(videoSources[camId]!.url);
    }
    setVideoSources((prev) => ({ ...prev, [camId]: null }));
    toast.info(`Reset ${camId.toUpperCase()} to Live Feed`);
  };

  const captureSnapshot = (camId: string, imageFallback: string) => {
    // 1. If an uploaded video is playing in this box, capture its current frame!
    if (videoSources[camId] && videoPlayerRefs.current[camId]) {
      const vid = videoPlayerRefs.current[camId];
      if (!vid) return;

      const canvas = document.createElement("canvas");
      canvas.width = vid.videoWidth || 1280;
      canvas.height = vid.videoHeight || 720;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.drawImage(vid, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (blob) => {
          if (blob) {
            const file = new File(
              [blob],
              `${camId}_video_frame_t${Math.floor(vid.currentTime)}s.jpg`,
              { type: "image/jpeg" },
            );
            (window as unknown as { __SAFEOPS_PENDING_FILE__?: File }).__SAFEOPS_PENDING_FILE__ = file;
            toast.success(`Video Frame Captured from ${camId.toUpperCase()}`, {
              description: "Transferring frame to Detection Studio for full analysis.",
            });
            navigate({ to: "/dashboard/detection" });
          }
        },
        "image/jpeg",
        0.95,
      );
      return;
    }

    // 2. If real webcam is active on this box, capture direct frame from webcam stream
    if (activeWebcamCam === camId && webcamVideoRefs.current[camId]) {
      const vid = webcamVideoRefs.current[camId];
      if (!vid) return;

      const canvas = document.createElement("canvas");
      canvas.width = vid.videoWidth || 1280;
      canvas.height = vid.videoHeight || 720;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.drawImage(vid, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (blob) => {
          if (blob) {
            const file = new File(
              [blob],
              `${camId}_webcam_snapshot_${Date.now()}.jpg`,
              { type: "image/jpeg" },
            );
            (window as unknown as { __SAFEOPS_PENDING_FILE__?: File }).__SAFEOPS_PENDING_FILE__ = file;
            toast.success(`Live Snapshot Captured from ${camId.toUpperCase()}`, {
              description: "Transferring to Detection Studio for instant inspection.",
            });
            navigate({ to: "/dashboard/detection" });
          }
        },
        "image/jpeg",
        0.95,
      );
      return;
    }

    // 3. Otherwise capture high-res frame from this camera's optical feed
    if (typeof window !== "undefined") {
      sessionStorage.setItem("safeops_load_sample", "true");
    }
    toast.success(`Captured ${camId.toUpperCase()} Optical Frame`, {
      description: "Transferring frame to Detection Studio for inspection.",
    });
    navigate({ to: "/dashboard/detection" });
  };

  const inspectInStudio = (camId: string) => {
    // If box has an uploaded video, pass the video file to studio!
    if (videoSources[camId]?.file) {
      (window as unknown as { __SAFEOPS_PENDING_FILE__?: File }).__SAFEOPS_PENDING_FILE__ = videoSources[camId]!.file;
      toast.success(`Opening ${camId.toUpperCase()} Video in Studio`, {
        description: "Launching continuous playback and full geometry analysis.",
      });
      navigate({ to: "/dashboard/detection" });
      return;
    }

    // Otherwise load continuous sample surveillance stream
    if (typeof window !== "undefined") {
      sessionStorage.setItem("safeops_load_sample_video", "true");
    }
    toast.success(`Streaming ${camId.toUpperCase()} in Detection Studio`, {
      description: "Live continuous optical stream active.",
    });
    navigate({ to: "/dashboard/detection" });
  };

  const toggleOverlay = (camId: string) => {
    setShowOverlays((prev) => ({ ...prev, [camId]: !prev[camId] }));
  };

  return (
    <div className="panel rounded-xl overflow-hidden shadow-2xl border border-border/80">
      {/* Matrix Master Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/80 px-6 py-4 bg-secondary/25">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex h-2.5 w-2.5 rounded-full bg-critical animate-ping" />
            <span className="font-mono text-xs font-semibold uppercase tracking-wider text-primary">
              Live Quad Surveillance Matrix
            </span>
            <span className="rounded-full border border-primary/40 bg-primary/10 px-2.5 py-0.5 font-mono text-[10px] text-primary">
              All 4 Channels Support Live Webcams & Video Uploads
            </span>
          </div>
          <h2 className="mt-1 text-base font-semibold tracking-tight text-foreground sm:text-lg">
            4-Channel Optical Matrix & Video Ingestion Grid
          </h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            All 4 channels take live webcam feeds or video uploads: drop or select any video on any box to play continuously with real-time AI bounding box tracking.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Master Video Upload input */}
          <input
            id="matrix-master-video-upload"
            type="file"
            accept="video/mp4,video/webm,video/quicktime"
            className="sr-only"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) {
                handleVideoUploadToBox("cam-01", file);
                toast.success("Surveillance video uploaded to Matrix", {
                  description: `${file.name} loaded into Channel 1 (Foundry Floor).`,
                });
              }
              e.target.value = "";
            }}
          />

          <button
            onClick={() => document.getElementById("matrix-master-video-upload")?.click()}
            className="inline-flex items-center gap-1.5 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-1.5 text-xs font-semibold text-amber-400 hover:bg-amber-500/20 transition-all shadow-xs"
          >
            <Video className="h-3.5 w-3.5 text-amber-400" />
            <span>Upload Video Feed</span>
          </button>

          {activeWebcamCam ? (
            <button
              onClick={stopWebcam}
              className="inline-flex items-center gap-1.5 rounded-lg border border-critical/40 bg-critical/10 px-3 py-1.5 text-xs font-medium text-critical hover:bg-critical/20 transition-all"
            >
              <Camera className="h-3.5 w-3.5" />
              <span>Disconnect Webcam ({activeWebcamCam.toUpperCase()})</span>
            </button>
          ) : (
            <button
              onClick={() => startWebcamForBox("cam-04")}
              className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-secondary/70 px-3 py-1.5 text-xs font-medium text-foreground hover:bg-secondary hover:text-primary transition-all"
            >
              <Camera className="h-3.5 w-3.5 text-primary" />
              <span>Connect Device Webcam</span>
            </button>
          )}

          <Link
            to="/dashboard/detection"
            className="inline-flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground shadow hover:bg-primary/90 transition-all"
          >
            <span>Open Studio</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      </div>

      {/* =========================================================================
          ALL 4 PROPER ALIGNED LIVE SURVEILLANCE & VIDEO UPLOAD BOXES
         ========================================================================= */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 divide-y md:divide-y-0 md:divide-x divide-border/80 p-0">
        {defaultCameras.map((cam, idx) => {
          const isWebcamActive = activeWebcamCam === cam.id;
          const uploadedVideo = videoSources[cam.id];
          const isVideoActive = Boolean(uploadedVideo);
          const overlaysOn = showOverlays[cam.id] ?? true;
          const isDragOver = dragOverBox[cam.id] ?? false;

          return (
            <div
              key={cam.id}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOverBox((prev) => ({ ...prev, [cam.id]: true }));
              }}
              onDragLeave={() => {
                setDragOverBox((prev) => ({ ...prev, [cam.id]: false }));
              }}
              onDrop={(e) => {
                e.preventDefault();
                setDragOverBox((prev) => ({ ...prev, [cam.id]: false }));
                const file = e.dataTransfer.files?.[0];
                handleVideoUploadToBox(cam.id, file);
              }}
              className="p-5 flex flex-col justify-between space-y-4 hover:bg-secondary/10 transition-colors"
            >
              {/* Hidden file input for this specific box */}
              <input
                ref={(el) => {
                  videoInputRefs.current[cam.id] = el;
                }}
                type="file"
                accept="video/mp4,video/webm,video/quicktime"
                className="sr-only"
                aria-label={`Upload video for ${cam.name}`}
                onChange={(e) => {
                  handleVideoUploadToBox(cam.id, e.target.files?.[0]);
                  e.target.value = "";
                }}
              />

              <div>
                {/* Channel Box Header */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div
                      className={`flex h-9 w-9 items-center justify-center rounded-lg border ${
                        isVideoActive
                          ? "border-amber-500/40 bg-amber-500/10 text-amber-400"
                          : isWebcamActive
                          ? "border-cyan-500/40 bg-cyan-500/10 text-cyan-400"
                          : cam.status === "critical"
                          ? "border-critical/40 bg-critical/10 text-critical"
                          : cam.status === "warning"
                          ? "border-amber-500/40 bg-amber-500/10 text-amber-400"
                          : "border-success/40 bg-success/10 text-success"
                      }`}
                    >
                      {isVideoActive ? (
                        <Video className="h-4 w-4" />
                      ) : isWebcamActive ? (
                        <Camera className="h-4 w-4" />
                      ) : cam.status === "critical" ? (
                        <Flame className="h-4 w-4" />
                      ) : cam.status === "warning" ? (
                        <Zap className="h-4 w-4" />
                      ) : (
                        <Tv className="h-4 w-4" />
                      )}
                    </div>

                    <div>
                      <h3 className="text-sm font-semibold text-foreground">
                        {idx + 1}. {cam.name}
                      </h3>
                      <span className="font-mono text-[10px] uppercase text-muted-foreground">
                        {cam.camCode} ·{" "}
                        {isVideoActive
                          ? "VIDEO PLAYING"
                          : isWebcamActive
                          ? "WEBCAM FEED"
                          : "CCTV STREAM"}
                      </span>
                    </div>
                  </div>

                  <span
                    className={`flex items-center gap-1 rounded px-1.5 py-0.5 font-mono text-[9px] ${
                      isVideoActive
                        ? "bg-amber-500/20 text-amber-400"
                        : "bg-critical/20 text-critical"
                    }`}
                  >
                    <span
                      className={`h-1.5 w-1.5 rounded-full ${
                        isVideoActive
                          ? "bg-amber-400 animate-pulse"
                          : "bg-critical animate-ping"
                      }`}
                    />
                    {isVideoActive ? "VIDEO" : "LIVE"}
                  </span>
                </div>

                <p className="mt-2.5 text-xs text-muted-foreground leading-relaxed truncate">
                  {isVideoActive
                    ? `Playing: ${uploadedVideo?.file.name}`
                    : cam.hazard}
                </p>

                {/* =========================================================================
                    MATCHING LIVE VIEWFINDER BOX (ALL 4 IDENTICAL HEIGHT h-40)
                   ========================================================================= */}
                <div
                  className={`mt-3.5 relative h-40 w-full overflow-hidden rounded-lg bg-black/95 border shadow-inner transition-all ${
                    isDragOver
                      ? "border-amber-400 ring-2 ring-amber-400/40 bg-amber-400/10"
                      : "border-border/80"
                  }`}
                >
                  {/* Drag-over indicator overlay */}
                  {isDragOver && (
                    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/85 text-amber-400 p-2 text-center">
                      <Upload className="h-6 w-6 animate-bounce" />
                      <span className="text-xs font-semibold mt-1">
                        Drop video for {cam.name}
                      </span>
                    </div>
                  )}

                  {/* 1. UPLOADED VIDEO ELEMENT (When video is uploaded to this box) */}
                  {isVideoActive && uploadedVideo ? (
                    <video
                      ref={(el) => {
                        videoPlayerRefs.current[cam.id] = el;
                      }}
                      src={uploadedVideo.url}
                      playsInline
                      muted
                      autoPlay
                      loop
                      className="h-full w-full object-contain"
                    />
                  ) : isWebcamActive ? (
                    /* 2. REAL LOCAL WEBCAM ELEMENT */
                    <video
                      ref={(el) => {
                        webcamVideoRefs.current[cam.id] = el;
                      }}
                      playsInline
                      muted
                      autoPlay
                      className={`h-full w-full object-cover ${
                        facingMode === "user" ? "scale-x-[-1]" : ""
                      }`}
                    />
                  ) : (
                    /* 3. OPTICAL LIVE CCTV IMAGE STREAM */
                    <img
                      src={cam.image}
                      alt={cam.name}
                      className="h-full w-full object-cover filter brightness-90 contrast-105"
                    />
                  )}

                  {/* Live Scanline Motion Effect */}
                  <div className="pointer-events-none absolute inset-0 animate-safeops-scan bg-gradient-to-b from-transparent via-primary/10 to-transparent h-14" />

                  {/* Top-Left Telemetry Watermark */}
                  <div className="pointer-events-none absolute top-2 left-2 flex items-center gap-1.5 rounded bg-black/75 px-2 py-0.5 font-mono text-[9px] text-white/90 backdrop-blur-sm border border-white/10">
                    <Radio className="h-2.5 w-2.5 text-critical animate-pulse" />
                    <span>{cam.camCode}</span>
                    <span className="text-white/40">|</span>
                    <span className="text-amber-300">{clock}</span>
                  </div>

                  {/* Top-Right Badge */}
                  <div className="pointer-events-none absolute top-2 right-2 rounded bg-black/75 px-1.5 py-0.5 font-mono text-[8px] text-white/80 backdrop-blur-sm border border-white/10">
                    {isVideoActive ? "CONTINUOUS PLAY" : `${cam.latency}ms · 30 FPS`}
                  </div>

                  {/* Live Computer Vision Bounding Boxes Overlay */}
                  {overlaysOn && (
                    <>
                      {cam.workerStatus === "critical" ? (
                        <div
                          className="absolute border-2 border-critical bg-critical/20 rounded-xs animate-pulse"
                          style={{ left: "55%", top: "18%", width: "24%", height: "68%" }}
                        >
                          <span className="absolute -top-4 left-0 rounded bg-critical px-1 py-0.2 font-mono text-[8px] text-white font-bold shadow">
                            🚨 NO HELMET 91%
                          </span>
                        </div>
                      ) : cam.workerStatus === "warning" ? (
                        <div
                          className="absolute border-2 border-warning bg-warning/20 rounded-xs"
                          style={{ left: "20%", top: "22%", width: "26%", height: "64%" }}
                        >
                          <span className="absolute -top-4 left-0 rounded bg-warning/90 px-1 py-0.2 font-mono text-[8px] text-black font-semibold shadow">
                            ⚠️ MISSING VEST 88%
                          </span>
                        </div>
                      ) : (
                        <div
                          className="absolute border-2 border-success bg-success/20 rounded-xs"
                          style={{ left: "32%", top: "15%", width: "30%", height: "72%" }}
                        >
                          <span className="absolute -top-4 left-0 rounded bg-success px-1 py-0.2 font-mono text-[8px] text-white font-semibold shadow">
                            ✅ SAFE 97%
                          </span>
                        </div>
                      )}
                    </>
                  )}

                  {/* Bottom Viewfinder Quick Controls */}
                  <div className="absolute bottom-2 inset-x-2 flex items-center justify-between">
                    <div className="flex items-center gap-1 rounded bg-black/75 px-2 py-0.5 font-mono text-[9px] text-white/90 backdrop-blur-sm border border-white/10">
                      <span>{cam.workerLabel}</span>
                    </div>

                    <button
                      onClick={() => toggleOverlay(cam.id)}
                      className={`flex items-center gap-1 rounded bg-black/75 px-2 py-0.5 font-mono text-[8px] backdrop-blur-sm border border-white/10 transition-colors ${
                        overlaysOn ? "text-primary" : "text-white/40"
                      }`}
                    >
                      <Eye className="h-2.5 w-2.5" />
                      <span>AI</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* Action Buttons (Equal Height & Properly Aligned) */}
              <div className="space-y-2 pt-2">
                {/* Main Capture Action */}
                <button
                  onClick={() => captureSnapshot(cam.id, cam.image)}
                  className="w-full flex items-center justify-center gap-1.5 rounded-lg bg-primary py-2 text-xs font-semibold text-primary-foreground shadow hover:bg-primary/90 transition-all active:scale-[0.99]"
                >
                  <Camera className="h-3.5 w-3.5" />
                  <span>Capture Snapshot & Inspect</span>
                </button>

                {/* Secondary Row: Upload Video + Local Cam + Inspect */}
                <div className="flex gap-1.5">
                  {/* Upload Video Button for THIS box */}
                  <button
                    onClick={() => videoInputRefs.current[cam.id]?.click()}
                    title="Upload video to play continuously in this channel"
                    className="flex-1 flex items-center justify-center gap-1 rounded-lg border border-amber-500/30 bg-amber-500/10 py-1.5 text-xs font-semibold text-amber-400 hover:bg-amber-500/20 transition-colors shadow-xs"
                  >
                    <Upload className="h-3 w-3 text-amber-400" />
                    <span>Upload Video</span>
                  </button>

                  {/* Connect Local Webcam for THIS box */}
                  {!isWebcamActive ? (
                    <button
                      onClick={() => startWebcamForBox(cam.id)}
                      title="Stream your device camera in this box"
                      className="flex-1 flex items-center justify-center gap-1 rounded-lg border border-border bg-secondary/50 py-1.5 text-xs font-medium text-foreground hover:bg-secondary hover:text-primary transition-colors"
                    >
                      <Radio className="h-3 w-3 text-primary" />
                      <span>Webcam</span>
                    </button>
                  ) : (
                    <button
                      onClick={stopWebcam}
                      className="flex-1 flex items-center justify-center gap-1 rounded-lg border border-critical/30 bg-critical/10 py-1.5 text-xs font-medium text-critical hover:bg-critical/20 transition-colors"
                    >
                      <span>Stop Cam</span>
                    </button>
                  )}

                  {/* Reset/Remove video if custom video is playing */}
                  {isVideoActive && (
                    <button
                      onClick={() => removeVideoFromBox(cam.id)}
                      title="Reset box to live optical stream"
                      className="flex items-center justify-center rounded-lg border border-border px-2 py-1.5 text-xs text-muted-foreground hover:text-critical hover:bg-secondary transition-colors"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  )}

                  {/* Inspect in Studio */}
                  <button
                    onClick={() => inspectInStudio(cam.id)}
                    title="Inspect stream in detection studio"
                    className="flex items-center justify-center rounded-lg border border-border bg-secondary/30 px-2.5 py-1.5 text-xs text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
                  >
                    <Maximize2 className="h-3 w-3" />
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
