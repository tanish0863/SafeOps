import { useRef, useState, useEffect, useCallback } from "react";
import {
  ImagePlus,
  RefreshCw,
  Trash2,
  Upload,
  Video,
  Camera,
  RotateCw,
  Sparkles,
  AlertCircle,
  CheckCircle2,
  Radio,
  Play,
  Layers,
} from "lucide-react";
import { toast } from "sonner";

import { TechLabel } from "@/components/common/TechLabel";
import type { WorkspaceImage, WorkspaceVideo, MediaType } from "@/hooks/useDetectionFlow";
import type { DetectionWorkspaceState } from "@/hooks/useDetectionFlow";
import { cn } from "@/lib/utils";

type UploadMode = "image" | "video" | "webcam";

interface ImageUploaderProps {
  state: DetectionWorkspaceState;
  image: WorkspaceImage | null;
  video?: WorkspaceVideo | null;
  webcamStream?: MediaStream | null;
  facingMode?: "user" | "environment";
  activeMediaType?: MediaType;
  onSelectMode?: (mode: MediaType) => void;
  onFile: (file: File, sourceType?: MediaType) => void;
  onVideoFile?: (file: File) => void;
  onStartWebcam?: () => void;
  onStopWebcam?: () => void;
  onToggleFacingMode?: () => void;
  onSample: () => void;
  onSampleVideo?: () => void;
  onRemove: () => void;
  onAnalyze?: () => void;
}

export function ImageUploader({
  state,
  image,
  video,
  webcamStream,
  facingMode = "user",
  activeMediaType = "image",
  onSelectMode,
  onFile,
  onVideoFile,
  onStartWebcam,
  onStopWebcam,
  onToggleFacingMode,
  onSample,
  onSampleVideo,
  onRemove,
  onAnalyze: _onAnalyze,
}: ImageUploaderProps) {
  const [mode, setMode] = useState<UploadMode>((activeMediaType as UploadMode) || "image");
  const [dragOver, setDragOver] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  // File input refs
  const imageInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);

  const processing = state === "processing";
  const isWebcamLive = Boolean(webcamStream);

  // Synchronize webcam control with parent detection flow
  const stopWebcam = useCallback(() => {
    onStopWebcam?.();
  }, [onStopWebcam]);

  const startWebcam = useCallback(() => {
    onStartWebcam?.();
  }, [onStartWebcam]);

  const toggleFacingMode = useCallback(() => {
    onToggleFacingMode?.();
  }, [onToggleFacingMode]);

  // Image file accept
  const acceptImage = (file: File | undefined) => {
    if (!file) return;
    if (!/^image\/(jpeg|jpg|png|webp)$/i.test(file.type)) {
      setLocalError("JPG, PNG or WebP only.");
      return;
    }
    setLocalError(null);
    onFile(file, "image");
  };

  // Video file accept
  const acceptVideo = (file: File | undefined) => {
    if (!file) return;
    if (!/^video\/(mp4|webm|quicktime|ogg|x-matroska)$/i.test(file.type) && !/\.(mp4|webm|mov|mkv|ogg)$/i.test(file.name)) {
      setLocalError("MP4, WebM or MOV videos only.");
      return;
    }
    setLocalError(null);
    if (onVideoFile) {
      onVideoFile(file);
      toast.success("Continuous video stream loaded", {
        description: `${file.name} is playing continuously and being analyzed in the canvas.`,
      });
    }
  };

  const onDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files?.[0];
    if (!file) return;

    const isVideo =
      /^video\//i.test(file.type) || /\.(mp4|webm|mov|mkv|ogg)$/i.test(file.name);
    const isImage =
      /^image\//i.test(file.type) || /\.(jpg|jpeg|png|webp)$/i.test(file.name);

    if (isVideo) {
      stopWebcam();
      setMode("video");
      onSelectMode?.("video");
      acceptVideo(file);
    } else if (isImage) {
      stopWebcam();
      setMode("image");
      onSelectMode?.("image");
      acceptImage(file);
    } else {
      setLocalError("Please upload an image (JPG, PNG, WebP) or video (MP4, WebM, MOV).");
    }
  };

  return (
    <section
      className="panel rounded-xl overflow-hidden shadow-xl"
      onDragOver={(event) => {
        event.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
    >
      {/* Hidden file inputs */}
      <input
        ref={imageInputRef}
        type="file"
        accept="image/jpeg,image/jpg,image/png,image/webp,video/mp4,video/webm,video/quicktime"
        className="sr-only"
        aria-label="Upload an image or video"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) {
            if (/^video\//i.test(file.type) || /\.(mp4|webm|mov|mkv)$/i.test(file.name)) {
              stopWebcam();
              setMode("video");
              onSelectMode?.("video");
              acceptVideo(file);
            } else {
              acceptImage(file);
            }
          }
          event.target.value = "";
        }}
      />
      <input
        ref={videoInputRef}
        type="file"
        accept="video/mp4,video/webm,video/quicktime,video/ogg"
        className="sr-only"
        aria-label="Upload a video"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) {
            stopWebcam();
            setMode("video");
            onSelectMode?.("video");
            acceptVideo(file);
          }
          event.target.value = "";
        }}
      />

      {/* 3 Upload Mode Selection Tabs */}
      <div className="flex border-b border-border/80 bg-secondary/30 p-1.5 gap-1">
        <button
          onClick={() => {
            setMode("image");
            onSelectMode?.("image");
            stopWebcam();
          }}
          className={cn(
            "flex flex-1 items-center justify-center gap-1.5 rounded-lg py-2 text-xs font-semibold transition-all",
            mode === "image"
              ? "bg-primary text-primary-foreground shadow-sm"
              : "text-muted-foreground hover:bg-secondary hover:text-foreground",
          )}
        >
          <ImagePlus className="h-3.5 w-3.5" />
          <span>Image</span>
        </button>

        <button
          onClick={() => {
            setMode("video");
            onSelectMode?.("video");
            stopWebcam();
          }}
          className={cn(
            "flex flex-1 items-center justify-center gap-1.5 rounded-lg py-2 text-xs font-semibold transition-all",
            mode === "video"
              ? "bg-primary text-primary-foreground shadow-sm"
              : "text-muted-foreground hover:bg-secondary hover:text-foreground",
          )}
        >
          <Video className="h-3.5 w-3.5" />
          <span>Video</span>
        </button>

        <button
          onClick={() => {
            setMode("webcam");
            onSelectMode?.("webcam");
          }}
          className={cn(
            "flex flex-1 items-center justify-center gap-1.5 rounded-lg py-2 text-xs font-semibold transition-all",
            mode === "webcam"
              ? "bg-primary text-primary-foreground shadow-sm"
              : "text-muted-foreground hover:bg-secondary hover:text-foreground",
          )}
        >
          <Camera className="h-3.5 w-3.5" />
          <span>Webcam</span>
        </button>
      </div>

      {/* Subheader info line */}
      <header className="flex items-center justify-between border-b border-border/60 px-4 py-2 text-[11px] text-muted-foreground bg-secondary/10">
        <TechLabel>
          {mode === "image" && "Still Camera Frame or Video"}
          {mode === "video" && "Continuous Video Stream"}
          {mode === "webcam" && "Live Webcam Feed & Video Intake"}
        </TechLabel>
        <span className="font-mono text-[10px] uppercase">
          {mode === "image" && "JPG · PNG · MP4 · WEBM"}
          {mode === "video" && "MP4 · WebM · MOV"}
          {mode === "webcam" && (isWebcamLive ? "Online" : "Ready")}
        </span>
      </header>

      {localError && (
        <div className="flex items-center gap-2 border-b border-critical/30 bg-critical/10 px-4 py-2 text-xs text-critical">
          <AlertCircle className="h-3.5 w-3.5 shrink-0" />
          <span>{localError}</span>
        </div>
      )}

      {/* =========================================================================
          TAB 1: IMAGE (ALSO TAKES VIDEO UPLOAD)
         ========================================================================= */}
      {mode === "image" && (
        <div>
          {image ? (
            <div className="space-y-3 p-4">
              <div className="flex items-center gap-3">
                <img
                  src={image.url}
                  alt="Uploaded frame thumbnail"
                  className="h-16 w-20 shrink-0 rounded-md border border-border object-cover"
                />
                <div className="min-w-0 flex-1">
                  <p className="truncate font-mono text-xs font-medium text-foreground">
                    {image.file.name}
                  </p>
                  <p className="mt-1 text-[11px] text-muted-foreground">
                    {(image.file.size / 1024).toFixed(0)} KB · {image.uploadedAt}
                  </p>
                  <span className="mt-1 inline-flex items-center gap-1 rounded bg-secondary px-1.5 py-0.5 text-[10px] text-foreground">
                    <CheckCircle2 className="h-3 w-3 text-success" /> Loaded into canvas
                  </span>
                </div>
              </div>

              <div className="flex gap-2 pt-1">
                <button
                  onClick={() => imageInputRef.current?.click()}
                  disabled={processing}
                  className="inline-flex h-8 flex-1 items-center justify-center gap-2 rounded-md border border-border bg-secondary/40 text-xs font-medium text-foreground transition-colors hover:bg-secondary disabled:opacity-50"
                >
                  <RefreshCw className="h-3 w-3" />
                  Replace
                </button>
                <button
                  onClick={() => {
                    videoInputRef.current?.click();
                  }}
                  className="inline-flex h-8 flex-1 items-center justify-center gap-1.5 rounded-md border border-amber-500/40 bg-amber-500/10 text-xs font-medium text-amber-400 transition-colors hover:bg-amber-500/20"
                >
                  <Video className="h-3.5 w-3.5" />
                  Upload Video
                </button>
                <button
                  onClick={onRemove}
                  disabled={processing}
                  className="inline-flex h-8 items-center justify-center gap-1.5 rounded-md border border-border px-3 text-xs text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground disabled:opacity-50"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>

              <p
                className={cn(
                  "rounded-md border border-dashed border-border/80 px-3 py-2 text-center text-xs text-muted-foreground",
                  dragOver && "border-primary text-primary bg-primary/5",
                )}
              >
                Drop replacement image or video here
              </p>
            </div>
          ) : (
            <div className="p-4 space-y-3">
              {/* Primary Dual Drop Zone */}
              <button
                onClick={() => imageInputRef.current?.click()}
                className={cn(
                  "flex h-36 w-full flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-border/80 bg-secondary/15 transition-all hover:border-primary/60 hover:bg-primary/5",
                  dragOver && "border-primary bg-primary/10",
                )}
              >
                <div className="flex items-center gap-2">
                  <ImagePlus className="h-6 w-6 text-primary" />
                  <span className="text-muted-foreground text-xs font-mono">/</span>
                  <Video className="h-6 w-6 text-amber-400" />
                </div>
                <span className="text-xs font-medium text-foreground">
                  Drop image or video here, or browse
                </span>
                <span className="text-[11px] text-muted-foreground">
                  Accepts still photos or continuous video streams
                </span>
              </button>

              {/* Upload Actions (Both Image & Video) */}
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => imageInputRef.current?.click()}
                  className="flex items-center justify-center gap-1.5 rounded-lg border border-border bg-secondary/50 py-2 text-xs font-medium text-foreground hover:bg-secondary transition-colors"
                >
                  <Upload className="h-3.5 w-3.5" />
                  Upload Image
                </button>
                <button
                  onClick={() => videoInputRef.current?.click()}
                  className="flex items-center justify-center gap-1.5 rounded-lg border border-amber-500/40 bg-amber-500/10 py-2 text-xs font-medium text-amber-400 hover:bg-amber-500/20 transition-colors shadow-sm"
                >
                  <Video className="h-3.5 w-3.5" />
                  Upload Video
                </button>
              </div>

              {/* Sample Presets */}
              <div className="flex gap-2">
                <button
                  onClick={onSample}
                  className="flex-1 flex items-center justify-center gap-1.5 rounded-lg border border-primary/40 bg-primary/10 py-1.5 text-xs font-medium text-primary hover:bg-primary/20 transition-colors"
                >
                  <Sparkles className="h-3 w-3" />
                  Sample Image
                </button>
                {onSampleVideo && (
                  <button
                    onClick={onSampleVideo}
                    className="flex-1 flex items-center justify-center gap-1.5 rounded-lg border border-border bg-secondary/40 py-1.5 text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
                  >
                    <Play className="h-3 w-3 text-amber-400" />
                    Sample Video
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* =========================================================================
          TAB 2: VIDEO CONTINUOUS PLAY & ANALYSIS
         ========================================================================= */}
      {mode === "video" && (
        <div className="p-4 space-y-3">
          {video ? (
            <div className="space-y-3">
              <div className="rounded-lg border border-primary/40 bg-primary/10 p-3">
                <div className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full bg-critical animate-ping" />
                  <span className="font-mono text-xs font-semibold uppercase text-primary">
                    Continuous Video Analysis Active
                  </span>
                </div>
                <p className="mt-1 text-xs font-medium text-foreground truncate">
                  {video.name}
                </p>
                <p className="mt-0.5 text-[11px] text-muted-foreground">
                  The video is currently playing in the investigation canvas with synchronized real-time PPE detection overlays.
                </p>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={() => videoInputRef.current?.click()}
                  className="flex-1 flex items-center justify-center gap-1.5 rounded-md border border-border bg-secondary/50 py-2 text-xs font-medium text-foreground hover:bg-secondary transition-colors"
                >
                  <RefreshCw className="h-3 w-3" />
                  Change Video
                </button>
                <button
                  onClick={onRemove}
                  className="flex items-center justify-center gap-1.5 rounded-md border border-border px-3 py-2 text-xs text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors"
                >
                  <Trash2 className="h-3 w-3" />
                  Remove
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <button
                onClick={() => videoInputRef.current?.click()}
                className={cn(
                  "flex h-36 w-full flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-border/80 bg-secondary/15 transition-all hover:border-primary/60 hover:bg-primary/5",
                  dragOver && "border-primary bg-primary/10",
                )}
              >
                <Video className="h-6 w-6 text-primary" />
                <span className="text-xs font-medium text-foreground">
                  Upload Surveillance Video
                </span>
                <span className="text-[11px] text-muted-foreground">
                  MP4, WebM or MOV · Plays & analyzes continuously
                </span>
              </button>

              <button
                onClick={() => videoInputRef.current?.click()}
                className="w-full flex items-center justify-center gap-2 rounded-lg border border-border bg-secondary/50 py-2 text-xs font-medium text-foreground hover:bg-secondary transition-colors"
              >
                <Upload className="h-3.5 w-3.5 text-amber-400" />
                Select Video File
              </button>

              {onSampleVideo && (
                <button
                  onClick={onSampleVideo}
                  className="w-full flex items-center justify-center gap-2 rounded-lg border border-primary/40 bg-primary/10 py-2 text-xs font-medium text-primary hover:bg-primary/20 transition-colors"
                >
                  <Sparkles className="h-3.5 w-3.5" />
                  Load Sample CCTV Surveillance Video
                </button>
              )}
            </div>
          )}
        </div>
      )}

      {/* =========================================================================
          TAB 3: LIVE WEBCAM (CONTINUOUS STREAMING DIRECTLY INTO CANVAS)
         ========================================================================= */}
      {mode === "webcam" && (
        <div className="p-4 space-y-3">
          {/* Status Monitor Viewfinder */}
          <div className="relative aspect-video w-full overflow-hidden rounded-lg bg-black/95 border border-border/80">
            {isWebcamLive ? (
              <div className="flex h-full w-full flex-col items-center justify-center p-4 text-center bg-gradient-to-b from-primary/15 via-black/80 to-black">
                <span className="flex h-3 w-3 rounded-full bg-critical animate-ping" />
                <p className="mt-2 text-xs font-semibold text-foreground">
                  Live Webcam Streaming into Canvas
                </p>
                <p className="mt-1 text-[11px] text-muted-foreground max-w-[220px]">
                  Real-time continuous PPE detection and worker bounding box tracking is active on the main canvas.
                </p>
                <span className="mt-2 inline-flex items-center gap-1.5 rounded-full border border-success/40 bg-success/10 px-2.5 py-0.5 text-[10px] font-semibold text-success">
                  <CheckCircle2 className="h-3 w-3" />
                  Continuous Inference Online
                </span>
              </div>
            ) : (
              <div className="flex h-full w-full flex-col items-center justify-center gap-2 p-4 text-center">
                <Camera className="h-8 w-8 text-primary/70 animate-pulse" />
                <p className="text-xs font-semibold text-foreground">
                  Live Webcam Stream Ready
                </p>
                <p className="text-[11px] text-muted-foreground max-w-[220px]">
                  Streams continuously into the Investigation Canvas with real-time AI worker & PPE detection.
                </p>
              </div>
            )}
          </div>

          <div className="space-y-2">
            {!isWebcamLive ? (
              <button
                onClick={() => onStartWebcam?.()}
                className="w-full flex items-center justify-center gap-2 rounded-lg bg-primary py-2.5 text-xs font-semibold text-primary-foreground shadow-lg hover:bg-primary/90 transition-all active:scale-[0.99]"
              >
                <Camera className="h-4 w-4" />
                Start Live Webcam Stream
              </button>
            ) : (
              <div className="space-y-2">
                <div className="flex gap-2">
                  <button
                    onClick={() => onToggleFacingMode?.()}
                    className="flex-1 flex items-center justify-center gap-1.5 rounded-lg border border-border bg-secondary/40 py-2 text-xs font-medium text-foreground hover:bg-secondary"
                  >
                    <RotateCw className="h-3.5 w-3.5" />
                    Flip Camera ({facingMode})
                  </button>
                  <button
                    onClick={() => onStopWebcam?.()}
                    className="flex-1 flex items-center justify-center gap-1.5 rounded-lg border border-critical/40 bg-critical/10 py-2 text-xs font-medium text-critical hover:bg-critical/20"
                  >
                    Stop Webcam
                  </button>
                </div>
              </div>
            )}

            {/* Video Upload Section Directly Inside Webcam Tab */}
            <div className="border-t border-border/70 pt-3 space-y-2">
              <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                <span className="font-medium text-foreground">Or Upload Video Stream</span>
                <span className="font-mono text-[10px]">MP4 · WebM · MOV</span>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => {
                    onStopWebcam?.();
                    videoInputRef.current?.click();
                  }}
                  className="flex items-center justify-center gap-1.5 rounded-lg border border-amber-500/40 bg-amber-500/10 py-2 text-xs font-medium text-amber-400 hover:bg-amber-500/20 transition-colors shadow-sm"
                >
                  <Upload className="h-3.5 w-3.5" />
                  Upload Video
                </button>

                {onSampleVideo && (
                  <button
                    onClick={() => {
                      onStopWebcam?.();
                      onSampleVideo();
                    }}
                    className="flex items-center justify-center gap-1.5 rounded-lg border border-border bg-secondary/40 py-2 text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
                  >
                    <Sparkles className="h-3.5 w-3.5" />
                    Sample Video
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
