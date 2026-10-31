/**
 * Drives the Detection Workspace flow: image/video/webcam input → continuous analysis → inspection.
 * Uses the placeholder detectionService, so replacing the service with the
 * real backend automatically powers this hook with no UI changes.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import type { PpeRules, WorkspaceAnalysis } from "@/data/demoData";
import { runImageAnalysis, uploadImage, uploadVideo } from "@/services/detectionService";
import { addRealtimeWorker } from "@/lib/workerStore";
import { createLiveWebcamFallbackStream, generateSurveillanceVideoBlob } from "@/lib/sampleVideo";

function syncAnalysisToWorkerStore(analysis: WorkspaceAnalysis) {
  if (!analysis.workers || analysis.workers.length === 0) return;
  const now = new Date();
  const timestamp = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")} ${now.toTimeString().split(" ")[0].slice(0, 5)}`;
  analysis.workers.forEach((dw) => {
    const isHelmetDetected = dw.helmet?.status === "detected";
    const isVestDetected = dw.vest?.status === "detected";
    const status = dw.compliance || "review";
    const workerId = dw.workerId || `W-${Math.floor(100 + Math.random() * 900)}`;
    const workerName = dw.workerId === "W-131" ? "Unverified Personnel" : `Personnel ${dw.workerId}`;

    addRealtimeWorker({
      id: workerId,
      name: workerName,
      zone: "Vision Intake Corridor",
      shift: "A",
      lastScan: timestamp,
      scans: 1,
      helmet: isHelmetDetected,
      vest: isVestDetected,
      status: status,
      needsReview: status === "review",
    });
  });
}

export type DetectionWorkspaceState = "idle" | "ready" | "processing" | "complete" | "error";
export type MediaType = "image" | "video" | "webcam";

export interface WorkspaceImage {
  file: File;
  url: string;
  uploadedAt: string;
  sourceType?: MediaType;
}

export interface WorkspaceVideo {
  file?: File;
  url: string;
  name: string;
  uploadedAt: string;
  isSample?: boolean;
}

export function useDetectionFlow() {
  const [state, setState] = useState<DetectionWorkspaceState>("idle");
  const [mediaType, setMediaType] = useState<MediaType>("image");
  const [image, setImageState] = useState<WorkspaceImage | null>(null);
  const [video, setVideoState] = useState<WorkspaceVideo | null>(null);
  const [webcamStream, setWebcamStream] = useState<MediaStream | null>(null);
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user");
  const [isVideoPlaying, setIsVideoPlaying] = useState<boolean>(true);
  const [continuousAnalysis, setContinuousAnalysis] = useState<boolean>(true);
  const [rules, setRules] = useState<PpeRules>({ helmetRequired: true, vestRequired: true });
  const [result, setResult] = useState<WorkspaceAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedWorkerId, setSelectedWorkerId] = useState<string | null>(null);

  const imageObjectUrlRef = useRef<string | null>(null);
  const videoObjectUrlRef = useRef<string | null>(null);
  const webcamStreamRef = useRef<MediaStream | null>(null);

  const stopWebcamStream = useCallback(() => {
    if (webcamStreamRef.current) {
      webcamStreamRef.current.getTracks().forEach((track) => track.stop());
      webcamStreamRef.current = null;
    }
    setWebcamStream(null);
  }, []);

  const releaseUrls = useCallback(() => {
    if (imageObjectUrlRef.current) {
      URL.revokeObjectURL(imageObjectUrlRef.current);
      imageObjectUrlRef.current = null;
    }
    if (videoObjectUrlRef.current) {
      URL.revokeObjectURL(videoObjectUrlRef.current);
      videoObjectUrlRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      releaseUrls();
      if (webcamStreamRef.current) {
        webcamStreamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, [releaseUrls]);

  // Start live webcam streaming directly into the investigation canvas
  const startWebcamStream = useCallback(
    async (targetFacingMode?: "user" | "environment") => {
      const mode = targetFacingMode ?? facingMode;
      try {
        stopWebcamStream();
        releaseUrls();
        setImageState(null);
        setVideoState(null);

        let stream: MediaStream | null = null;
        let isSimulated = false;

        // Progressive hardware webcam acquisition
        if (typeof navigator !== "undefined" && navigator?.mediaDevices?.getUserMedia) {
          // Attempt 1: Target facing mode with standard HD resolution
          try {
            stream = await navigator.mediaDevices.getUserMedia({
              video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: mode,
              },
              audio: false,
            });
          } catch {
            // Attempt 2: Relaxed facing mode constraint
            try {
              stream = await navigator.mediaDevices.getUserMedia({
                video: {
                  width: { ideal: 1280 },
                  height: { ideal: 720 },
                },
                audio: false,
              });
            } catch {
              // Attempt 3: Minimum video constraint
              try {
                stream = await navigator.mediaDevices.getUserMedia({
                  video: true,
                  audio: false,
                });
              } catch {
                // Physical device unavailable or permission denied
              }
            }
          }
        }

        // If hardware camera is unavailable, switch to live optical fallback stream
        if (!stream) {
          stream = createLiveWebcamFallbackStream(mode);
          isSimulated = true;
        }

        webcamStreamRef.current = stream;
        setWebcamStream(stream);
        setMediaType("webcam");
        setIsVideoPlaying(true);
        setContinuousAnalysis(true);
        setState("processing");
        setError(null);
        setSelectedWorkerId(null);

        if (isSimulated) {
          toast.info("Live Optical Camera Stream Connected", {
            description: "Physical camera unavailable or permission denied. Continuous live simulated CCTV stream is running.",
          });
        } else {
          toast.success("Live Optical Feed Connected", {
            description: "Streaming directly into canvas with real-time continuous AI PPE detection.",
          });
        }

        // Immediately initialize continuous detection on the live stream
        const analysis = await runImageAnalysis("live_webcam_feed.mp4", rules);
        setResult(analysis);
        syncAnalysisToWorkerStore(analysis);
        setState("complete");
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Camera access denied or unavailable.";
        setError(msg);
        setState("error");
        toast.error("Camera connection failed", { description: msg });
      }
    },
    [facingMode, rules, releaseUrls, stopWebcamStream],
  );

  const toggleFacingMode = useCallback(() => {
    setFacingMode((prev) => {
      const next = prev === "user" ? "environment" : "user";
      if (webcamStreamRef.current) {
        setTimeout(() => {
          void startWebcamStream(next);
        }, 50);
      }
      return next;
    });
  }, [startWebcamStream]);

  // Set image file input
  const setImage = useCallback(
    (file: File, sourceType: MediaType = "image") => {
      stopWebcamStream();
      if (!/^image\/(jpeg|jpg|png|webp)$/i.test(file.type)) {
        setError("Unsupported format. Upload a JPG, PNG or WebP image.");
        setState("error");
        return;
      }
      if (imageObjectUrlRef.current) {
        URL.revokeObjectURL(imageObjectUrlRef.current);
      }
      const url = URL.createObjectURL(file);
      imageObjectUrlRef.current = url;
      setImageState({
        file,
        url,
        uploadedAt: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
        sourceType,
      });
      setMediaType(sourceType);
      setState("ready");
      setResult(null);
      setSelectedWorkerId(null);
      setError(null);
      // Persist to backend media storage asynchronously
      uploadImage(file).catch((e) => console.warn("Background image upload sync:", e));
    },
    [stopWebcamStream],
  );

  // Set video file input and automatically start continuous analysis
  const setVideo = useCallback(
    async (file: File) => {
      stopWebcamStream();
      if (!/^video\/(mp4|webm|quicktime|ogg|x-matroska)$/i.test(file.type)) {
        setError("Unsupported video format. Upload MP4, WebM or MOV.");
        setState("error");
        return;
      }
      if (videoObjectUrlRef.current) {
        URL.revokeObjectURL(videoObjectUrlRef.current);
      }
      const url = URL.createObjectURL(file);
      videoObjectUrlRef.current = url;
      setVideoState({
        file,
        url,
        name: file.name,
        uploadedAt: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
        isSample: false,
      });
      setMediaType("video");
      setIsVideoPlaying(true);
      setState("processing");
      setError(null);
      setSelectedWorkerId(null);

      // Persist video to backend media store
      uploadVideo(file).catch((e) => console.warn("Background video upload sync:", e));

      // Auto-initialize real-time detection model on the video
      try {
        const analysis = await runImageAnalysis(file.name, rules);
        setResult(analysis);
        syncAnalysisToWorkerStore(analysis);
        setState("complete");
      } catch {
        setError("Video analysis initialization failed.");
        setState("error");
      }
    },
    [rules],
  );

  // Load sample video stream
  const loadSampleVideo = useCallback(
    async (sampleAssetUrl: string) => {
      stopWebcamStream();
      setMediaType("video");
      setIsVideoPlaying(true);
      setState("processing");
      setError(null);
      setSelectedWorkerId(null);

      try {
        const realVideoUrl = await generateSurveillanceVideoBlob(sampleAssetUrl);
        if (videoObjectUrlRef.current && videoObjectUrlRef.current.startsWith("blob:")) {
          URL.revokeObjectURL(videoObjectUrlRef.current);
        }
        videoObjectUrlRef.current = realVideoUrl;

        setVideoState({
          url: realVideoUrl,
          name: "cctv_foundry_cam04_stream.webm",
          uploadedAt: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
          isSample: true,
        });

        const analysis = await runImageAnalysis("cctv_foundry_cam04_stream.webm", rules);
        setResult(analysis);
        syncAnalysisToWorkerStore(analysis);
        setState("complete");
      } catch {
        setError("Sample video stream generation failed.");
        setState("error");
      }
    },
    [rules],
  );

  const loadSample = useCallback(
    async (assetUrl: string) => {
      try {
        const blob = await (await fetch(assetUrl)).blob();
        setImage(new File([blob], "sample_floor_frame.jpg", { type: blob.type || "image/jpeg" }));
      } catch {
        setError("The sample frame could not be loaded. Try uploading an image instead.");
        setState("error");
      }
    },
    [setImage],
  );

  const analyze = useCallback(async () => {
    const sourceName = mediaType === "video" ? video?.name : image?.file.name;
    if (!sourceName) return;

    setState("processing");
    setError(null);
    setSelectedWorkerId(null);
    try {
      const analysis = await runImageAnalysis(sourceName, rules);
      setResult(analysis);
      syncAnalysisToWorkerStore(analysis);
      setState("complete");
    } catch {
      setError("Analysis failed. The detection backend is not connected — retry or reset the investigation.");
      setState("error");
    }
  }, [image, video, mediaType, rules]);

  const toggleRule = useCallback((key: keyof PpeRules) => {
    setRules((current) => {
      const updated = { ...current, [key]: !current[key] };
      // If video is active and complete, re-evaluate live rules immediately!
      if (result) {
        runImageAnalysis(result.fileName, updated).then(setResult).catch(() => {});
      }
      return updated;
    });
  }, [result]);

  const toggleVideoPlay = useCallback(() => {
    setIsVideoPlaying((prev) => !prev);
  }, []);

  const toggleContinuousAnalysis = useCallback(() => {
    setContinuousAnalysis((prev) => !prev);
  }, []);

  const removeMedia = useCallback(() => {
    releaseUrls();
    stopWebcamStream();
    setImageState(null);
    setVideoState(null);
    setState("idle");
    setResult(null);
    setSelectedWorkerId(null);
    setError(null);
  }, [releaseUrls, stopWebcamStream]);

  const reset = removeMedia;

  return {
    state,
    mediaType,
    image,
    video,
    webcamStream,
    facingMode,
    isVideoPlaying,
    continuousAnalysis,
    rules,
    result,
    error,
    selectedWorkerId,
    setMediaType,
    setImage,
    setVideo,
    startWebcamStream,
    stopWebcamStream,
    toggleFacingMode,
    loadSampleVideo,
    loadSample,
    analyze,
    toggleRule,
    toggleVideoPlay,
    toggleContinuousAnalysis,
    setIsVideoPlaying,
    removeMedia,
    reset,
    setSelectedWorkerId,
  };
}
