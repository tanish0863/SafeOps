/**
 * SAFEOPS — detection and backend service layer.
 *
 * Connects the frontend to the FastAPI backend API endpoints under /api/v1.
 * Provides resilient fallbacks so the UI remains operational in all states.
 */

import { apiRequest } from "./api";
import {
  demoDetectionResult,
  demoViolations,
  demoWorkspaceWorkers,
  demoActivity,
  complianceTrend,
  zoneStats,
  ppeBreakdown,
  overviewStats,
  type ComplianceStatus,
  type DetectionResult,
  type PpeDetection,
  type PpeRules,
  type PpeStatus,
  type ViolationRecord,
  type WorkspaceAnalysis,
  type WorkspaceWorker,
  type Worker,
  type ActivityItem,
  type TrendPoint,
  type ZoneStat,
  type PpeBreakdownItem,
} from "@/data/demoData";

export interface UploadedMedia {
  mediaId: string;
  fileName: string;
}

export interface DetectionJob {
  jobId: string;
  status: "queued" | "processing" | "complete" | "failed";
}

export interface DashboardStats {
  complianceRate: number;
  totalWorkers: number;
  compliantCount: number;
  violationCount: number;
  reviewCount: number;
  criticalCount: number;
  workersDetected: { value: string; delta: string };
  compliant: { value: string; delta: string };
  violations: { value: string; delta: string };
  needsReview: { value: string; delta: string };
}

export interface SystemSettings {
  personConfidenceThreshold: number;
  ppeConfidenceThreshold: number;
  torsoOverlapThreshold: number;
  alertCriticalViolations: boolean;
  alertReviewRequired: boolean;
  enableAudioAlerts: boolean;
  retentionDays: number;
  webhookUrl?: string;
  notificationEmail?: string;
}

/** Check backend health connectivity */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await apiRequest<{ status: string }>("/health");
    return res.status === "healthy";
  } catch {
    return false;
  }
}

/**
 * Upload an image for detection.
 * Calls backend POST /media/image.
 */
export async function uploadImage(file: File): Promise<UploadedMedia> {
  const formData = new FormData();
  formData.append("file", file);
  try {
    return await apiRequest<UploadedMedia>("/media/image", {
      method: "POST",
      body: formData,
    });
  } catch (err) {
    console.warn("Backend /media/image unreachable, using local fallback", err);
    return { mediaId: `media_demo_${Date.now()}`, fileName: file.name };
  }
}

/**
 * Upload a video for detection.
 * Calls backend POST /media/video.
 */
export async function uploadVideo(file: File): Promise<UploadedMedia> {
  const formData = new FormData();
  formData.append("file", file);
  try {
    return await apiRequest<UploadedMedia>("/media/video", {
      method: "POST",
      body: formData,
    });
  } catch (err) {
    console.warn("Backend /media/video unreachable, using local fallback", err);
    return { mediaId: `media_demo_${Date.now()}`, fileName: file.name };
  }
}

/**
 * Request a detection run on an uploaded medium.
 * Calls backend POST /detection/jobs.
 */
export async function requestDetection(mediaId: string, rules?: PpeRules): Promise<DetectionJob> {
  try {
    return await apiRequest<DetectionJob>("/detection/jobs", {
      method: "POST",
      body: JSON.stringify({ mediaId, rules }),
    });
  } catch (err) {
    console.warn("Backend /detection/jobs unreachable, using local fallback", err);
    return { jobId: `job_${mediaId}`, status: "complete" };
  }
}

/**
 * Fetch the results of a detection job (worker boxes, PPE association, status).
 * Calls backend GET /detection/jobs/:jobId/result.
 */
export async function fetchDetectionResult(jobId: string): Promise<DetectionResult> {
  try {
    return await apiRequest<DetectionResult>(`/detection/jobs/${jobId}/result`);
  } catch (err) {
    console.warn("Backend /detection/jobs/:id/result unreachable, using demo data", err);
    return { ...demoDetectionResult, jobId };
  }
}

/**
 * Fetch recorded violations with server-side filters.
 * Calls backend GET /violations.
 */
export async function fetchViolations(severity?: string, status?: string): Promise<ViolationRecord[]> {
  const params = new URLSearchParams();
  if (severity && severity !== "all") params.append("severity", severity);
  if (status && status !== "all") params.append("reviewStatus", status);
  const query = params.toString() ? `?${params.toString()}` : "";

  try {
    return await apiRequest<ViolationRecord[]>(`/violations${query}`);
  } catch (err) {
    console.warn("Backend /violations unreachable", err);
    return [];
  }
}

/**
 * Resolve an active violation incident.
 * Calls backend POST /violations/{id}/resolve.
 */
export async function resolveViolation(id: string): Promise<ViolationRecord> {
  try {
    return await apiRequest<ViolationRecord>(`/violations/${id}/resolve`, {
      method: "POST",
    });
  } catch (err) {
    console.warn("Backend /violations/:id/resolve unreachable, returning simulated resolution", err);
    const existing = demoViolations.find((v) => v.id === id);
    return existing ? { ...existing, reviewStatus: "resolved" } : ({} as ViolationRecord);
  }
}

function statusFor(statuses: PpeStatus[]): ComplianceStatus {
  if (statuses.includes("missing")) return "violation";
  if (statuses.includes("uncertain")) return "review";
  return "compliant";
}

/**
 * Run a detection analysis on an uploaded image or video stream.
 * Calls backend POST /detection/analyze.
 */
export async function runImageAnalysis(fileName: string, rules: PpeRules): Promise<WorkspaceAnalysis> {
  try {
    return await apiRequest<WorkspaceAnalysis>("/detection/analyze", {
      method: "POST",
      body: JSON.stringify({ fileName, rules }),
    });
  } catch (err) {
    console.warn("Backend /detection/analyze unreachable, executing client evaluation fallback", err);
    const workers = demoWorkspaceWorkers.map((worker) => {
      const helmet: PpeDetection = { ...worker.helmet, evaluated: rules.helmetRequired };
      const vest: PpeDetection = { ...worker.vest, evaluated: rules.vestRequired };
      const evaluated = [helmet, vest].filter((item) => item.evaluated);

      let compliance: ComplianceStatus;
      let reviewReason = worker.reviewReason;
      if (evaluated.length === 0) {
        compliance = "review";
        reviewReason = "No PPE rules enabled for this run — worker routed to review.";
      } else {
        compliance = statusFor(evaluated.map((item) => item.status));
        if (compliance !== "review") reviewReason = null;
      }

      return { ...worker, helmet, vest, compliance, reviewReason };
    });

    return {
      jobId: `job_media_demo_${Date.now()}`,
      fileName,
      analyzedAt: new Date().toISOString(),
      model: "SafeOps-Vision-v2.1 (Client Fallback)",
      isDemo: true,
      workers,
    };
  }
}

/**
 * Dashboard & Analytics Service Calls
 */
export async function fetchDashboardStats(): Promise<DashboardStats> {
  try {
    return await apiRequest<DashboardStats>("/dashboard/stats");
  } catch (err) {
    console.warn("Backend /dashboard/stats unreachable", err);
    return {
      complianceRate: 100.0,
      totalWorkers: 0,
      compliantCount: 0,
      violationCount: 0,
      reviewCount: 0,
      criticalCount: 0,
      workersDetected: { value: "0", delta: "Live registered" },
      compliant: { value: "0", delta: "100% compliance" },
      violations: { value: "0", delta: "0 critical" },
      needsReview: { value: "0", delta: "0 flagged" },
    };
  }
}

export async function fetchRecentActivity(): Promise<ActivityItem[]> {
  try {
    return await apiRequest<ActivityItem[]>("/dashboard/activity");
  } catch (err) {
    console.warn("Backend /dashboard/activity unreachable", err);
    return [];
  }
}

export async function fetchComplianceTrend(): Promise<TrendPoint[]> {
  try {
    return await apiRequest<TrendPoint[]>("/analytics/compliance-trend");
  } catch (err) {
    console.warn("Backend /analytics/compliance-trend unreachable, using demo data", err);
    return complianceTrend;
  }
}

export async function fetchZoneStats(): Promise<ZoneStat[]> {
  try {
    return await apiRequest<ZoneStat[]>("/analytics/zone-distribution");
  } catch (err) {
    console.warn("Backend /analytics/zone-distribution unreachable, using demo data", err);
    return zoneStats;
  }
}

export async function fetchPpeBreakdown(): Promise<PpeBreakdownItem[]> {
  try {
    return await apiRequest<PpeBreakdownItem[]>("/analytics/ppe-breakdown");
  } catch (err) {
    console.warn("Backend /analytics/ppe-breakdown unreachable, using demo data", err);
    return ppeBreakdown;
  }
}

/**
 * Workers Service Calls
 */
export async function fetchWorkers(search?: string, status?: string): Promise<Worker[]> {
  const params = new URLSearchParams();
  if (search) params.append("search", search);
  if (status && status !== "all") params.append("status", status);
  const query = params.toString() ? `?${params.toString()}` : "";

  try {
    return await apiRequest<Worker[]>(`/workers${query}`);
  } catch (err) {
    console.warn("Backend /workers unreachable, using local store", err);
    return [];
  }
}

export async function createWorkerApi(worker: Partial<Worker>): Promise<Worker> {
  return await apiRequest<Worker>("/workers", {
    method: "POST",
    body: JSON.stringify(worker),
  });
}

export async function deleteWorkerApi(id: string): Promise<void> {
  await apiRequest(`/workers/${id}`, {
    method: "DELETE",
  });
}

/**
 * Settings Service Calls
 */
export async function fetchSettings(): Promise<SystemSettings> {
  try {
    return await apiRequest<SystemSettings>("/settings");
  } catch (err) {
    console.warn("Backend /settings unreachable, using defaults", err);
    return {
      personConfidenceThreshold: 0.50,
      ppeConfidenceThreshold: 0.40,
      torsoOverlapThreshold: 0.30,
      alertCriticalViolations: true,
      alertReviewRequired: false,
      enableAudioAlerts: true,
      retentionDays: 30,
    };
  }
}

export async function saveSettings(settings: SystemSettings): Promise<SystemSettings> {
  return await apiRequest<SystemSettings>("/settings", {
    method: "PUT",
    body: JSON.stringify(settings),
  });
}
