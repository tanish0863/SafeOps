/**
 * SAFEOPS — demonstration data.
 *
 * Everything in this file is clearly-labelled DEMONSTRATION data for visual
 * foundation purposes only. It is replaced by real backend responses once the
 * computer-vision service is connected (see src/services/detectionService.ts).
 */

export type ComplianceStatus = "compliant" | "violation" | "review";
export type Severity = "critical" | "high" | "medium" | "low";
export type ReviewStatus = "confirmed" | "pending" | "resolved";

export interface Worker {
  id: string;
  name: string;
  zone: string;
  shift: string;
  lastScan: string;
  scans: number;
  helmet: boolean;
  vest: boolean;
  status: ComplianceStatus;
  needsReview: boolean;
}

export interface ViolationRecord {
  id: string;
  workerId: string;
  workerName: string;
  type: string;
  zone: string;
  timestamp: string;
  severity: Severity;
  reviewStatus: ReviewStatus;
  note: string;
}

export interface ActivityItem {
  id: string;
  time: string;
  message: string;
  kind: "scan" | "violation" | "review";
}

export interface ZoneStat {
  zone: string;
  violations: number;
  compliance: number;
}

export interface PpeBreakdownItem {
  ppe: string;
  violations: number;
}

export interface TrendPoint {
  label: string;
  rate: number;
}

export interface DetectionFrame {
  id: string;
  fileName: string;
  capturedAt: string;
}

/** Bounding boxes are percentages of the frame (0-100) so overlays scale with any image. */
export interface DetectedWorker {
  workerId: string;
  name: string;
  box: { x: number; y: number; w: number; h: number };
  status: ComplianceStatus;
  helmet: boolean;
  vest: boolean;
  confidence: number;
  missing: string[];
}

export interface DetectionResult {
  jobId: string;
  fileName: string;
  processedAt: string;
  model: string;
  workers: DetectedWorker[];
}

export const demoWorkers: Worker[] = [
  { id: "W-041", name: "Rahul Mehta", zone: "Assembly Line A", shift: "A", lastScan: "2026-10-30 06:42", scans: 214, helmet: true, vest: true, status: "compliant", needsReview: false },
  { id: "W-074", name: "Arun Singh", zone: "Assembly Line B", shift: "A", lastScan: "2026-10-30 06:37", scans: 187, helmet: true, vest: true, status: "compliant", needsReview: false },
  { id: "W-022", name: "Priya Nair", zone: "Welding Bay", shift: "B", lastScan: "2026-10-30 05:58", scans: 156, helmet: true, vest: false, status: "violation", needsReview: false },
  { id: "W-117", name: "Vikram Rao", zone: "Loading Dock", shift: "B", lastScan: "2026-10-30 05:51", scans: 98, helmet: false, vest: true, status: "violation", needsReview: false },
  { id: "W-063", name: "Sana Qureshi", zone: "Quality Control", shift: "A", lastScan: "2026-10-30 06:21", scans: 142, helmet: true, vest: true, status: "compliant", needsReview: false },
  { id: "W-131", name: "Unverified", zone: "Foundry Floor", shift: "C", lastScan: "2026-10-30 04:12", scans: 37, helmet: false, vest: true, status: "violation", needsReview: false },
  { id: "W-089", name: "Joseph Mathew", zone: "Welding Bay", shift: "C", lastScan: "2026-10-30 03:44", scans: 63, helmet: true, vest: true, status: "review", needsReview: true },
  { id: "W-105", name: "Deepa Iyer", zone: "Assembly Line A", shift: "C", lastScan: "2026-10-30 03:29", scans: 51, helmet: true, vest: true, status: "review", needsReview: true },
];

export const demoViolations: ViolationRecord[] = [
  { id: "V-1024", workerId: "W-117", workerName: "Vikram Rao", type: "Missing Helmet", zone: "Loading Dock", timestamp: "2026-10-30 06:51:07", severity: "critical", reviewStatus: "confirmed", note: "Helmet undetected for 2 consecutive frames while entering dock." },
  { id: "V-1023", workerId: "W-022", workerName: "Priya Nair", type: "Missing Safety Vest", zone: "Welding Bay", timestamp: "2026-10-30 05:58:41", severity: "high", reviewStatus: "confirmed", note: "High-visibility vest not detected at welding station." },
  { id: "V-1022", workerId: "W-131", workerName: "Unverified", type: "Missing Helmet", zone: "Foundry Floor", timestamp: "2026-10-30 04:12:19", severity: "critical", reviewStatus: "pending", note: "Worker identity unverified; PPE match unavailable." },
  { id: "V-1021", workerId: "W-089", workerName: "Joseph Mathew", type: "Unverified PPE", zone: "Welding Bay", timestamp: "2026-10-30 03:44:02", severity: "medium", reviewStatus: "pending", note: "Low confidence vest detection — sent for manual review." },
  { id: "V-1020", workerId: "W-117", workerName: "Vikram Rao", type: "Missing Helmet", zone: "Loading Dock", timestamp: "2026-10-29 22:17:55", severity: "high", reviewStatus: "resolved", note: "Operator confirmed helmet removed during short break." },
  { id: "V-1019", workerId: "W-022", workerName: "Priya Nair", type: "Missing Safety Vest", zone: "Welding Bay", timestamp: "2026-10-29 18:03:12", severity: "medium", reviewStatus: "resolved", note: "Vest replaced on site; follow-up scan compliant." },
  { id: "V-1018", workerId: "W-105", workerName: "Deepa Iyer", type: "Unverified PPE", zone: "Assembly Line A", timestamp: "2026-10-29 15:47:30", severity: "low", reviewStatus: "pending", note: "Occlusion by trolley — reviewer confirmation required." },
];

export const demoActivity: ActivityItem[] = [
  { id: "A-08", time: "06:42", message: "Scan completed — W-041 compliant (Assembly Line A)", kind: "scan" },
  { id: "A-07", time: "06:51", message: "Critical violation raised — W-117 missing helmet (Loading Dock)", kind: "violation" },
  { id: "A-06", time: "05:58", message: "Violation confirmed — W-022 missing safety vest (Welding Bay)", kind: "violation" },
  { id: "A-05", time: "05:31", message: "Scan batch processed — 23 workers, 22 compliant", kind: "scan" },
  { id: "A-04", time: "04:58", message: "Manual review requested — W-089 low-confidence detection", kind: "review" },
  { id: "A-03", time: "04:12", message: "Critical violation raised — W-131 missing helmet (Foundry Floor)", kind: "violation" },
];

export const complianceTrend: TrendPoint[] = [
  { label: "W-19", rate: 84.1 },
  { label: "W-20", rate: 85.0 },
  { label: "W-21", rate: 86.4 },
  { label: "W-22", rate: 85.9 },
  { label: "W-23", rate: 87.2 },
  { label: "W-24", rate: 88.6 },
  { label: "W-25", rate: 88.1 },
  { label: "W-26", rate: 90.3 },
  { label: "W-27", rate: 91.7 },
  { label: "W-28", rate: 91.2 },
  { label: "W-29", rate: 93.4 },
  { label: "W-30", rate: 94.6 },
];

export const zoneStats: ZoneStat[] = [
  { zone: "Assembly A", violations: 6, compliance: 93.5 },
  { zone: "Assembly B", violations: 4, compliance: 95.1 },
  { zone: "Welding", violations: 9, compliance: 86.2 },
  { zone: "Foundry", violations: 11, compliance: 82.8 },
  { zone: "Loading Dock", violations: 8, compliance: 88.4 },
];

export const ppeBreakdown: PpeBreakdownItem[] = [
  { ppe: "Helmet", violations: 11 },
  { ppe: "Safety Vest", violations: 8 },
  { ppe: "Gloves", violations: 4 },
  { ppe: "Footwear", violations: 3 },
];

export const overviewStats = {
  workersDetected: { value: "148", delta: "+6 this week" },
  compliant: { value: "121", delta: "81.8% compliance" },
  violations: { value: "19", delta: "3 critical" },
  needsReview: { value: "8", delta: "4 flagged today" },
};

/** Demo result returned by detectionService.fetchDetectionResult(). */
export const demoDetectionResult: DetectionResult = {
  jobId: "job_demo_001",
  fileName: "floor_cam_04.jpg",
  processedAt: "2026-10-30 06:42:11",
  model: "safeops-vision v0.1 (demo)",
  workers: [
    { workerId: "W-074", name: "Arun Singh", box: { x: 8, y: 12, w: 24, h: 76 }, status: "compliant", helmet: true, vest: true, confidence: 0.97, missing: [] },
    { workerId: "W-131", name: "Unverified", box: { x: 38, y: 20, w: 17, h: 66 }, status: "violation", helmet: false, vest: true, confidence: 0.91, missing: ["Helmet"] },
    { workerId: "W-089", name: "Joseph Mathew", box: { x: 63, y: 14, w: 21, h: 74 }, status: "review", helmet: true, vest: false, confidence: 0.58, missing: ["Safety Vest"] },
  ],
};

// ---------- Detection Workspace types ----------
// Boxes use normalized 0-1 coordinates relative to the analyzed image so the
// overlay system maps directly onto real backend responses later.

export interface NormalizedBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export type PpeStatus = "detected" | "missing" | "uncertain";

export interface PpeDetection {
  kind: "helmet" | "vest";
  status: PpeStatus;
  /** Item bounding box in image space, when the item was localized. */
  box: NormalizedBox | null;
  confidence: number;
  /** False when the corresponding PPE rule is disabled for the run. */
  evaluated: boolean;
}

export interface WorkspaceWorker {
  workerId: string;
  box: NormalizedBox;
  personConfidence: number;
  helmet: PpeDetection;
  vest: PpeDetection;
  compliance: ComplianceStatus;
  /** Confidence that each detected PPE item belongs to this worker. */
  associationConfidence: number;
  reviewReason: string | null;
}

export interface WorkspaceAnalysis {
  jobId: string;
  fileName: string;
  analyzedAt: string;
  model: string;
  isDemo: true;
  workers: WorkspaceWorker[];
}

export interface PpeRules {
  helmetRequired: boolean;
  vestRequired: boolean;
}

/**
 * Demo worker geometry for the bundled sample frame (detection-frame.jpg).
 * Values are placeholders for the real vision backend — never presented as
 * actual AI inference in the UI.
 */
export const demoWorkspaceWorkers: WorkspaceWorker[] = [
  {
    workerId: "WORKER 01",
    box: { x: 0.295, y: 0.12, w: 0.27, h: 0.82 },
    personConfidence: 0.96,
    helmet: { kind: "helmet", status: "detected", box: { x: 0.345, y: 0.13, w: 0.13, h: 0.14 }, confidence: 0.97, evaluated: true },
    vest: { kind: "vest", status: "detected", box: { x: 0.325, y: 0.36, w: 0.21, h: 0.4 }, confidence: 0.94, evaluated: true },
    compliance: "compliant",
    associationConfidence: 0.93,
    reviewReason: null,
  },
  {
    workerId: "WORKER 02",
    box: { x: 0.615, y: 0.2, w: 0.17, h: 0.66 },
    personConfidence: 0.91,
    helmet: { kind: "helmet", status: "missing", box: { x: 0.655, y: 0.2, w: 0.09, h: 0.1 }, confidence: 0.88, evaluated: true },
    vest: { kind: "vest", status: "detected", box: { x: 0.635, y: 0.38, w: 0.14, h: 0.36 }, confidence: 0.9, evaluated: true },
    compliance: "violation",
    associationConfidence: 0.9,
    reviewReason: null,
  },
  {
    workerId: "WORKER 03",
    box: { x: 0.055, y: 0.24, w: 0.16, h: 0.62 },
    personConfidence: 0.84,
    helmet: { kind: "helmet", status: "uncertain", box: null, confidence: 0.54, evaluated: true },
    vest: { kind: "vest", status: "detected", box: { x: 0.075, y: 0.4, w: 0.12, h: 0.34 }, confidence: 0.86, evaluated: true },
    compliance: "review",
    associationConfidence: 0.61,
    reviewReason: "Helmet partially occluded — low-confidence match routed to manual review.",
  },
];
