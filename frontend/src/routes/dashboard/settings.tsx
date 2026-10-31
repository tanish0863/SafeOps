import { useEffect, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { toast } from "sonner";

import { PageHeader } from "@/components/common/PageHeader";
import { TechLabel } from "@/components/common/TechLabel";
import { API_CONFIG } from "@/services/api";
import {
  checkBackendHealth,
  fetchSettings,
  saveSettings,
  uploadImage,
  uploadVideo,
  type SystemSettings,
} from "@/services/detectionService";

export const Route = createFileRoute("/dashboard/settings")({
  head: () => ({
    meta: [
      { title: "Settings — SafeOps Dashboard" },
      { name: "description", content: "Detection rules, alert thresholds and backend connection settings." },
      { property: "og:title", content: "Settings — SafeOps Dashboard" },
      { property: "og:description", content: "Detection rules, alert thresholds and backend connection settings." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: SettingsPage,
});

const endpoints = [
  { method: "GET", path: "/health", label: "Backend health and service status" },
  { method: "POST", path: "/media/image", label: "Upload and persist an image" },
  { method: "POST", path: "/media/video", label: "Upload and persist a video stream" },
  { method: "POST", path: "/detection/jobs", label: "Request a detection job" },
  { method: "GET", path: "/detection/jobs/:id/result", label: "Fetch detection job results" },
  { method: "GET", path: "/violations", label: "Fetch violation records and filter" },
  { method: "GET", path: "/workers", label: "Retrieve registered worker roster" },
  { method: "GET", path: "/settings", label: "Fetch persisted system settings" },
  { method: "PUT", path: "/settings", label: "Update system settings" },
];

/** System settings: backend configuration, live endpoints and interface options. */
function SettingsPage() {
  const [alertCritical, setAlertCritical] = useState(true);
  const [alertReview, setAlertReview] = useState(false);
  const [enableAudio, setEnableAudio] = useState(true);
  const [isConnected, setIsConnected] = useState<boolean | null>(null);
  const [currentSettings, setCurrentSettings] = useState<SystemSettings | null>(null);

  useEffect(() => {
    let isMounted = true;
    async function initSettings() {
      const healthy = await checkBackendHealth();
      if (!isMounted) return;
      setIsConnected(healthy);

      if (healthy) {
        try {
          const cfg = await fetchSettings();
          if (isMounted && cfg) {
            setCurrentSettings(cfg);
            setAlertCritical(cfg.alertCriticalViolations);
            setAlertReview(cfg.alertReviewRequired);
            setEnableAudio(cfg.enableAudioAlerts);
          }
        } catch (e) {
          console.warn("Could not load backend settings:", e);
        }
      }
    }
    initSettings();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleToggleCritical = async (checked: boolean) => {
    setAlertCritical(checked);
    if (currentSettings) {
      const updated = { ...currentSettings, alertCriticalViolations: checked };
      setCurrentSettings(updated);
      try {
        await saveSettings(updated);
        toast.success("Settings saved to backend database");
      } catch {
        toast.error("Failed to persist settings to backend");
      }
    }
  };

  const handleToggleReview = async (checked: boolean) => {
    setAlertReview(checked);
    if (currentSettings) {
      const updated = { ...currentSettings, alertReviewRequired: checked };
      setCurrentSettings(updated);
      try {
        await saveSettings(updated);
        toast.success("Settings saved to backend database");
      } catch {
        toast.error("Failed to persist settings to backend");
      }
    }
  };

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="System Configuration"
        title="Settings"
        description="Configuration and live connection to the SafeOps FastAPI backend server and SQLite database."
      />

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="border border-border bg-card p-5">
          <TechLabel tone="accent">Backend connection</TechLabel>
          <dl className="mt-4 space-y-3 text-sm">
            <div className="flex items-center justify-between gap-4 border-b border-border pb-3">
              <dt className="text-muted-foreground">Base URL</dt>
              <dd className="font-mono text-xs text-foreground">{API_CONFIG.baseUrl}</dd>
            </div>
            <div className="flex items-center justify-between gap-4 border-b border-border pb-3">
              <dt className="text-muted-foreground">Environment override</dt>
              <dd className="font-mono text-xs text-foreground">VITE_SAFEOPS_API_URL</dd>
            </div>
            <div className="flex items-center justify-between gap-4">
              <dt className="text-muted-foreground">Status</dt>
              <dd>
                {isConnected === null ? (
                  <span className="font-mono text-xs uppercase tracking-[0.14em] text-muted-foreground">Checking...</span>
                ) : isConnected ? (
                  <span className="inline-flex items-center gap-1.5 font-mono text-xs uppercase tracking-[0.14em] text-emerald-400">
                    <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                    Connected (Online)
                  </span>
                ) : (
                  <span className="font-mono text-xs uppercase tracking-[0.14em] text-warning">
                    Offline / Not connected
                  </span>
                )}
              </dd>
            </div>
          </dl>
          <p className="mt-4 text-sm text-muted-foreground">
            Backend is connected to SafeOps FastAPI service at {API_CONFIG.baseUrl}. All API requests persist to SQLite database safeops.db.
          </p>
        </div>

        <div className="border border-border bg-card">
          <div className="border-b border-border px-5 py-3">
            <TechLabel tone="accent">Connected API endpoints</TechLabel>
          </div>
          <ul className="max-h-[280px] overflow-y-auto">
            {endpoints.map((endpoint) => (
              <li
                key={endpoint.path}
                className="flex flex-wrap items-center gap-3 border-b border-border px-5 py-3 last:border-0"
              >
                <span className="border border-border px-1.5 py-px font-mono text-[10px] uppercase tracking-[0.14em] text-primary">
                  {endpoint.method}
                </span>
                <span className="font-mono text-xs text-foreground">{endpoint.path}</span>
                <span className="ml-auto text-xs text-muted-foreground">{endpoint.label}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="border border-border bg-card">
        <div className="border-b border-border px-5 py-3">
          <TechLabel tone="accent">Interface options</TechLabel>
        </div>
        <div className="divide-y divide-border">
          <label className="flex items-center justify-between gap-4 px-5 py-4 cursor-pointer">
            <span>
              <span className="block text-sm text-foreground">Critical violation alerts</span>
              <span className="mt-1 block text-xs text-muted-foreground">
                Surface a banner in the overview when a critical violation is raised.
              </span>
            </span>
            <input
              type="checkbox"
              checked={alertCritical}
              onChange={(event) => handleToggleCritical(event.target.checked)}
              className="h-4 w-4 accent-[var(--primary)] cursor-pointer"
            />
          </label>
          <label className="flex items-center justify-between gap-4 px-5 py-4 cursor-pointer">
            <span>
              <span className="block text-sm text-foreground">Manual review reminders</span>
              <span className="mt-1 block text-xs text-muted-foreground">
                Remind operators about detections waiting for manual review.
              </span>
            </span>
            <input
              type="checkbox"
              checked={alertReview}
              onChange={(event) => handleToggleReview(event.target.checked)}
              className="h-4 w-4 accent-[var(--primary)] cursor-pointer"
            />
          </label>
        </div>
      </div>

      <div className="border border-border bg-card p-5">
        <TechLabel>Backend Service Layer</TechLabel>
        <p className="mt-2 text-sm text-muted-foreground">
          uploadImage() and uploadVideo() are wired to SafeOps backend media APIs ({uploadImage.name}, {uploadVideo.name} in src/services/detectionService.ts). Uploaded files are stored in the backend uploads directory and tracked in the media_files database table.
        </p>
      </div>
    </div>
  );
}
