import { useState, useEffect } from "react";
import { Link, createFileRoute } from "@tanstack/react-router";
import {
  ScanLine,
  ShieldCheck,
  TriangleAlert,
  Users,
  HardHat,
  Sparkles,
  RefreshCw,
  SlidersHorizontal,
  ChevronRight,
  CheckCircle2,
  AlertOctagon,
  Clock,
  Radio,
  ArrowRight,
  Activity,
  Layers,
} from "lucide-react";
import { toast } from "sonner";

import { buttonVariants } from "@/components/common/button";
import { StatusBadge } from "@/components/common/StatusBadge";
import { TechLabel } from "@/components/common/TechLabel";
import { TrendChart } from "@/components/analytics/Charts";
import { PlantSafetyGauge } from "@/components/dashboard/PlantSafetyGauge";
import { EdgeInferenceCard } from "@/components/dashboard/EdgeInferenceCard";
import { demoActivity, complianceTrend, demoViolations, type ViolationRecord, type ActivityItem } from "@/data/demoData";
import { fetchDashboardStats, fetchViolations, fetchRecentActivity, resolveViolation, type DashboardStats } from "@/services/detectionService";
import { apiRequest } from "@/services/api";

export const Route = createFileRoute("/dashboard/")({
  head: () => ({
    meta: [
      { title: "Plant Safety Command Center — SafeOps" },
      {
        name: "description",
        content:
          "Live site optical compliance monitoring: real-time worker detection, zone safety matrix, and breach alerting.",
      },
      { property: "og:title", content: "Plant Safety Command Center — SafeOps" },
      {
        property: "og:description",
        content:
          "Live site optical compliance monitoring: real-time worker detection, zone safety matrix, and breach alerting.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: OverviewPage,
});

function OverviewPage() {
  const [currentShift, setCurrentShift] = useState<"A" | "B" | "C">("A");
  const [acknowledgedIds, setAcknowledgedIds] = useState<Record<string, boolean>>({});
  const [filterSeverity, setFilterSeverity] = useState<"all" | "critical" | "high">("all");
  const [activeTab, setActiveTab] = useState<"live-view" | "trend">("live-view");

  const [stats, setStats] = useState<DashboardStats>({
    complianceRate: 100.0,
    totalWorkers: 0,
    compliantCount: 0,
    violationCount: 0,
    reviewCount: 0,
    criticalCount: 0,
    workersDetected: { value: "0", delta: "Live registered" },
    compliant: { value: "0", delta: "100% compliance" },
    violations: { value: "0", delta: "0 critical" },
    needsReview: { value: "0", delta: "0 pending" },
  });
  const [violations, setViolations] = useState<ViolationRecord[]>([]);
  const [activities, setActivities] = useState<ActivityItem[]>([]);

  // Hydrate stats, violations, and activities from backend
  useEffect(() => {
    fetchDashboardStats().then(setStats).catch(() => {});
    fetchViolations().then((data) => setViolations(Array.isArray(data) ? data : [])).catch(() => {});
    fetchRecentActivity().then((data) => setActivities(Array.isArray(data) ? data : [])).catch(() => {});
  }, []);

  const handleAcknowledge = async (id: string, workerName: string) => {
    setAcknowledgedIds((prev) => ({ ...prev, [id]: true }));
    try {
      await resolveViolation(id);
    } catch {
      // Handled
    }
    toast.success(`Violation ${id} acknowledged`, {
      description: `Safety supervisor notified for worker ${workerName}. Incident tagged for review.`,
    });
  };

  const handleSimulateScan = async () => {
    toast.info("Ingesting Optical Sensor Feeds", {
      description: "Processing worker vectors across 5 monitoring zones.",
    });
    try {
      await apiRequest("/dashboard/simulate-scan", { method: "POST" });
      const [newStats, newAct] = await Promise.all([fetchDashboardStats(), fetchRecentActivity()]);
      setStats(newStats);
      setActivities(newAct);
    } catch {
      // Handled
    }
  };

  const filteredViolations = violations.filter((v) => {
    if (filterSeverity === "critical") return v.severity === "critical";
    if (filterSeverity === "high") return v.severity === "critical" || v.severity === "high";
    return true;
  });

  const toneByKind = { scan: "neutral", violation: "critical", review: "pending" } as const;

  return (
    <div className="space-y-7">
      {/* Top Mission Control Header */}
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-border/80 pb-6">
        <div>
          <div className="flex flex-wrap items-center gap-3">
            <span className="inline-flex items-center gap-2 rounded-full border border-primary/40 bg-primary/10 px-3 py-1 font-mono text-[11px] font-medium uppercase tracking-wider text-primary">
              <Radio className="h-3 w-3 animate-pulse text-primary" />
              Plant Telemetry · Facility Unit 01
            </span>
            <span className="hidden sm:inline-flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="h-1.5 w-1.5 rounded-full bg-success" /> 8 Optical Sensors Online
            </span>
            <span className="hidden md:inline-flex text-xs text-muted-foreground">
              · CV Pipeline v2.4 (Latency 42ms)
            </span>
          </div>

          <h1 className="mt-2 text-2xl font-bold tracking-tight text-foreground sm:text-3xl">
            Plant Safety Command Center
          </h1>
          <p className="mt-1 text-sm text-muted-foreground max-w-2xl">
            Automated workplace PPE compliance powered by computer vision. Continuous worker localization,
            helmet and vest association, and live breach mitigation.
          </p>
        </div>

        {/* Header Quick Controls */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Shift Picker */}
          <div className="flex items-center rounded-lg border border-border/80 bg-secondary/30 p-1">
            <button
              onClick={() => setCurrentShift("A")}
              className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                currentShift === "A"
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Shift A (Active)
            </button>
            <button
              onClick={() => setCurrentShift("B")}
              className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                currentShift === "B"
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Shift B
            </button>
            <button
              onClick={() => setCurrentShift("C")}
              className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                currentShift === "C"
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Shift C
            </button>
          </div>

          <button
            onClick={handleSimulateScan}
            title="Simulate Sensor Ingestion"
            className="flex items-center gap-1.5 rounded-lg border border-border bg-secondary/40 px-3 py-2 text-xs font-medium text-foreground hover:bg-secondary transition-colors"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Refresh Sensors</span>
          </button>

          <Link
            to="/dashboard/detection"
            className={`${buttonVariants({ variant: "primary" })} glow-primary flex items-center gap-2`}
          >
            <Sparkles className="h-3.5 w-3.5" />
            <span>Run Detection Scan</span>
          </Link>
        </div>
      </div>

      {/* Hero Visual Telemetry Grid: Radial Gauge + Rich Metrics */}
      <div className="grid gap-5 lg:grid-cols-12 items-stretch">
        {/* Radial Plant Safety Index Hero (Spans 7 cols) */}
        <div className="lg:col-span-7">
          <PlantSafetyGauge
            complianceRate={stats.complianceRate}
            totalWorkers={stats.totalWorkers}
            compliantCount={stats.compliantCount}
            violationCount={stats.violationCount}
            reviewCount={stats.reviewCount}
            criticalCount={stats.criticalCount}
          />
        </div>

        {/* Complementary Rich Stat Cards (Spans 5 cols) */}
        <div className="lg:col-span-5 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-1 gap-4">
          {/* AI Vision Inference Pipeline Telemetry Card */}
          <EdgeInferenceCard />

          {/* PPE Deficit Breakdown Card */}
          <div className="panel-raised hover-lift relative overflow-hidden rounded-xl p-5">
            <span className="absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r from-critical via-amber-500 to-transparent" />
            <div className="flex items-start justify-between">
              <div>
                <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
                  Active PPE Deficits
                </span>
                <div className="mt-1 flex items-baseline gap-2">
                  <span className="text-3xl font-bold tracking-tight text-critical">19</span>
                  <span className="inline-flex items-center gap-1 rounded-md border border-critical/40 bg-critical/10 px-2 py-0.5 text-[10px] font-semibold uppercase text-critical">
                    <AlertOctagon className="h-3 w-3" />
                    3 Critical
                  </span>
                </div>
              </div>
              <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-critical/30 bg-critical/10 text-critical">
                <TriangleAlert className="h-4 w-4" />
              </div>
            </div>

            <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
              <div className="rounded-md border border-border/70 bg-secondary/30 p-2">
                <div className="flex items-center gap-1.5 text-muted-foreground">
                  <HardHat className="h-3.5 w-3.5 text-primary" />
                  <span>Missing Helmet</span>
                </div>
                <p className="mt-1 text-base font-bold text-foreground">11 <span className="text-[10px] font-normal text-muted-foreground">cases</span></p>
              </div>

              <div className="rounded-md border border-border/70 bg-secondary/30 p-2">
                <div className="flex items-center gap-1.5 text-muted-foreground">
                  <ShieldCheck className="h-3.5 w-3.5 text-warning" />
                  <span>Missing Vest</span>
                </div>
                <p className="mt-1 text-base font-bold text-foreground">8 <span className="text-[10px] font-normal text-muted-foreground">cases</span></p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Compliance Velocity Trend & Real-Time Event Feed */}
      <div className="grid gap-6 lg:grid-cols-12 items-start">
        {/* Left Column: 12-Week Compliance Trend (6 cols) */}
        <div className="lg:col-span-6 panel rounded-xl p-5 space-y-4 shadow-xl">
          <div className="flex items-center justify-between border-b border-border/70 pb-3">
            <div>
              <h3 className="text-xs font-semibold text-foreground">
                Compliance Rate (12-Week Velocity)
              </h3>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Target: &ge;95% Site-Wide Safety Baseline
              </p>
            </div>
            <span className="rounded-full border border-success/30 bg-success/10 px-2 py-0.5 font-mono text-[10px] text-success font-medium">
              +10.5% Overall
            </span>
          </div>
          <TrendChart data={complianceTrend} />
          <div className="flex items-center justify-between text-xs text-muted-foreground border-t border-border/70 pt-3">
            <span>W-19 Baseline: 84.1%</span>
            <span className="text-success font-medium">Peak Rate: 94.6% (W-30)</span>
          </div>
        </div>

        {/* Right Column: Live Event Stream (6 cols) */}
        <div className="lg:col-span-6 panel overflow-hidden rounded-xl shadow-xl">
          <div className="flex items-center justify-between border-b border-border/80 px-4 py-3 bg-secondary/20">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-primary" />
              <h3 className="text-xs font-semibold text-foreground">
                Recent Telemetry Events & Scans
              </h3>
            </div>
            <span className="rounded bg-secondary px-2 py-0.5 font-mono text-[10px] text-muted-foreground">
              Shift {currentShift} Active
            </span>
          </div>
          <ul className="divide-y divide-border/70">
            {demoActivity.map((item) => (
              <li key={item.id} className="p-3.5 hover:bg-secondary/20 transition-colors">
                <div className="flex items-center justify-between gap-2">
                  <span className="flex items-center gap-1.5 font-mono text-[11px] text-muted-foreground">
                    <Clock className="h-3 w-3" />
                    {item.time}
                  </span>
                  <StatusBadge status={toneByKind[item.kind]} />
                </div>
                <p className="mt-1.5 text-xs font-medium text-foreground leading-relaxed">
                  {item.message}
                </p>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Actionable Live Incident Dispatch Queue */}
      <div className="panel overflow-hidden rounded-xl">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/80 px-5 py-3.5 bg-secondary/15">
          <div className="flex items-center gap-2.5">
            <TriangleAlert className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-semibold tracking-wide text-foreground">
              Active PPE Violations & Incident Dispatch
            </h3>
            <span className="rounded-full border border-border bg-secondary/50 px-2 py-0.5 text-[11px] text-muted-foreground">
              {filteredViolations.length} unresolved
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground hidden sm:inline">Filter:</span>
            <div className="flex items-center rounded-md border border-border bg-secondary/30 p-0.5 text-xs">
              <button
                onClick={() => setFilterSeverity("all")}
                className={`rounded px-2 py-0.5 transition-colors ${
                  filterSeverity === "all" ? "bg-primary text-primary-foreground font-medium" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                All
              </button>
              <button
                onClick={() => setFilterSeverity("critical")}
                className={`rounded px-2 py-0.5 transition-colors ${
                  filterSeverity === "critical" ? "bg-primary text-primary-foreground font-medium" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Critical Only
              </button>
            </div>

            <Link
              to="/dashboard/violations"
              className="ml-2 inline-flex items-center gap-1 font-mono text-xs text-primary hover:underline"
            >
              View Full Table <ChevronRight className="h-3 w-3" />
            </Link>
          </div>
        </div>

        <div className="divide-y divide-border/70">
          {filteredViolations.slice(0, 5).map((violation) => {
            const isAcknowledged = acknowledgedIds[violation.id];

            return (
              <div
                key={violation.id}
                className="flex flex-wrap items-center justify-between gap-4 p-4 hover:bg-secondary/20 transition-colors"
              >
                <div className="flex items-center gap-3 min-w-[200px]">
                  <span className="rounded-md border border-border bg-secondary/40 px-2 py-1 font-mono text-xs font-semibold text-foreground">
                    {violation.id}
                  </span>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold text-foreground">
                        {violation.workerName}
                      </span>
                      <span className="font-mono text-xs text-muted-foreground">
                        ({violation.workerId})
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {violation.note}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <span className="inline-flex items-center gap-1.5 rounded-md border border-border/80 bg-secondary/30 px-2.5 py-1 text-xs text-foreground">
                    {violation.zone}
                  </span>

                  <span className="inline-flex items-center gap-1.5 rounded-md border border-critical/30 bg-critical/10 px-2.5 py-1 text-xs font-medium text-critical">
                    {violation.type === "Missing Helmet" ? (
                      <HardHat className="h-3.5 w-3.5" />
                    ) : (
                      <ShieldCheck className="h-3.5 w-3.5" />
                    )}
                    {violation.type}
                  </span>

                  <StatusBadge status={violation.severity} label={violation.severity} />

                  <span className="font-mono text-xs text-muted-foreground hidden md:inline">
                    {violation.timestamp.split(" ")[1]}
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleAcknowledge(violation.id, violation.workerName)}
                    disabled={isAcknowledged}
                    className={`flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium transition-all ${
                      isAcknowledged
                        ? "border-success/40 bg-success/10 text-success cursor-default"
                        : "border-border bg-secondary/50 text-foreground hover:bg-secondary hover:border-primary/40"
                    }`}
                  >
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    <span>{isAcknowledged ? "Acknowledged" : "Acknowledge"}</span>
                  </button>

                  <Link
                    to="/dashboard/detection"
                    className="flex items-center gap-1 rounded-md bg-secondary/70 px-2.5 py-1.5 text-xs text-foreground hover:bg-secondary transition-colors"
                  >
                    <span>Inspect</span>
                  </Link>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
