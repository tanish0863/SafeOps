import { TechLabel } from "@/components/common/TechLabel";
import type { ComplianceStatus, WorkspaceAnalysis } from "@/data/demoData";
import { cn } from "@/lib/utils";

const statTone: Record<ComplianceStatus, string> = {
  compliant: "text-success",
  violation: "text-critical",
  review: "text-warning",
};

const statMark: Record<ComplianceStatus, string> = {
  compliant: "bg-success",
  violation: "bg-critical",
  review: "bg-warning",
};

/** Compact result summary above the investigation details. */
export function DetectionSummary({ result }: { result: WorkspaceAnalysis }) {
  const count = (status: ComplianceStatus) => result.workers.filter((w) => w.compliance === status).length;

  const stats: { label: string; value: number; tone?: ComplianceStatus }[] = [
    { label: "Workers detected", value: result.workers.length },
    { label: "Compliant", value: count("compliant"), tone: "compliant" },
    { label: "Violations", value: count("violation"), tone: "violation" },
    { label: "Needs review", value: count("review"), tone: "review" },
  ];

  return (
    <div className="panel flex flex-wrap items-center gap-x-8 gap-y-2 rounded-md px-4 py-3">
      <TechLabel tone="accent">Frame summary</TechLabel>
      {stats.map((stat) => (
        <span key={stat.label} className="flex items-baseline gap-2">
          <span
            className={cn(
              "font-mono text-lg leading-none",
              stat.tone ? statTone[stat.tone] : "text-foreground",
            )}
          >
            {String(stat.value).padStart(2, "0")}
          </span>
          <span className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
            {stat.tone && <span className={cn("h-1.5 w-1.5 rounded-full", statMark[stat.tone])} aria-hidden />}
            {stat.label}
          </span>
        </span>
      ))}
    </div>
  );
}
