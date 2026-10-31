import { AlertTriangle, CheckCircle2, HelpCircle, HardHat, ShieldCheck } from "lucide-react";

import { StatusBadge } from "@/components/common/StatusBadge";
import type { WorkspaceWorker } from "@/data/demoData";
import { cn } from "@/lib/utils";

const ppeRow: Record<"detected" | "missing" | "uncertain", { icon: typeof CheckCircle2; text: string; bg: string }> = {
  detected: { icon: CheckCircle2, text: "text-success", bg: "bg-success/10 border-success/30" },
  missing: { icon: AlertTriangle, text: "text-critical", bg: "bg-critical/10 border-critical/30" },
  uncertain: { icon: HelpCircle, text: "text-warning", bg: "bg-warning/10 border-warning/30" },
};

function PpeStatusPill({ item, name }: { item: WorkspaceWorker["helmet"]; name: string }) {
  const tone = ppeRow[item.status];
  const Icon = tone.icon;
  const label =
    item.status === "detected"
      ? `${name} OK`
      : item.status === "missing"
      ? `Missing ${name}`
      : `${name} Unconfirmed`;

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-medium transition-colors",
        tone.bg,
        tone.text,
      )}
    >
      <Icon className="h-3.5 w-3.5 shrink-0" aria-hidden />
      <span>{label}</span>
      <span className="font-mono text-[11px] opacity-80">{Math.round(item.confidence * 100)}%</span>
    </span>
  );
}

/** One worker's inspection result in the right panel. */
export function WorkerResultItem({
  worker,
  selected,
  onSelect,
}: {
  worker: WorkspaceWorker;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      aria-pressed={selected}
      className={cn(
        "block w-full border-b border-border/70 p-4 text-left transition-all duration-200 last:border-0 hover:bg-secondary/40",
        selected
          ? "bg-primary/[0.08] shadow-[inset_3px_0_0_var(--color-primary)] hover:bg-primary/[0.12]"
          : "",
      )}
    >
      {/* Worker Header Line */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span
            className={cn(
              "text-sm font-semibold tracking-tight transition-colors",
              selected ? "text-primary" : "text-foreground",
            )}
          >
            {worker.workerId}
          </span>
          <span className="rounded bg-secondary/70 px-1.5 py-0.5 text-[11px] text-muted-foreground font-mono">
            Tracked
          </span>
        </div>
        <StatusBadge status={worker.compliance} />
      </div>

      {/* PPE Status Chips */}
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <PpeStatusPill item={worker.helmet} name="Helmet" />
        <PpeStatusPill item={worker.vest} name="Vest" />
      </div>

      {/* Confidence & Review Notes */}
      <div className="mt-3 space-y-1.5 text-xs text-muted-foreground">
        <div className="flex items-center justify-between">
          <span>Person Detection: <strong className="text-foreground">{Math.round(worker.personConfidence * 100)}%</strong></span>
          <span>PPE Match: <strong className="text-foreground">{Math.round(worker.associationConfidence * 100)}%</strong></span>
        </div>

        {worker.reviewReason && (
          <div className="mt-2 flex items-start gap-1.5 rounded-md border border-warning/30 bg-warning/10 p-2 text-xs text-warning leading-relaxed">
            <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
            <span>{worker.reviewReason}</span>
          </div>
        )}
      </div>
    </button>
  );
}
