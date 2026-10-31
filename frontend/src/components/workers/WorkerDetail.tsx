import { CheckCircle2, UserSearch, XCircle, Trash2 } from "lucide-react";

import { EmptyState } from "@/components/common/EmptyState";
import { StatusBadge } from "@/components/common/StatusBadge";
import { TechLabel } from "@/components/common/TechLabel";
import type { Worker } from "@/data/demoData";
import { cn } from "@/lib/utils";

interface WorkerDetailProps {
  worker: Worker | null;
  onDelete?: (id: string) => void;
}

const ppeTone = {
  present: { icon: CheckCircle2, text: "text-success", chip: "border-success/30 bg-success/10 text-success" },
  missing: { icon: XCircle, text: "text-critical", chip: "border-critical/35 bg-critical/10 text-critical" },
} as const;

/** Worker detail view: identification, PPE information, evidence area. */
export function WorkerDetail({ worker, onDelete }: WorkerDetailProps) {
  if (!worker) {
    return (
      <EmptyState
        icon={UserSearch}
        title="Select a worker"
        description="Choose a worker from the list to see PPE compliance details and evidence."
        className="h-full"
      />
    );
  }

  return (
    <div className="panel rounded-md">
      <div className="border-b border-border px-5 py-4">
        <div className="flex items-center justify-between gap-3">
          <p className="font-mono text-sm text-primary">{worker.id}</p>
          <StatusBadge status={worker.status} icon />
        </div>
        <p className="mt-1 text-lg font-semibold text-foreground">{worker.name}</p>
        <TechLabel className="mt-1 block">
          {worker.zone} · Shift {worker.shift}
        </TechLabel>
      </div>

      <div className="grid grid-cols-2 gap-px border-b border-border bg-white/[0.06]">
        {[
          { label: "Last scan", value: worker.lastScan },
          { label: "Total scans", value: String(worker.scans) },
        ].map((item) => (
          <div key={item.label} className="bg-card/80 p-4">
            <TechLabel>{item.label}</TechLabel>
            <p className="mt-1 font-mono text-sm text-foreground">{item.value}</p>
          </div>
        ))}
      </div>

      <div className="border-b border-border px-5 py-4">
        <TechLabel>Associated PPE</TechLabel>
        <ul className="mt-3 space-y-2">
          {[
            { name: "Helmet", present: worker.helmet },
            { name: "Safety Vest", present: worker.vest },
          ].map((item) => {
            const tone = ppeTone[item.present ? "present" : "missing"];
            const Icon = tone.icon;
            return (
              <li key={item.name} className="flex items-center justify-between gap-2 text-sm">
                <span className="flex items-center gap-2 text-foreground">
                  <Icon className={cn("h-4 w-4", tone.text)} aria-hidden />
                  {item.name}
                </span>
                <span
                  className={cn(
                    "rounded-sm border px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.12em]",
                    tone.chip,
                  )}
                >
                  {item.present ? "Detected" : "Missing"}
                </span>
              </li>
            );
          })}
        </ul>
      </div>

      <div className="p-5">
        <div className="corner-brackets panel-sunken rounded-sm p-4">
          {worker.imageUrl ? (
            <img
              src={worker.imageUrl}
              alt={`Evidence frame for ${worker.id}`}
              width={1280}
              height={720}
              loading="lazy"
              className="h-auto w-full rounded-sm object-cover"
            />
          ) : (
            <div className="flex flex-col items-center justify-center p-6 text-center bg-black/30 border border-border/50 rounded">
              <div className="h-12 w-12 rounded-full border border-primary/40 bg-primary/10 flex items-center justify-center text-primary font-mono text-base font-bold mb-3">
                {worker.name.split(" ").map((n) => n[0]).join("").slice(0, 2).toUpperCase() || worker.id.slice(-2)}
              </div>
              <p className="font-mono text-xs text-foreground font-semibold">{worker.name} ({worker.id})</p>
              <p className="text-[11px] text-muted-foreground mt-0.5">{worker.zone} · Shift {worker.shift}</p>
              <div className="mt-3 inline-flex items-center gap-2 rounded border border-border/70 bg-card/60 px-3 py-1 text-[11px] font-mono text-muted-foreground">
                <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
                <span>Recorded scan: {worker.lastScan || "Recent"}</span>
              </div>
            </div>
          )}
          <div className="px-1 pt-3">
            <TechLabel>Personnel Telemetry · {worker.id}</TechLabel>
          </div>
        </div>
        {worker.needsReview && (
          <div className="mt-4 flex items-start gap-3 rounded-sm border border-warning/40 bg-warning/10 px-4 py-3">
            <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-warning" aria-hidden />
            <div>
              <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-warning">Needs review</p>
              <p className="mt-1 text-sm text-muted-foreground">
                A detection for this worker had low confidence and was routed for manual confirmation.
              </p>
            </div>
          </div>
        )}

        {onDelete && (
          <div className="mt-5 pt-3 border-t border-border/70 flex justify-end">
            <button
              onClick={() => onDelete(worker.id)}
              className="flex items-center gap-1.5 rounded-md border border-critical/30 bg-critical/10 px-3 py-1.5 text-xs text-critical hover:bg-critical/20 transition-colors font-medium"
            >
              <Trash2 className="h-3.5 w-3.5" />
              <span>Remove from Roster</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
