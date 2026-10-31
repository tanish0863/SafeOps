import { AlertTriangle, CheckCircle2, HelpCircle } from "lucide-react";

import type { PpeDetection, PpeStatus, WorkspaceWorker } from "@/data/demoData";
import { cn } from "@/lib/utils";

const ppeTone: Record<PpeStatus, { text: string; mark: string; icon: typeof CheckCircle2 }> = {
  detected: { text: "text-success", mark: "border-success/50 bg-background/85 text-success", icon: CheckCircle2 },
  missing: { text: "text-critical", mark: "border-critical/50 bg-background/85 text-critical", icon: AlertTriangle },
  uncertain: { text: "text-warning", mark: "border-warning/50 bg-background/85 text-warning", icon: HelpCircle },
};

const ppeLabel: Record<PpeStatus, (kind: PpeDetection["kind"]) => string> = {
  detected: (kind) => (kind === "helmet" ? "Helmet" : "Safety vest"),
  missing: (kind) => (kind === "helmet" ? "Missing helmet" : "Missing vest"),
  uncertain: (kind) => (kind === "helmet" ? "Helmet unconfirmed" : "Vest unconfirmed"),
};

const workerTone: Record<WorkspaceWorker["compliance"], string> = {
  compliant: "border-success",
  violation: "border-critical",
  review: "border-warning",
};

/**
 * Normalized-coordinate bounding-box overlay. Boxes are stored as 0-1 values
 * relative to the image, so the same layer scales with any frame size and maps
 * directly onto a real backend response.
 */
export function BoundingBoxOverlay({
  workers,
  selectedWorkerId,
  onSelect,
}: {
  workers: WorkspaceWorker[];
  selectedWorkerId: string | null;
  onSelect: (workerId: string) => void;
}) {
  return (
    <div className="pointer-events-none absolute inset-0">
      {workers.map((worker) => {
        const selected = worker.workerId === selectedWorkerId;
        const dimmed = selectedWorkerId !== null && !selected;

        return (
          <div
            key={worker.workerId}
            className="absolute"
            style={{
              left: `${worker.box.x * 100}%`,
              top: `${worker.box.y * 100}%`,
              width: `${worker.box.w * 100}%`,
              height: `${worker.box.h * 100}%`,
            }}
          >
            <button
              onClick={(event) => {
                event.stopPropagation();
                onSelect(worker.workerId);
              }}
              aria-label={`Select ${worker.workerId}`}
              aria-pressed={selected}
              className={cn(
                "pointer-events-auto absolute inset-0 cursor-pointer border-2 transition-all",
                workerTone[worker.compliance],
                selected ? "bg-primary/10 outline outline-2 -outline-offset-2 outline-primary" : "hover:bg-primary/10",
                dimmed && "opacity-45",
              )}
            />

            <span
              className={cn(
                "pointer-events-none absolute -top-6 left-0 whitespace-nowrap rounded border px-2 py-0.5 text-[11px] font-semibold tracking-tight shadow-md backdrop-blur-sm",
                selected ? "border-primary bg-primary text-primary-foreground" : "border-border/80 bg-background/90 text-foreground",
                dimmed && "opacity-60",
              )}
            >
              {worker.workerId}
            </span>

            {(["helmet", "vest"] as const).map((kind, index) => {
              const item = worker[kind];
              if (!item.evaluated) return null;
              const tone = ppeTone[item.status];
              const Icon = tone.icon;
              return (
                <span
                  key={kind}
                  className={cn(
                    "pointer-events-none absolute left-full top-0 ml-1.5 flex items-center gap-1.5 whitespace-nowrap rounded border px-2 py-0.5 text-[10px] font-semibold tracking-tight shadow-md backdrop-blur-sm",
                    tone.mark,
                    dimmed && "opacity-60",
                    index === 1 && "top-1/2",
                  )}
                  style={index === 1 ? { transform: "translateY(-50%)" } : undefined}
                >
                  <Icon className="h-3 w-3" aria-hidden />
                  {ppeLabel[item.status](kind)}
                  <span className="font-mono text-[9px] opacity-75">{Math.round(item.confidence * 100)}%</span>
                </span>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}
