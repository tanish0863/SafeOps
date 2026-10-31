import type { DetectedWorker } from "@/data/demoData";
import { cn } from "@/lib/utils";

const toneClasses: Record<DetectedWorker["status"], { frame: string; chip: string }> = {
  compliant: {
    frame: "border-success shadow-[0_0_0_1px_oklch(0.68_0.13_155/20%)]",
    chip: "border-success/50 bg-background/85 text-success",
  },
  violation: {
    frame: "border-critical shadow-[0_0_0_1px_oklch(0.62_0.21_25/25%)]",
    chip: "border-critical/50 bg-background/85 text-critical",
  },
  review: {
    frame: "border-warning shadow-[0_0_0_1px_oklch(0.8_0.13_80/20%)]",
    chip: "border-warning/50 bg-background/85 text-warning",
  },
};

/**
 * Renders worker bounding-box overlays. Boxes are stored as percentages of the
 * frame, so the same data scales across the landing visuals and the dashboard
 * detection preview.
 */
export function DetectionBoxes({ workers, showLabels = true }: { workers: DetectedWorker[]; showLabels?: boolean }) {
  return (
    <div className="pointer-events-none absolute inset-0" aria-hidden>
      {workers.map((worker) => {
        const tone = toneClasses[worker.status];
        return (
          <div
            key={worker.workerId}
            className="absolute"
            style={{
              left: `${worker.box.x}%`,
              top: `${worker.box.y}%`,
              width: `${worker.box.w}%`,
              height: `${worker.box.h}%`,
            }}
          >
            <div className={cn("h-full w-full border-2", tone.frame)} />
            {showLabels && (
              <div
                className={cn(
                  "absolute -top-5 left-0 flex items-center gap-2 whitespace-nowrap border px-1.5 py-px font-mono text-[9px] uppercase tracking-[0.12em]",
                  tone.chip,
                )}
              >
                {worker.workerId}
                {worker.missing.length > 0 && <span>· missing {worker.missing.join(", ")}</span>}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
