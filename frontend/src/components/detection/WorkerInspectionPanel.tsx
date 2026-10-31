import { ListChecks, MousePointerClick, ShieldAlert } from "lucide-react";

import { EmptyState } from "@/components/common/EmptyState";
import { WorkerResultItem } from "@/components/detection/WorkerResultItem";
import type { WorkspaceAnalysis } from "@/data/demoData";

interface WorkerInspectionPanelProps {
  result: WorkspaceAnalysis | null;
  selectedWorkerId: string | null;
  onSelectWorker: (workerId: string | null) => void;
}

/** Right panel: individual worker inspection, synchronized with the canvas. */
export function WorkerInspectionPanel({ result, selectedWorkerId, onSelectWorker }: WorkerInspectionPanelProps) {
  const workers = result?.workers ?? [];

  return (
    <section className="panel flex h-full flex-col rounded-xl overflow-hidden shadow-xl border border-border/80">
      <header className="flex items-center justify-between border-b border-border/80 px-4 py-3 bg-secondary/30">
        <div className="flex items-center gap-2">
          <ListChecks className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold tracking-tight text-foreground">
            Worker Inspection
          </h3>
        </div>
        <span className="rounded-full border border-primary/40 bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary">
          {result ? `${workers.length} Detected` : "—"}
        </span>
      </header>

      {!result ? (
        <EmptyState
          icon={ListChecks}
          title="No analysis yet"
          description="Run an analysis or upload a video to inspect each detected worker's PPE status, confidence and compliance."
          className="border-0 bg-transparent py-12"
        />
      ) : (
        <>
          <div className="flex flex-1 flex-col divide-y divide-border/70">
            {workers.map((worker) => (
              <WorkerResultItem
                key={worker.workerId}
                worker={worker}
                selected={worker.workerId === selectedWorkerId}
                onSelect={() => onSelectWorker(worker.workerId === selectedWorkerId ? null : worker.workerId)}
              />
            ))}
          </div>
          <footer className="flex items-center gap-2 border-t border-border/80 px-4 py-2.5 bg-secondary/15 text-xs text-muted-foreground">
            <MousePointerClick className="h-3.5 w-3.5 text-primary" aria-hidden />
            <span>Select a worker or bounding box to inspect</span>
          </footer>
        </>
      )}
    </section>
  );
}
