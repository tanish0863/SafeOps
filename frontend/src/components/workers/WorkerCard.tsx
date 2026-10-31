import { CheckCircle2, XCircle, HardHat, ShieldCheck, User } from "lucide-react";

import { StatusBadge } from "@/components/common/StatusBadge";
import { TechLabel } from "@/components/common/TechLabel";
import type { Worker } from "@/data/demoData";
import { cn } from "@/lib/utils";

interface WorkerCardProps {
  worker: Worker;
  selected: boolean;
  onSelect: () => void;
}

/** Worker listing card: identification, PPE compliance chips, authentic status strip. */
export function WorkerCard({ worker, selected, onSelect }: WorkerCardProps) {
  return (
    <button
      onClick={onSelect}
      aria-pressed={selected}
      className={cn(
        "panel hover-lift group overflow-hidden rounded-md text-left transition-shadow",
        selected
          ? "border-primary/60 shadow-[0_0_0_1px_oklch(0.78_0.15_72/45%),0_18px_40px_-20px_oklch(0_0_0/60%)]"
          : "hover:border-foreground/25",
      )}
    >
      <div className="relative h-24 overflow-hidden border-b border-border bg-gradient-to-br from-card via-secondary/40 to-muted/20 p-3">
        {worker.imageUrl ? (
          <img
            src={worker.imageUrl}
            alt=""
            loading="lazy"
            className="absolute inset-0 h-full w-full object-cover opacity-80 transition-all duration-300 group-hover:scale-[1.03] group-hover:opacity-95"
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-between px-4 bg-gradient-to-r from-primary/5 via-transparent to-primary/10">
            <div className="flex items-center gap-2.5">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-primary/30 bg-primary/10 text-primary font-mono font-bold text-sm">
                {worker.name.split(" ").map((n) => n[0]).join("").slice(0, 2).toUpperCase() || worker.id.slice(-2)}
              </div>
              <div>
                <span className="font-mono text-xs font-bold text-foreground block">{worker.id}</span>
                <span className="text-[10px] text-muted-foreground uppercase">{worker.zone}</span>
              </div>
            </div>
            <div className="text-right">
              <span className="font-mono text-[10px] text-muted-foreground block">{worker.lastScan || "Recent"}</span>
              <span className="font-mono text-[10px] text-primary">{worker.scans} scan{worker.scans === 1 ? "" : "s"}</span>
            </div>
          </div>
        )}
        <div className="absolute bottom-2 left-3">
          <StatusBadge status={worker.status} icon />
        </div>
      </div>

      <div className="p-4">
        <div className="flex items-center justify-between">
          <p className="font-mono text-xs text-primary">{worker.id}</p>
          <TechLabel>{worker.shift} shift</TechLabel>
        </div>
        <p className="mt-1 text-base font-semibold text-foreground">{worker.name}</p>
        <TechLabel className="mt-1 block">{worker.zone}</TechLabel>

        <div className="mt-3.5 flex gap-2">
          <PpeChip label="Helmet" present={worker.helmet} />
          <PpeChip label="Vest" present={worker.vest} />
        </div>
      </div>
    </button>
  );
}

function PpeChip({ label, present }: { label: string; present: boolean }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-sm border px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.12em]",
        present
          ? "border-success/30 bg-success/10 text-success"
          : "border-critical/35 bg-critical/10 text-critical",
      )}
    >
      {present ? (
        <CheckCircle2 className="h-3 w-3" aria-hidden />
      ) : (
        <XCircle className="h-3 w-3" aria-hidden />
      )}
      {label}
    </span>
  );
}
