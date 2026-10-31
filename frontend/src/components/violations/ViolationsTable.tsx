import { Fragment } from "react";

import detectionImage from "@/assets/detection-frame.jpg";

import { StatusBadge } from "@/components/common/StatusBadge";
import type { ViolationRecord } from "@/data/demoData";
import { cn } from "@/lib/utils";

interface ViolationsTableProps {
  rows: ViolationRecord[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

const severityMark: Record<string, string> = {
  critical: "bg-critical",
  high: "bg-warning",
  medium: "bg-primary",
  low: "bg-muted-foreground",
};

/** Violation records table with an expandable evidence row. */
export function ViolationsTable({ rows, selectedId, onSelect }: ViolationsTableProps) {
  return (
    <div className="panel overflow-x-auto rounded-md">
      <table className="w-full min-w-[860px] border-collapse text-left">
        <thead>
          <tr className="border-b border-border bg-white/[0.03]">
            {["Evidence", "ID", "Worker", "Type", "Zone", "Timestamp", "Severity", "Review"].map((heading) => (
              <th
                key={heading}
                className="px-4 py-3 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground"
              >
                {heading}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <Fragment key={row.id}>
              <tr
                onClick={() => onSelect(selectedId === row.id ? null : row.id)}
                className={cn(
                  "cursor-pointer border-b border-border transition-colors",
                  selectedId === row.id
                    ? "bg-primary/[0.07] shadow-[inset_2px_0_0_var(--color-primary)]"
                    : "hover:bg-secondary/40",
                )}
              >
                <td className="px-4 py-3">
                  <img
                    src={detectionImage}
                    alt={`Evidence frame for ${row.id}`}
                    width={1280}
                    height={720}
                    loading="lazy"
                    className="h-10 w-14 rounded-sm border border-border object-cover"
                  />
                </td>
                <td className="px-4 py-3">
                  <span className="flex items-center gap-2">
                    <span
                      className={cn("h-2 w-2 shrink-0 rounded-full", severityMark[row.severity] ?? "bg-muted-foreground")}
                      aria-hidden
                    />
                    <span className="font-mono text-xs text-foreground">{row.id}</span>
                  </span>
                </td>
                <td className="px-4 py-3">
                  <p className="font-mono text-xs text-foreground">{row.workerId}</p>
                  <p className="text-xs text-muted-foreground">{row.workerName}</p>
                </td>
                <td className="px-4 py-3 text-sm text-foreground">{row.type}</td>
                <td className="px-4 py-3 text-sm text-muted-foreground">{row.zone}</td>
                <td className="px-4 py-3 font-mono text-xs text-muted-foreground">{row.timestamp}</td>
                <td className="px-4 py-3">
                  <StatusBadge
                    status={row.severity === "critical" || row.severity === "high" ? row.severity : "medium"}
                    label={row.severity}
                  />
                </td>
                <td className="px-4 py-3">
                  <StatusBadge status={row.reviewStatus} />
                </td>
              </tr>
              {selectedId === row.id && (
                <tr key={`${row.id}-evidence`} className="border-b border-border bg-primary/[0.04]">
                  <td colSpan={8} className="px-4 py-5">
                    <div className="flex flex-wrap items-start gap-6">
                      <img
                        src={detectionImage}
                        alt={`Evidence frame for ${row.id}`}
                        width={1280}
                        height={720}
                        loading="lazy"
                        className="corner-brackets w-72 rounded-sm border border-border object-cover"
                      />
                      <div className="max-w-lg">
                        <p className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.18em] text-primary">
                          <span className="h-px w-5 bg-primary/60" aria-hidden />
                          Evidence — {row.id}
                        </p>
                        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{row.note}</p>
                        <p className="mt-3 font-mono text-xs text-muted-foreground">
                          {row.zone} · {row.timestamp}
                        </p>
                      </div>
                    </div>
                  </td>
                </tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}
