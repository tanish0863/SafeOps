import { useMemo, useState, useEffect } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Search } from "lucide-react";

import { PageHeader } from "@/components/common/PageHeader";
import { TechLabel } from "@/components/common/TechLabel";
import { EmptyState } from "@/components/common/EmptyState";
import { ViolationsTable } from "@/components/violations/ViolationsTable";
import { demoViolations, type Severity, type ViolationRecord } from "@/data/demoData";
import { fetchViolations } from "@/services/detectionService";

export const Route = createFileRoute("/dashboard/violations")({
  head: () => ({
    meta: [
      { title: "Violations — SafeOps Dashboard" },
      { name: "description", content: "Browse and filter recorded PPE violations by worker, type, zone and severity." },
      { property: "og:title", content: "Violations — SafeOps Dashboard" },
      { property: "og:description", content: "Browse and filter recorded PPE violations by worker, type, zone and severity." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: ViolationsPage,
});

const severityOptions: Array<Severity | "all"> = ["all", "critical", "high", "medium", "low"];

/** Violation records with search, severity filter and an expandable evidence row. */
function ViolationsPage() {
  const [query, setQuery] = useState("");
  const [severity, setSeverity] = useState<Severity | "all">("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [violations, setViolations] = useState<ViolationRecord[]>([]);

  // Fetch real violations from backend
  useEffect(() => {
    fetchViolations(severity)
      .then((data) => {
        setViolations(Array.isArray(data) ? data : []);
      })
      .catch(() => {
        setViolations([]);
      });
  }, [severity]);

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return violations.filter((row) => {
      const matchesQuery =
        q.length === 0 ||
        [row.id, row.workerId, row.workerName, row.type, row.zone].some((field) =>
          field.toLowerCase().includes(q),
        );
      const matchesSeverity = severity === "all" || row.severity === severity;
      return matchesQuery && matchesSeverity;
    });
  }, [violations, query, severity]);

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Safety Records"
        title="Violations"
        description="Every flagged PPE breach with its evidence frame, severity and review state. Select a row to expand the evidence trail."
      />

      <div className="flex flex-wrap items-end gap-4 border border-border bg-card p-4">
        <div className="min-w-[240px] flex-1">
          <TechLabel>Search</TechLabel>
          <div className="mt-2 flex items-center border border-border bg-background px-3">
            <Search className="h-4 w-4 text-muted-foreground" aria-hidden />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="ID, worker, type or zone…"
              aria-label="Search violations"
              className="w-full bg-transparent px-3 py-2 text-sm text-foreground outline-none placeholder:text-muted-foreground"
            />
          </div>
        </div>
        <div>
          <TechLabel>Severity</TechLabel>
          <div className="mt-2 flex gap-px bg-border">
            {severityOptions.map((option) => (
              <button
                key={option}
                onClick={() => setSeverity(option)}
                aria-pressed={severity === option}
                className={
                  "px-3 py-2 font-mono text-[11px] uppercase tracking-[0.14em] transition-colors " +
                  (severity === option
                    ? "bg-secondary text-foreground"
                    : "bg-card text-muted-foreground hover:text-foreground")
                }
              >
                {option}
              </button>
            ))}
          </div>
        </div>
      </div>

      {rows.length > 0 ? (
        <ViolationsTable rows={rows} selectedId={selectedId} onSelect={setSelectedId} />
      ) : (
        <EmptyState
          icon={Search}
          title="No violations match"
          description="No recorded violation matches the current search and severity filter. Clear them to see the full record set."
          action={
            <button
              onClick={() => {
                setQuery("");
                setSeverity("all");
              }}
              className="border border-border px-4 py-2 font-mono text-[11px] uppercase tracking-[0.16em] text-foreground transition-colors hover:bg-secondary"
            >
              Clear filters
            </button>
          }
        />
      )}
    </div>
  );
}
