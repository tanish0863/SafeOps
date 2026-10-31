import { useState, useMemo, useEffect } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import { Search, Users, UserPlus, Trash2, Sparkles, Filter, CheckCircle2, AlertOctagon, AlertTriangle } from "lucide-react";
import { toast } from "sonner";

import { EmptyState } from "@/components/common/EmptyState";
import { PageHeader } from "@/components/common/PageHeader";
import { TechLabel } from "@/components/common/TechLabel";
import { WorkerCard } from "@/components/workers/WorkerCard";
import { WorkerDetail } from "@/components/workers/WorkerDetail";
import { AddWorkerModal } from "@/components/workers/AddWorkerModal";
import type { Worker, ComplianceStatus } from "@/data/demoData";
import {
  getRealtimeWorkers,
  subscribeToRealtimeWorkers,
  removeRealtimeWorker,
  clearRealtimeWorkers,
  syncWorkersFromBackend,
} from "@/lib/workerStore";

export const Route = createFileRoute("/dashboard/workers")({
  head: () => ({
    meta: [
      { title: "Personnel Registry — SafeOps Dashboard" },
      { name: "description", content: "Real-time worker roster with live PPE compliance and shift records." },
      { property: "og:title", content: "Personnel Registry — SafeOps Dashboard" },
      { property: "og:description", content: "Real-time worker roster with live PPE compliance and shift records." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: WorkersPage,
});

type WorkerFilter = "all" | ComplianceStatus;

const filterOptions: { label: string; value: WorkerFilter }[] = [
  { label: "All Personnel", value: "all" },
  { label: "Compliant", value: "compliant" },
  { label: "Violation", value: "violation" },
  { label: "Needs Review", value: "review" },
];

function WorkersPage() {
  const [allWorkers, setAllWorkers] = useState<Worker[]>(() => getRealtimeWorkers());
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<WorkerFilter>("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);

  // Subscribe to real-time worker updates and hydrate from backend
  useEffect(() => {
    setAllWorkers(getRealtimeWorkers());
    void syncWorkersFromBackend().then((hydrated) => {
      setAllWorkers(Array.isArray(hydrated) ? hydrated : []);
    });
    return subscribeToRealtimeWorkers((updated) => {
      setAllWorkers(updated);
    });
  }, []);

  // Filtered worker list
  const filteredWorkers = useMemo(() => {
    const q = query.trim().toLowerCase();
    return allWorkers.filter((worker) => {
      const matchesQuery =
        q.length === 0 || [worker.id, worker.name, worker.zone].some((field) => field.toLowerCase().includes(q));
      const matchesFilter = filter === "all" || worker.status === filter;
      return matchesQuery && matchesFilter;
    });
  }, [allWorkers, query, filter]);

  // Keep selected ID valid
  useEffect(() => {
    if (filteredWorkers.length > 0) {
      if (!selectedId || !filteredWorkers.some((w) => w.id === selectedId)) {
        setSelectedId(filteredWorkers[0].id);
      }
    } else {
      setSelectedId(null);
    }
  }, [filteredWorkers, selectedId]);

  const selectedWorker = allWorkers.find((w) => w.id === selectedId) ?? null;

  const handleRemoveWorker = (id: string) => {
    const updated = removeRealtimeWorker(id);
    setAllWorkers(updated);
    toast.info("Worker removed from registry", { description: `Record ${id} has been deleted.` });
  };

  const handleClearAll = () => {
    if (allWorkers.length === 0) return;
    if (window.confirm("Are you sure you want to clear all entered workers from the registry?")) {
      clearRealtimeWorkers();
      setAllWorkers([]);
      setSelectedId(null);
      toast.info("Registry cleared", { description: "All real-time personnel data has been reset." });
    }
  };

  return (
    <div className="space-y-7">
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-border/80 pb-6">
        <div>
          <PageHeader
            eyebrow="Live Personnel Registry"
            title="Workers"
            description="Real-time personnel directory with live optical PPE compliance. New entries intake directly from detection scans or manual enrollment."
          />
        </div>

        {/* Top Header Actions */}
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            onClick={() => setIsAddModalOpen(true)}
            className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-xs font-semibold text-primary-foreground shadow-md hover:bg-primary/90 transition-all active:scale-[0.99]"
          >
            <UserPlus className="h-4 w-4" />
            <span>Add Worker</span>
          </button>

          <Link
            to="/dashboard/detection"
            className="flex items-center gap-2 rounded-lg border border-border bg-secondary/50 px-3 py-2 text-xs font-medium text-foreground hover:bg-secondary transition-colors"
          >
            <Sparkles className="h-3.5 w-3.5 text-primary" />
            <span>Vision Studio</span>
          </Link>

          {allWorkers.length > 0 && (
            <button
              onClick={handleClearAll}
              title="Reset registry to empty"
              className="flex items-center gap-1.5 rounded-lg border border-border px-3 py-2 text-xs text-muted-foreground hover:text-critical hover:bg-critical/10 hover:border-critical/30 transition-colors"
            >
              <Trash2 className="h-3.5 w-3.5" />
              <span>Clear All</span>
            </button>
          )}
        </div>
      </div>

      {/* Roster Counts Strip */}
      {allWorkers.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="panel p-3.5 rounded-lg flex items-center justify-between">
            <div>
              <span className="text-[11px] text-muted-foreground uppercase font-medium">Total Registered</span>
              <p className="mt-0.5 text-xl font-bold text-foreground font-mono">{allWorkers.length}</p>
            </div>
            <Users className="h-4 w-4 text-primary opacity-60" />
          </div>

          <div className="panel p-3.5 rounded-lg flex items-center justify-between">
            <div>
              <span className="text-[11px] text-muted-foreground uppercase font-medium">Compliant</span>
              <p className="mt-0.5 text-xl font-bold text-success font-mono">
                {allWorkers.filter((w) => w.status === "compliant").length}
              </p>
            </div>
            <CheckCircle2 className="h-4 w-4 text-success opacity-60" />
          </div>

          <div className="panel p-3.5 rounded-lg flex items-center justify-between">
            <div>
              <span className="text-[11px] text-muted-foreground uppercase font-medium">Violations</span>
              <p className="mt-0.5 text-xl font-bold text-critical font-mono">
                {allWorkers.filter((w) => w.status === "violation").length}
              </p>
            </div>
            <AlertOctagon className="h-4 w-4 text-critical opacity-60" />
          </div>

          <div className="panel p-3.5 rounded-lg flex items-center justify-between">
            <div>
              <span className="text-[11px] text-muted-foreground uppercase font-medium">Needs Review</span>
              <p className="mt-0.5 text-xl font-bold text-warning font-mono">
                {allWorkers.filter((w) => w.status === "review").length}
              </p>
            </div>
            <AlertTriangle className="h-4 w-4 text-warning opacity-60" />
          </div>
        </div>
      )}

      {/* Search and Filters Bar */}
      {allWorkers.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-4 border border-border/80 bg-secondary/20 p-3.5 rounded-xl">
          <div className="min-w-[260px] flex-1">
            <div className="flex items-center rounded-lg border border-border bg-background/80 px-3 py-1.5 focus-within:border-primary">
              <Search className="h-4 w-4 text-muted-foreground shrink-0" />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search real-time workers by ID, name or zone…"
                className="w-full bg-transparent px-2.5 text-xs text-foreground outline-none placeholder:text-muted-foreground"
              />
            </div>
          </div>

          <div className="flex items-center gap-1.5 rounded-lg border border-border/70 bg-secondary/40 p-1">
            {filterOptions.map((opt) => (
              <button
                key={opt.value}
                onClick={() => setFilter(opt.value)}
                className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                  filter === opt.value
                    ? "bg-primary text-primary-foreground shadow-sm font-semibold"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Main Content Area */}
      {allWorkers.length === 0 ? (
        /* Empty State when no real-time workers entered yet */
        <div className="panel rounded-xl p-12 text-center border-dashed border-2 border-border/80">
          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 border border-primary/30 text-primary mb-4">
            <Users className="h-7 w-7" />
          </div>
          <h3 className="text-base font-bold text-foreground sm:text-lg">No Personnel Records Entered Yet</h3>
          <p className="mt-1.5 text-xs text-muted-foreground max-w-md mx-auto leading-relaxed">
            The real-time registry is currently empty. You can register workers manually with their assigned zones, shifts, and PPE state, or run detection scans in the Vision Studio.
          </p>

          <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
            <button
              onClick={() => setIsAddModalOpen(true)}
              className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-xs font-semibold text-primary-foreground shadow-md hover:bg-primary/90 transition-all active:scale-[0.99]"
            >
              <UserPlus className="h-4 w-4" />
              <span>Enter Worker Data</span>
            </button>

            <Link
              to="/dashboard/detection"
              className="flex items-center gap-2 rounded-lg border border-border bg-secondary/50 px-4 py-2 text-xs font-medium text-foreground hover:bg-secondary transition-colors"
            >
              <Sparkles className="h-4 w-4 text-primary" />
              <span>Intake via Vision Studio</span>
            </Link>
          </div>
        </div>
      ) : filteredWorkers.length > 0 ? (
        /* Grid with Real-Time Worker Cards and Details */
        <div className="grid gap-6 lg:grid-cols-5">
          <div className="space-y-4 lg:col-span-3">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>{filteredWorkers.length} personnel matching query</span>
              <span>Click a card to inspect compliance</span>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              {filteredWorkers.map((worker) => (
                <WorkerCard
                  key={worker.id}
                  worker={worker}
                  selected={worker.id === selectedId}
                  onSelect={() => setSelectedId(worker.id)}
                />
              ))}
            </div>
          </div>

          <div className="lg:col-span-2">
            <div className="lg:sticky lg:top-20">
              <WorkerDetail worker={selectedWorker} onDelete={handleRemoveWorker} />
            </div>
          </div>
        </div>
      ) : (
        /* Empty search/filter match */
        <EmptyState
          icon={Users}
          title="No workers match filters"
          description="No personnel record matches the search query or status filter."
          action={
            <button
              onClick={() => {
                setQuery("");
                setFilter("all");
              }}
              className="rounded-lg border border-border px-4 py-2 text-xs font-medium text-foreground hover:bg-secondary"
            >
              Clear filters
            </button>
          }
        />
      )}

      {/* Add Worker Modal */}
      <AddWorkerModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onWorkerAdded={(worker) => {
          setSelectedId(worker.id);
        }}
      />
    </div>
  );
}
