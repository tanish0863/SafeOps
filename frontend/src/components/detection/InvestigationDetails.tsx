import { useState } from "react";
import { ChevronDown } from "lucide-react";

import { TechLabel } from "@/components/common/TechLabel";
import type { PpeRules, WorkspaceAnalysis } from "@/data/demoData";
import type { WorkspaceImage, DetectionWorkspaceState } from "@/hooks/useDetectionFlow";
import { cn } from "@/lib/utils";

interface InvestigationDetailsProps {
  image: WorkspaceImage | null;
  result: WorkspaceAnalysis | null;
  state: DetectionWorkspaceState;
  rules: PpeRules;
}

/** Collapsible run metadata: file, timing, status, model and enabled rules. */
export function InvestigationDetails({ image, result, state, rules }: InvestigationDetailsProps) {
  const [open, setOpen] = useState(false);

  const rows: { label: string; value: string }[] = [
    { label: "Image file", value: image?.file.name ?? "—" },
    { label: "Upload time", value: image?.uploadedAt ?? "—" },
    {
      label: "Analysis status",
      value:
        state === "complete"
          ? "Complete (demo result)"
          : state === "processing"
            ? "Processing"
            : state === "error"
              ? "Failed"
              : state === "ready"
                ? "Awaiting analysis"
                : "No image",
    },
    { label: "Detection model", value: result?.model ?? "Not connected" },
    {
      label: "Enabled PPE rules",
      value: [rules.helmetRequired && "Helmet required", rules.vestRequired && "Safety vest required"]
        .filter(Boolean)
        .join(" · ") || "None enabled",
    },
    { label: "Job ID", value: result?.jobId ?? "—" },
  ];

  return (
    <section className="border border-border bg-card">
      <button
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="flex w-full items-center justify-between px-4 py-2.5 transition-colors hover:bg-secondary/60"
      >
        <TechLabel>Investigation details</TechLabel>
        <ChevronDown className={cn("h-4 w-4 text-muted-foreground transition-transform", open && "rotate-180")} aria-hidden />
      </button>

      {open && (
        <dl className="grid gap-x-8 gap-y-3 border-t border-border px-4 py-4 sm:grid-cols-2">
          {rows.map((row) => (
            <div key={row.label} className="min-w-0">
              <dt>
                <TechLabel>{row.label}</TechLabel>
              </dt>
              <dd className="mt-1 truncate text-sm text-foreground">{row.value}</dd>
            </div>
          ))}
        </dl>
      )}
    </section>
  );
}
