import { RotateCcw, ScanLine } from "lucide-react";

import { TechLabel } from "@/components/common/TechLabel";
import type { PpeRules } from "@/data/demoData";
import type { DetectionWorkspaceState } from "@/hooks/useDetectionFlow";
import { cn } from "@/lib/utils";

interface DetectionControlsProps {
  state: DetectionWorkspaceState;
  hasImage: boolean;
  rules: PpeRules;
  onToggleRule: (key: keyof PpeRules) => void;
  onAnalyze: () => void;
  onReset: () => void;
}

function RuleToggle({
  label,
  hint,
  checked,
  onToggle,
}: {
  label: string;
  hint: string;
  checked: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      onClick={onToggle}
      className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left transition-colors hover:bg-secondary/60"
    >
      <span className="min-w-0">
        <span className="block text-sm text-foreground">{label}</span>
        <span className="mt-0.5 block text-xs text-muted-foreground">{hint}</span>
      </span>
      <span
        className={cn(
          "relative h-5 w-9 shrink-0 border transition-colors",
          checked ? "border-primary/60 bg-primary/25" : "border-border bg-secondary",
        )}
      >
        <span
          className={cn(
            "absolute top-1/2 h-3.5 w-3.5 -translate-y-1/2 transition-all",
            checked ? "left-[18px] bg-primary" : "left-[2px] bg-muted-foreground",
          )}
        />
      </span>
    </button>
  );
}

/** PPE rules, the primary analyze action and investigation reset. */
export function DetectionControls({ state, hasImage, rules, onToggleRule, onAnalyze, onReset }: DetectionControlsProps) {
  const processing = state === "processing";

  return (
    <section className="panel rounded-md">
      <header className="flex items-center justify-between border-b border-border px-4 py-2.5">
        <TechLabel>Detection controls</TechLabel>
        <TechLabel>PPE rules</TechLabel>
      </header>

      <div className="divide-y divide-border">
        <RuleToggle
          label="Helmet required"
          hint="Flag workers without a helmet"
          checked={rules.helmetRequired}
          onToggle={() => onToggleRule("helmetRequired")}
        />
        <RuleToggle
          label="Safety vest required"
          hint="Flag workers without a vest"
          checked={rules.vestRequired}
          onToggle={() => onToggleRule("vestRequired")}
        />
      </div>

      <div className="space-y-2 border-t border-border p-3">
        <button
          onClick={onAnalyze}
          disabled={!hasImage || processing}
          className="inline-flex h-10 w-full items-center justify-center gap-2 rounded-sm bg-primary font-mono text-xs uppercase tracking-[0.16em] text-primary-foreground shadow-[0_10px_26px_-12px_oklch(0.78_0.15_72/55%)] transition-all hover:bg-primary/90 active:translate-y-px disabled:pointer-events-none disabled:opacity-40"
        >
          <ScanLine className={cn("h-4 w-4", processing && "animate-pulse")} aria-hidden />
          {processing ? "Analyzing…" : "Analyze image"}
        </button>
        <button
          onClick={onReset}
          disabled={state === "idle"}
          className="inline-flex h-8 w-full items-center justify-center gap-2 font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground disabled:pointer-events-none disabled:opacity-40"
        >
          <RotateCcw className="h-3 w-3" aria-hidden />
          Reset investigation
        </button>
      </div>

      <footer className="border-t border-border px-4 py-2.5">
        <TechLabel>Model — not connected · demo pipeline</TechLabel>
      </footer>
    </section>
  );
}
