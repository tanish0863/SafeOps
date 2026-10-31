import type { LucideIcon } from "lucide-react";

import { TechLabel } from "@/components/common/TechLabel";
import { cn } from "@/lib/utils";

interface StatCardProps {
  label: string;
  value: string;
  delta?: string;
  deltaTone?: "positive" | "negative" | "neutral";
  icon?: LucideIcon;
  className?: string;
}

/** KPI tile used on the dashboard overview: strong number hierarchy, icon chip, accent rule. */
export function StatCard({ label, value, delta, deltaTone = "neutral", icon: Icon, className }: StatCardProps) {
  return (
    <div className={cn("panel-raised hover-lift relative overflow-hidden rounded-md p-5", className)}>
      <span
        className="absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r from-primary via-primary/35 to-transparent"
        aria-hidden
      />
      <div className="flex items-start justify-between gap-3">
        <TechLabel>{label}</TechLabel>
        {Icon && (
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-sm border border-border bg-secondary/50 text-primary">
            <Icon className="h-3.5 w-3.5" aria-hidden />
          </span>
        )}
      </div>
      <p className="mt-4 text-[2.6rem] font-semibold leading-none tracking-tight text-foreground">{value}</p>
      {delta && (
        <p
          className={cn(
            "mt-3 font-mono text-[11px] uppercase tracking-[0.14em]",
            deltaTone === "positive" && "text-success",
            deltaTone === "negative" && "text-critical",
            deltaTone === "neutral" && "text-muted-foreground",
          )}
        >
          {delta}
        </p>
      )}
    </div>
  );
}
