import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

interface TechLabelProps {
  children: ReactNode;
  className?: string;
  /** Renders a small live dot before the label (used for status lines). */
  dot?: boolean;
  tone?: "default" | "accent" | "live";
}

/** Monospaced technical annotation label used across the SAFEOPS UI. */
export function TechLabel({ children, className, dot = false, tone = "default" }: TechLabelProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.18em]",
        tone === "default" && "text-muted-foreground",
        tone === "accent" && "text-primary",
        tone === "live" && "text-success",
        className,
      )}
    >
      {dot && (
        <span
          className={cn(
            "h-1.5 w-1.5 rounded-full bg-current animate-pulse-dot",
            tone === "default" && "bg-muted-foreground",
          )}
        />
      )}
      {children}
    </span>
  );
}
