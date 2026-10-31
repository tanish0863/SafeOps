import { cn } from "@/lib/utils";

interface LogoMarkProps {
  className?: string;
  /** Compact wordmark-free mark for tight spaces (sidebar, mobile). */
  markOnly?: boolean;
}

/** SAFEOPS wordmark: technical square mark + tracked-out name. */
export function LogoMark({ className, markOnly = false }: LogoMarkProps) {
  return (
    <span className={cn("inline-flex items-center gap-3", className)}>
      <span className="relative flex h-8 w-8 items-center justify-center border border-primary/70">
        <span className="absolute inset-[5px] border border-primary/25" aria-hidden />
        <span className="h-1.5 w-1.5 bg-primary" aria-hidden />
      </span>
      {!markOnly && (
        <span className="text-sm font-semibold uppercase tracking-[0.28em] text-foreground">
          Safe<span className="text-primary">Ops</span>
        </span>
      )}
    </span>
  );
}
