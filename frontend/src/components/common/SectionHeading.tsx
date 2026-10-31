import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

interface SectionHeadingProps {
  /** Technical index shown before the eyebrow, e.g. "01". */
  index?: string;
  eyebrow: string;
  title: ReactNode;
  description?: ReactNode;
  align?: "left" | "center";
  className?: string;
}

/** Consistent editorial heading block for landing sections and page headers. */
export function SectionHeading({
  index,
  eyebrow,
  title,
  description,
  align = "left",
  className,
}: SectionHeadingProps) {
  return (
    <div className={cn("max-w-2xl", align === "center" && "mx-auto text-center", className)}>
      <p className="flex items-center gap-3 font-mono text-[11px] uppercase tracking-[0.22em] text-primary">
        {index && <span className="text-muted-foreground/70">{index}</span>}
        <span className="h-px w-6 bg-primary/50" aria-hidden />
        {eyebrow}
      </p>
      <h2 className="mt-4 text-3xl font-semibold leading-[1.1] tracking-tight text-foreground sm:text-4xl">
        {title}
      </h2>
      {description && (
        <p className={cn("mt-4 text-base leading-relaxed text-muted-foreground", align === "center" && "mx-auto")}>
          {description}
        </p>
      )}
    </div>
  );
}
