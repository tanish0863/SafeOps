import { cva, type VariantProps } from "class-variance-authority";
import type { LucideIcon } from "lucide-react";
import { AlertTriangle, CheckCircle2, HelpCircle } from "lucide-react";

import type { ComplianceStatus } from "@/data/demoData";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-md border px-2.5 py-0.5 text-xs font-semibold whitespace-nowrap transition-colors",
  {
    variants: {
      status: {
        compliant: "border-success/40 bg-success/10 text-success",
        violation: "border-critical/45 bg-critical/12 text-critical",
        review: "border-warning/40 bg-warning/10 text-warning",
        critical: "border-critical/45 bg-critical/12 text-critical",
        high: "border-warning/40 bg-warning/10 text-warning",
        medium: "border-primary/40 bg-primary/10 text-primary",
        low: "border-border bg-secondary text-muted-foreground",
        confirmed: "border-critical/45 bg-critical/12 text-critical",
        pending: "border-warning/40 bg-warning/10 text-warning",
        resolved: "border-success/40 bg-success/10 text-success",
        neutral: "border-border bg-secondary text-muted-foreground",
      },
    },
    defaultVariants: {
      status: "neutral",
    },
  },
);

const badgeLabels: Record<string, string> = {
  compliant: "Compliant",
  violation: "Violation",
  review: "Needs review",
  confirmed: "Confirmed",
  pending: "Pending",
  resolved: "Resolved",
};

const badgeIcons: Record<string, LucideIcon> = {
  compliant: CheckCircle2,
  violation: AlertTriangle,
  review: HelpCircle,
};

type BadgeStatus = NonNullable<VariantProps<typeof badgeVariants>["status"]>;

interface StatusBadgeProps {
  status: BadgeStatus;
  /** Overrides the automatic label for custom vocabulary (e.g. severity names). */
  label?: string;
  icon?: boolean;
  className?: string;
}

/** Unified badge for compliance status, severity and review state. */
export function StatusBadge({ status, label, icon = false, className }: StatusBadgeProps) {
  const Icon = badgeIcons[status];
  return (
    <span className={cn(badgeVariants({ status }), className)}>
      {icon && Icon && <Icon className="h-3 w-3" />}
      {label ?? badgeLabels[status] ?? status}
    </span>
  );
}

/** Convenience mapping from a worker's PPE status to the badge vocabulary. */
export function complianceBadgeStatus(status: ComplianceStatus): BadgeStatus {
  return status;
}
