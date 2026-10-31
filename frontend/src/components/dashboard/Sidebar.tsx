import { Link, useRouterState } from "@tanstack/react-router";
import {
  ArrowLeft,
  ChartColumn,
  LayoutDashboard,
  ScanLine,
  Settings,
  TriangleAlert,
  Users,
  X,
} from "lucide-react";

import { LogoMark } from "@/components/common/LogoMark";
import { TechLabel } from "@/components/common/TechLabel";
import { cn } from "@/lib/utils";

export const dashboardNav = [
  { to: "/dashboard", label: "Overview", icon: LayoutDashboard, exact: true },
  { to: "/dashboard/detection", label: "Detection", icon: ScanLine, exact: false },
  { to: "/dashboard/violations", label: "Violations", icon: TriangleAlert, exact: false },
  { to: "/dashboard/workers", label: "Workers", icon: Users, exact: false },
  { to: "/dashboard/analytics", label: "Analytics", icon: ChartColumn, exact: false },
  { to: "/dashboard/settings", label: "Settings", icon: Settings, exact: false },
] as const;

export function SidebarNav({ onNavigate }: { onNavigate?: (() => void) | undefined }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  return (
    <nav className="flex flex-1 flex-col gap-1 px-3 py-6" aria-label="Dashboard sections">
      {dashboardNav.map((item) => {
        const active = item.exact ? pathname === item.to : pathname.startsWith(item.to);
        const Icon = item.icon;
        return (
          <Link
            key={item.to}
            to={item.to}
            onClick={onNavigate}
            className={cn(
              "group relative flex items-center gap-3 rounded-md border px-3.5 py-2.5 font-mono text-[11px] uppercase tracking-[0.16em] transition-all duration-200",
              active
                ? "border-glass-border bg-secondary/70 text-foreground shadow-[inset_0_1px_0_oklch(1_0_0/8%)]"
                : "border-transparent text-muted-foreground hover:border-glass-border hover:bg-secondary/40 hover:text-foreground",
            )}
          >
            <span
              className={cn(
                "absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-r-full bg-primary transition-all duration-200",
                active ? "opacity-100" : "opacity-0",
              )}
              aria-hidden
            />
            <Icon
              className={cn("h-4 w-4 transition-colors", active ? "text-primary" : "text-muted-foreground group-hover:text-foreground")}
              aria-hidden
            />
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}

interface SidebarProps {
  onNavigate?: () => void;
  /** Rendered only inside the mobile drawer. */
  onClose?: () => void;
}

/** Persistent dashboard sidebar: wordmark, section navigation, system status. */
export function Sidebar({ onNavigate, onClose }: SidebarProps) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-border px-5 py-4">
        <Link to="/dashboard" aria-label="SafeOps dashboard">
          <LogoMark />
        </Link>
        {onClose && (
          <button onClick={onClose} aria-label="Close menu" className="text-muted-foreground hover:text-foreground">
            <X className="h-5 w-5" />
          </button>
        )}
      </div>

      <SidebarNav onNavigate={onNavigate} />

      <div className="border-t border-border px-5 py-4">
        <div className="panel-sunken rounded-md px-3 py-3">
          <TechLabel dot tone="live">
            System online
          </TechLabel>
          <TechLabel className="mt-2 block">v0.1.0 — demo data</TechLabel>
        </div>
        <Link
          to="/"
          className="mt-4 flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.16em] text-muted-foreground transition-colors hover:text-foreground"
        >
          <ArrowLeft className="h-3.5 w-3.5" aria-hidden />
          Back to site
        </Link>
      </div>
    </div>
  );
}
