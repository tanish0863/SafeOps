import { useRouterState } from "@tanstack/react-router";
import { Bell, Menu } from "lucide-react";

import { TechLabel } from "@/components/common/TechLabel";

interface TopBarProps {
  onMenuClick: () => void;
}

/** Compact dashboard top bar: current page title, system status, notifications, profile. */
export function TopBar({ onMenuClick }: TopBarProps) {
  const matches = useRouterState({ select: (s) => s.matches });
  const title =
    matches
      .map((match) => (match.staticData as { title?: string } | undefined)?.title)
      .filter(Boolean)
      .at(-1) ?? "Overview";

  return (
    <div className="flex h-14 items-center justify-between gap-4 border-b border-glass-border bg-background/60 px-4 backdrop-blur-xl sm:px-6">
      <div className="flex min-w-0 items-center gap-3">
        <button
          onClick={onMenuClick}
          aria-label="Open menu"
          className="border border-border p-2 text-muted-foreground transition-colors hover:text-foreground lg:hidden"
        >
          <Menu className="h-4 w-4" />
        </button>
        <h1 className="truncate font-mono text-xs uppercase tracking-[0.2em] text-foreground">{title}</h1>
      </div>

      <div className="flex items-center gap-4 sm:gap-5">
        <TechLabel dot tone="live" className="hidden md:inline-flex">
          Monitoring active
        </TechLabel>

        <button
          aria-label="Notifications"
          className="relative border border-border p-2 text-muted-foreground transition-colors hover:text-foreground"
        >
          <Bell className="h-4 w-4" />
          <span className="absolute -right-0.5 -top-0.5 h-2 w-2 bg-primary" aria-hidden />
        </button>

        <div className="hidden items-center gap-3 sm:flex">
          <div className="flex h-8 w-8 items-center justify-center border border-border bg-secondary font-mono text-[10px] tracking-widest text-foreground">
            SO
          </div>
          <div className="hidden lg:block">
            <p className="text-xs leading-tight text-foreground">Site Operator</p>
            <TechLabel className="leading-tight">Shift A</TechLabel>
          </div>
        </div>
      </div>
    </div>
  );
}
