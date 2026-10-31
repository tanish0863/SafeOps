import { Link } from "@tanstack/react-router";

import { LogoMark } from "@/components/common/LogoMark";
import { TechLabel } from "@/components/common/TechLabel";

export function LandingFooter() {
  return (
    <footer className="glass-card border-t border-glass-border">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-6 px-6 py-10">
        <LogoMark />
        <div className="flex items-center gap-8">
          <a href="#introduction" className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground transition-colors hover:text-foreground">
            System
          </a>
          <Link to="/dashboard" className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground transition-colors hover:text-foreground">
            Dashboard
          </Link>
          <Link to="/dashboard/settings" className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground transition-colors hover:text-foreground">
            Settings
          </Link>
        </div>
        <TechLabel>© 2026 SafeOps — Built for safer industrial sites</TechLabel>
      </div>
    </footer>
  );
}
