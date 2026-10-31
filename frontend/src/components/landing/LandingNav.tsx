import { Link } from "@tanstack/react-router";

import { LogoMark } from "@/components/common/LogoMark";
import { buttonVariants } from "@/components/common/button";

const sectionLinks = [
  { id: "introduction", label: "System" },
  { id: "problem", label: "The Risk" },
  { id: "detection", label: "Vision" },
  { id: "analysis", label: "Analysis" },
  { id: "intelligence", label: "Intelligence" },
];

export function LandingNav() {
  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b border-glass-border bg-background/60 backdrop-blur-xl">
      <div
        className="pointer-events-none absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-primary/45 to-transparent"
        aria-hidden
      />
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
        <Link to="/" aria-label="SafeOps home">
          <LogoMark />
        </Link>

        <nav className="hidden items-center gap-8 md:flex" aria-label="Sections">
          {sectionLinks.map((link) => (
            <a
              key={link.id}
              href={`#${link.id}`}
              className="group relative font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground transition-colors hover:text-foreground"
            >
              {link.label}
              <span
                className="absolute -bottom-1.5 left-0 h-px w-0 bg-primary transition-all duration-300 group-hover:w-full"
                aria-hidden
              />
            </a>
          ))}
        </nav>

        <Link to="/dashboard" className={buttonVariants({ variant: "primary", size: "sm" })}>
          Open Dashboard
        </Link>
      </div>
    </header>
  );
}
