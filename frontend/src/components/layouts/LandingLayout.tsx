import type { ReactNode } from "react";

import { LandingFooter } from "@/components/landing/LandingFooter";
import { LandingNav } from "@/components/landing/LandingNav";

/** Shared shell for the cinematic landing experience. */
export function LandingLayout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-background">
      <LandingNav />
      <main>{children}</main>
      <LandingFooter />
    </div>
  );
}
