import { Link } from "@tanstack/react-router";
import { ChevronDown } from "lucide-react";

import { buttonVariants } from "@/components/common/button";
import { TechLabel } from "@/components/common/TechLabel";
import heroImage from "@/assets/hero-factory.jpg";

export function LandingHero() {
  return (
    <section className="relative flex min-h-screen flex-col justify-end overflow-hidden">
      {/* Factory visual — the future 3D scene will replace this layer. */}
      <img
        src={heroImage}
        alt="Night shift inside a heavy-industry factory"
        width={1920}
        height={1088}
        className="absolute inset-0 h-full w-full object-cover"
      />
      <div className="absolute inset-0 bg-hero-shade" aria-hidden />
      <div className="absolute inset-0 bg-blueprint opacity-40" aria-hidden />

      <div className="animate-rise relative mx-auto w-full max-w-7xl px-6 pb-28 pt-36 sm:pb-32">
        <TechLabel tone="accent" dot>
          SafeOps Vision System — Unit 01 · Live Site Feed
        </TechLabel>

        <h1 className="mt-6 max-w-4xl text-5xl font-semibold leading-[0.98] tracking-tight text-foreground sm:text-7xl lg:text-8xl">
          See the risk.
          <br />
          <span className="text-gradient">Before it becomes</span> an incident.
        </h1>


        <p className="mt-7 max-w-xl text-base leading-relaxed text-muted-foreground sm:text-lg">
          SafeOps watches the factory floor with industrial computer vision — detecting workers,
          associating helmets and safety vests to the right person, and flagging missing PPE the
          moment it happens, not after the audit.
        </p>

        <div className="mt-10 flex flex-wrap items-center gap-4">
          <a
            href="#introduction"
            className={`${buttonVariants({ variant: "primary", size: "lg" })} glow-primary transition-transform duration-300 hover:-translate-y-0.5`}
          >
            Explore the System
          </a>
          <Link to="/dashboard" className={buttonVariants({ variant: "glass", size: "lg" })}>
            Open Monitoring Dashboard
          </Link>
        </div>


        <div className="mt-20 flex items-center gap-4" aria-hidden>
          <span className="flex h-9 w-9 items-center justify-center border border-border text-muted-foreground">
            <ChevronDown className="h-4 w-4 animate-scroll-hint" />
          </span>
          <TechLabel>Scroll — System overview</TechLabel>
        </div>
      </div>
    </section>
  );
}
