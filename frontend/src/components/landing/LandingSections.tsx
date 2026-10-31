import { Link } from "@tanstack/react-router";
import { ArrowRight } from "lucide-react";

import { buttonVariants } from "@/components/common/button";
import { DetectionBoxes } from "@/components/detection/DetectionBoxes";
import { SectionHeading } from "@/components/common/SectionHeading";
import { StatusBadge } from "@/components/common/StatusBadge";
import { TechLabel } from "@/components/common/TechLabel";
import { demoDetectionResult, demoViolations } from "@/data/demoData";
import { cn } from "@/lib/utils";
import detectionImage from "@/assets/detection-frame.jpg";

/**
 * Landing narrative sections. Each has a stable id so future GSAP
 * ScrollTrigger animations can hook in without restructuring the DOM.
 */

function ImagePanel({ caption }: { caption: string }) {
  return (
    <div className="panel-raised hover-lift corner-brackets relative rounded-md p-2">
      <img
        src={detectionImage}
        alt="Worker in PPE on the factory floor"
        width={1280}
        height={720}
        loading="lazy"
        className="h-auto w-full object-cover"
      />
      <div className="flex items-center justify-between px-1 py-2">
        <TechLabel>{caption}</TechLabel>
        <TechLabel tone="accent">Cam 04</TechLabel>
      </div>
    </div>
  );
}

export function LandingSections() {
  const workers = demoDetectionResult.workers;

  return (
    <div className="mx-auto max-w-7xl px-6">
      {/* 01 — Industrial introduction */}
      <section id="introduction" className="scroll-mt-24 border-t border-border py-24 sm:py-32">
        <div className="grid items-center gap-12 lg:grid-cols-2">
          <div>
            <SectionHeading
              index="01"
              eyebrow="Industrial Introduction"
              title="A new layer of sight for the factory floor."
              description="SafeOps is a fixed-vision monitoring layer for heavy industry. It reads the floor continuously — every worker, every helmet, every vest — and turns raw footage into a live compliance picture that safety teams can act on in seconds."
            />
            <div className="mt-8 grid grid-cols-3 gap-px overflow-hidden rounded-md border border-border bg-white/[0.07]">
              {[
                { k: "Coverage", v: "24/7" },
                { k: "Latency", v: "<2 s" },
                { k: "Humans needed", v: "0" },
              ].map((item) => (
                <div key={item.k} className="bg-card/80 p-4 transition-colors hover:bg-secondary/50">
                  <p className="text-xl font-semibold text-foreground">{item.v}</p>
                  <TechLabel className="mt-1 block">{item.k}</TechLabel>
                </div>
              ))}
            </div>
          </div>
          <ImagePanel caption="Frontline — Assembly hall, night shift" />
        </div>
      </section>

      {/* 02 — The problem */}
      <section id="problem" className="scroll-mt-24 border-t border-border py-24 sm:py-32">
        <SectionHeading
          index="02"
          eyebrow="The Workplace Safety Problem"
          title="Manual safety checks see only a fraction of the floor."
          description="PPE compliance is still verified by walking the floor. The result is delayed intervention, inconsistent enforcement, and no evidence trail when an incident does occur."
        />
        <div className="mt-12 grid gap-4 md:grid-cols-3">
          {[
            {
              stat: "Spot checks",
              title: "Audits sample minutes of a 480-minute shift",
              body: "A supervisor walking the floor observes a tiny slice of operations. Violations that happen outside that window are invisible.",
            },
            {
              stat: "Delayed action",
              title: "Risks are found after the exposure",
              body: "By the time a report is written, the unsafe condition has often existed for hours — and is likely to recur tomorrow.",
            },
            {
              stat: "No evidence",
              title: "Discussions rely on memory, not footage",
              body: "Without a visual record, safety reviews turn into disputes about what happened instead of decisions about what to fix.",
            },
          ].map((item) => (
            <div key={item.stat} className="panel hover-lift relative overflow-hidden rounded-md p-8">
              <span className="absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r from-primary via-primary/30 to-transparent" aria-hidden />
              <TechLabel tone="accent">{item.stat}</TechLabel>
              <h3 className="mt-4 text-lg font-semibold leading-snug text-foreground">{item.title}</h3>
              <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{item.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* 03 — Computer vision detection */}
      <section id="detection" className="scroll-mt-24 border-t border-border py-24 sm:py-32">
        <div className="grid items-center gap-12 lg:grid-cols-2">
          <div className="relative order-2 lg:order-1">
            <div className="panel-raised hover-lift corner-brackets relative rounded-md p-2">
              <div className="relative">
                <img
                  src={detectionImage}
                  alt="Detection overlay on factory camera frame"
                  width={1280}
                  height={720}
                  loading="lazy"
                  className="h-auto w-full object-cover"
                />
                <DetectionBoxes workers={workers} />
                <div className="absolute inset-0 bg-blueprint-fine opacity-30" aria-hidden />
              </div>
              <div className="flex items-center justify-between px-1 py-2">
                <TechLabel>Demonstration overlay — detection output</TechLabel>
                <TechLabel tone="accent">{demoDetectionResult.model}</TechLabel>
              </div>
            </div>
          </div>
          <div className="order-1 lg:order-2">
            <SectionHeading
              index="03"
              eyebrow="Computer Vision Detection"
              title="Every frame, three questions."
              description="The vision pipeline answers the same questions for every detected person: who is in frame, which PPE items are present, and whether each item belongs to that worker. Low-confidence matches are never auto-classified — they are routed to human review."
            />
            <ul className="mt-8 divide-y divide-border overflow-hidden rounded-md border border-border">
              {[
                "Individual worker detection and re-identification across frames",
                "Helmet and safety-vest association to the correct worker",
                "Missing-equipment classification with visual evidence",
              ].map((line) => (
                <li key={line} className="flex items-start gap-3 bg-card/60 p-4 text-sm text-muted-foreground transition-colors hover:bg-secondary/50">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 bg-primary" aria-hidden />
                  {line}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      {/* 04 — Worker-wise PPE analysis */}
      <section id="analysis" className="scroll-mt-24 border-t border-border py-24 sm:py-32">
        <SectionHeading
          index="04"
          eyebrow="Worker-Wise PPE Analysis"
          title="Compliance, person by person."
          description="Each detection resolves into a worker-level record: what was detected, what was missing, and whether the system is confident enough to classify it."
        />
        <div className="panel mt-12 overflow-x-auto rounded-md">
          <table className="w-full min-w-[640px] border-collapse bg-transparent text-left">
            <thead>
              <tr className="border-b border-border bg-white/[0.03]">
                {["Worker", "Name", "Helmet", "Vest", "Confidence", "Status"].map((h) => (
                  <th key={h} className="px-5 py-3 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {workers.map((worker) => (
                <tr key={worker.workerId} className="border-b border-border transition-colors last:border-0 hover:bg-secondary/40">
                  <td className="px-5 py-4 font-mono text-sm text-foreground">{worker.workerId}</td>
                  <td className="px-5 py-4 text-sm text-muted-foreground">{worker.name}</td>
                  <td className="px-5 py-4 font-mono text-xs text-muted-foreground">{worker.helmet ? "Detected" : "Missing"}</td>
                  <td className="px-5 py-4 font-mono text-xs text-muted-foreground">{worker.vest ? "Detected" : "Missing"}</td>
                  <td className="px-5 py-4 font-mono text-xs text-muted-foreground">{Math.round(worker.confidence * 100)}%</td>
                  <td className="px-5 py-4">
                    <StatusBadge status={worker.status} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <TechLabel className="mt-3 block">Demonstration records — model not connected</TechLabel>
      </section>

      {/* 05 — Safety intelligence & alerts */}
      <section id="intelligence" className="scroll-mt-24 border-t border-border py-24 sm:py-32">
        <SectionHeading
          index="05"
          eyebrow="Safety Intelligence & Alerts"
          title="Violations surface as actionable alerts."
          description="Confirmed detections raise structured alerts — severity, zone, worker and evidence — routed to the monitoring dashboard the moment they occur."
        />
        <div className="mt-12 space-y-3">
          {demoViolations.slice(0, 3).map((violation) => (
            <div
              key={violation.id}
              className="panel hover-lift flex flex-wrap items-center gap-x-6 gap-y-2 rounded-md px-5 py-4"
            >
              <span className={cn("h-8 w-1 shrink-0 rounded-full", violation.severity === "critical" ? "bg-critical" : "bg-warning")} aria-hidden />
              <div className="min-w-40">
                <p className="font-mono text-sm text-foreground">{violation.type}</p>
                <TechLabel className="mt-1 block">{violation.workerId} · {violation.zone}</TechLabel>
              </div>
              <StatusBadge status={violation.severity === "critical" ? "critical" : "high"} label={violation.severity} />
              <span className="ml-auto font-mono text-xs text-muted-foreground">{violation.timestamp}</span>
            </div>
          ))}
        </div>
      </section>

      {/* 06 — Enter the monitoring system */}
      <section id="enter" className="scroll-mt-24 border-t border-border py-24 sm:py-32">
        <div className="panel-raised corner-brackets relative rounded-lg border border-primary/35 px-6 py-16 text-center sm:px-16">
          <div className="absolute inset-0 bg-blueprint opacity-30" aria-hidden />
          <div className="relative">
            <TechLabel tone="accent" dot>
              Monitoring System — Operational
            </TechLabel>
            <h2 className="mx-auto mt-6 max-w-2xl text-3xl font-semibold tracking-tight text-foreground sm:text-5xl">
              Step onto the monitored floor.
            </h2>
            <p className="mx-auto mt-5 max-w-xl text-base text-muted-foreground">
              Open the dashboard to see live detection, violations, worker compliance and analytics —
              running on clearly-labelled demonstration data until the vision model is connected.
            </p>
            <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
              <Link to="/dashboard" className={buttonVariants({ variant: "primary", size: "lg" })}>
                Open Monitoring Dashboard <ArrowRight className="h-4 w-4" />
              </Link>
              <Link to="/dashboard/detection" className={buttonVariants({ variant: "outline", size: "lg" })}>
                Run a Detection
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
