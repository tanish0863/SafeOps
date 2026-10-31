import { ShieldCheck, AlertTriangle, HelpCircle, ArrowUpRight } from "lucide-react";
import { Link } from "@tanstack/react-router";

interface PlantSafetyGaugeProps {
  complianceRate: number;
  totalWorkers: number;
  compliantCount: number;
  violationCount: number;
  reviewCount: number;
  criticalCount: number;
}

export function PlantSafetyGauge({
  complianceRate = 81.8,
  totalWorkers = 148,
  compliantCount = 121,
  violationCount = 19,
  reviewCount = 8,
  criticalCount = 3,
}: PlantSafetyGaugeProps) {
  // SVG circular progress parameters
  const size = 170;
  const strokeWidth = 12;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  // Calculate stroke offset (from 0 to 100)
  const strokeDashoffset = circumference - (complianceRate / 100) * circumference;

  return (
    <div className="panel-raised relative overflow-hidden rounded-xl p-5 sm:p-6">
      {/* Ambient decorative glow */}
      <div
        className="pointer-events-none absolute -right-16 -top-16 h-48 w-48 rounded-full bg-primary/10 blur-3xl"
        aria-hidden
      />

      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/70 pb-4">
        <div>
          <span className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-wider text-primary">
            <span className="h-2 w-2 rounded-full bg-primary animate-pulse" />
            Real-Time Plant Health Index
          </span>
          <h2 className="mt-1 text-lg font-semibold tracking-tight text-foreground">
            Operational Safety Status
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-warning/40 bg-warning/10 px-2.5 py-0.5 text-xs font-medium text-warning">
            <AlertTriangle className="h-3 w-3" />
            Elevated Hazard in Sector 3
          </span>
        </div>
      </div>

      <div className="mt-5 grid grid-cols-1 items-center gap-6 sm:grid-cols-[170px_1fr]">
        {/* Radial Circular Meter */}
        <div className="relative mx-auto flex h-[170px] w-[170px] items-center justify-center">
          <svg className="h-full w-full -rotate-90 transform" viewBox={`0 0 ${size} ${size}`}>
            {/* Background track */}
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              stroke="oklch(0.28 0.03 260)"
              strokeWidth={strokeWidth}
              fill="transparent"
            />
            {/* Target indicator ring marker (95% target) */}
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              stroke="oklch(0.5 0.05 250 / 40%)"
              strokeWidth={strokeWidth + 2}
              strokeDasharray={`${circumference * 0.01} ${circumference}`}
              strokeDashoffset={circumference - 0.95 * circumference}
              fill="transparent"
              strokeLinecap="round"
            />
            {/* Active progress arc */}
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              stroke="url(#safetyGradient)"
              strokeWidth={strokeWidth}
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              fill="transparent"
              className="transition-all duration-1000 ease-out"
            />
            <defs>
              <linearGradient id="safetyGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="var(--color-primary)" />
                <stop offset="70%" stopColor="oklch(0.85 0.16 75)" />
                <stop offset="100%" stopColor="oklch(0.7 0.18 55)" />
              </linearGradient>
            </defs>
          </svg>

          {/* Central typography inside gauge */}
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
            <span className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
              {complianceRate}%
            </span>
            <span className="mt-0.5 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
              Shift Index
            </span>
          </div>
        </div>

        {/* Detailed breakdown metrics */}
        <div className="space-y-3.5">
          <div className="grid grid-cols-3 gap-2.5">
            <div className="rounded-lg border border-border/80 bg-secondary/30 p-2.5">
              <div className="flex items-center gap-1.5 text-xs text-success">
                <ShieldCheck className="h-3.5 w-3.5" />
                <span>Compliant</span>
              </div>
              <p className="mt-1 text-xl font-bold text-foreground">{compliantCount}</p>
              <p className="text-[10px] text-muted-foreground">81.8% of site</p>
            </div>

            <div className="rounded-lg border border-critical/30 bg-critical/10 p-2.5">
              <div className="flex items-center gap-1.5 text-xs text-critical">
                <AlertTriangle className="h-3.5 w-3.5" />
                <span>Violations</span>
              </div>
              <p className="mt-1 text-xl font-bold text-critical">{violationCount}</p>
              <p className="text-[10px] text-critical/80">{criticalCount} critical</p>
            </div>

            <div className="rounded-lg border border-warning/30 bg-warning/10 p-2.5">
              <div className="flex items-center gap-1.5 text-xs text-warning">
                <HelpCircle className="h-3.5 w-3.5" />
                <span>Auditing</span>
              </div>
              <p className="mt-1 text-xl font-bold text-warning">{reviewCount}</p>
              <p className="text-[10px] text-warning/80">manual queue</p>
            </div>
          </div>

          {/* Target comparison progress bar */}
          <div className="rounded-lg border border-border/60 bg-secondary/20 p-3">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground">Shift Target: 95.0% Compliance</span>
              <span className="font-mono text-primary">+2.4% vs Shift B</span>
            </div>
            <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-secondary">
              <div
                className="h-full rounded-full bg-gradient-to-r from-primary to-amber-400 transition-all duration-700"
                style={{ width: `${complianceRate}%` }}
              />
            </div>
          </div>

          <div className="flex items-center justify-between text-xs text-muted-foreground pt-1">
            <span>{totalWorkers} workers continuously tracked by optical CV</span>
            <Link
              to="/dashboard/analytics"
              className="inline-flex items-center gap-1 text-primary hover:underline"
            >
              Full Analytics <ArrowUpRight className="h-3 w-3" />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
