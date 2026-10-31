import { useState } from "react";
import { Camera, AlertCircle, CheckCircle, Flame, Zap, Truck, Cog, Box } from "lucide-react";
import { cn } from "@/lib/utils";

interface ZoneData {
  id: string;
  name: string;
  type: string;
  compliance: number;
  violations: number;
  workers: number;
  camera: string;
  status: "optimal" | "warning" | "critical";
  hazardDetail: string;
  primaryBreach: string;
}

const zones: ZoneData[] = [
  {
    id: "zone-foundry",
    name: "Foundry Floor",
    type: "Heavy Metallurgy",
    compliance: 82.8,
    violations: 11,
    workers: 34,
    camera: "CAM-04",
    status: "critical",
    hazardDetail: "Molten ladle transport & radiant heat hazard",
    primaryBreach: "Missing Helmets (4 flagged)",
  },
  {
    id: "zone-welding",
    name: "Welding Bay",
    type: "High-Temp Fabrication",
    compliance: 86.2,
    violations: 9,
    workers: 28,
    camera: "CAM-02",
    status: "warning",
    hazardDetail: "Arc flash hazard & spark dispersion zone",
    primaryBreach: "Missing Safety Vests (5 flagged)",
  },
  {
    id: "zone-dock",
    name: "Loading Dock",
    type: "Logistics Corridor",
    compliance: 88.4,
    violations: 8,
    workers: 28,
    camera: "CAM-05",
    status: "warning",
    hazardDetail: "Automated forklift transit & blind corners",
    primaryBreach: "Missing Helmets (2 flagged)",
  },
  {
    id: "zone-assembly-a",
    name: "Assembly Line A",
    type: "Robotic Assembly",
    compliance: 93.5,
    violations: 6,
    workers: 32,
    camera: "CAM-01",
    status: "optimal",
    hazardDetail: "Conveyor pinch-points & pneumatic tools",
    primaryBreach: "Unverified Gloves (1 flagged)",
  },
  {
    id: "zone-assembly-b",
    name: "Assembly Line B",
    type: "Final Inspection & Pack",
    compliance: 95.1,
    violations: 4,
    workers: 26,
    camera: "CAM-03",
    status: "optimal",
    hazardDetail: "Packaging cell & automated strapping",
    primaryBreach: "Minor posture occlusion",
  },
];

const zoneIcons: Record<string, typeof Flame> = {
  "zone-foundry": Flame,
  "zone-welding": Zap,
  "zone-dock": Truck,
  "zone-assembly-a": Cog,
  "zone-assembly-b": Box,
};

export function ZoneSafetyMatrix({ onSelectZone }: { onSelectZone?: (zoneName: string) => void }) {
  const [activeZone, setActiveZone] = useState<string>("zone-foundry");

  const selected = zones.find((z) => z.id === activeZone) || zones[0];
  const Icon = zoneIcons[selected.id] || Flame;

  return (
    <div className="panel overflow-hidden rounded-xl">
      {/* Header with status badges */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/80 px-5 py-3.5">
        <div className="flex items-center gap-2.5">
          <span className="flex h-2.5 w-2.5 rounded-full bg-primary animate-pulse" />
          <h3 className="text-sm font-semibold tracking-wide text-foreground">
            Plant Spatial Safety Matrix
          </h3>
          <span className="rounded-full border border-border bg-secondary/40 px-2 py-0.5 text-[11px] text-muted-foreground">
            5 Monitored Sectors
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-critical" /> Critical (&lt;85%)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-warning" /> Warning (&lt;90%)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-success" /> Optimal (≥90%)
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 divide-y lg:divide-y-0 lg:divide-x divide-border/70">
        {zones.map((zone) => {
          const isSelected = activeZone === zone.id;
          const ZoneIcon = zoneIcons[zone.id] || Box;

          return (
            <button
              key={zone.id}
              onClick={() => {
                setActiveZone(zone.id);
                onSelectZone?.(zone.name);
              }}
              className={cn(
                "group relative flex flex-col justify-between p-4 text-left transition-all duration-200",
                isSelected
                  ? "bg-secondary/40 shadow-inner"
                  : "hover:bg-secondary/20",
              )}
            >
              {/* Highlight top bar when selected */}
              {isSelected && (
                <span className="absolute inset-x-0 top-0 h-1 bg-primary" aria-hidden />
              )}

              <div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <ZoneIcon className={cn(
                      "h-3.5 w-3.5",
                      zone.status === "critical" && "text-critical",
                      zone.status === "warning" && "text-warning",
                      zone.status === "optimal" && "text-success",
                    )} />
                    <span className="font-mono text-[11px] text-muted-foreground">
                      {zone.camera}
                    </span>
                  </div>
                  <span
                    className={cn(
                      "h-2 w-2 rounded-full",
                      zone.status === "critical" && "bg-critical animate-ping",
                      zone.status === "warning" && "bg-warning",
                      zone.status === "optimal" && "bg-success",
                    )}
                  />
                </div>

                <h4 className="mt-2.5 text-sm font-semibold text-foreground group-hover:text-primary transition-colors">
                  {zone.name}
                </h4>
                <p className="text-[11px] text-muted-foreground">{zone.type}</p>
              </div>

              <div className="mt-4">
                <div className="flex items-baseline justify-between">
                  <span className="text-xl font-bold tracking-tight text-foreground">
                    {zone.compliance}%
                  </span>
                  <span
                    className={cn(
                      "text-xs font-medium",
                      zone.violations > 8
                        ? "text-critical"
                        : zone.violations > 5
                        ? "text-warning"
                        : "text-muted-foreground",
                    )}
                  >
                    {zone.violations} alerts
                  </span>
                </div>

                {/* Progress bar */}
                <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-secondary">
                  <div
                    className={cn(
                      "h-full rounded-full transition-all duration-500",
                      zone.status === "critical" && "bg-critical",
                      zone.status === "warning" && "bg-warning",
                      zone.status === "optimal" && "bg-success",
                    )}
                    style={{ width: `${zone.compliance}%` }}
                  />
                </div>

                <div className="mt-2 flex items-center justify-between text-[10px] text-muted-foreground">
                  <span>{zone.workers} workers</span>
                  <span className="font-mono">LIVE</span>
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Selected Sector Context Drawer */}
      <div className="border-t border-border/80 bg-secondary/15 px-5 py-3.5 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-3">
          <span className="inline-flex items-center gap-1.5 rounded-md border border-border bg-secondary/50 px-2.5 py-1 font-mono text-[11px] text-foreground">
            <Camera className="h-3 w-3 text-primary" />
            {selected.camera} FEED ACTIVE
          </span>
          <span className="text-muted-foreground">
            <strong className="text-foreground">{selected.name}:</strong> {selected.hazardDetail}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="inline-flex items-center gap-1.5 text-critical font-medium">
            <AlertCircle className="h-3.5 w-3.5" />
            Active Warning: {selected.primaryBreach}
          </span>
        </div>
      </div>
    </div>
  );
}
