import { useState, useEffect } from "react";
import { Link } from "@tanstack/react-router";
import { Eye, ShieldAlert, Sparkles, Maximize2, Radio, Video, SlidersHorizontal } from "lucide-react";
import sampleFrame from "@/assets/detection-frame.jpg";
import { buttonVariants } from "@/components/common/button";

interface LiveVisionStreamProps {
  onInspect?: () => void;
}

export function LiveVisionStream({ onInspect: _onInspect }: LiveVisionStreamProps) {
  const [showOverlays, setShowOverlays] = useState(true);
  const [activeCam, setActiveCam] = useState<"CAM-04" | "CAM-02" | "CAM-01">("CAM-04");
  const [clock, setClock] = useState("06:51:24");

  // Simulated live CCTV clock
  useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      setClock(now.toTimeString().split(" ")[0]);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="panel overflow-hidden rounded-xl">
      {/* Stream Controls Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/80 px-5 py-3">
        <div className="flex items-center gap-3">
          <span className="flex h-2.5 w-2.5 rounded-full bg-critical animate-pulse" />
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs font-semibold uppercase tracking-wider text-foreground">
              {activeCam} · Optical Vision Feed
            </span>
            <span className="rounded-full border border-critical/40 bg-critical/10 px-2 py-0.5 font-mono text-[10px] text-critical">
              LIVE REC
            </span>
          </div>
        </div>

        {/* Camera Selector Tabs */}
        <div className="flex items-center gap-1.5 rounded-lg border border-border/70 bg-secondary/30 p-1">
          <button
            onClick={() => setActiveCam("CAM-04")}
            className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
              activeCam === "CAM-04"
                ? "bg-primary text-primary-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            CAM-04 (Foundry)
          </button>
          <button
            onClick={() => setActiveCam("CAM-02")}
            className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
              activeCam === "CAM-02"
                ? "bg-primary text-primary-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            CAM-02 (Welding)
          </button>
          <button
            onClick={() => setActiveCam("CAM-01")}
            className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
              activeCam === "CAM-01"
                ? "bg-primary text-primary-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            CAM-01 (Assembly)
          </button>
        </div>

        {/* Quick Toggles */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowOverlays(!showOverlays)}
            className={`flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs transition-colors ${
              showOverlays
                ? "border-primary/40 bg-primary/10 text-primary"
                : "border-border bg-secondary/30 text-muted-foreground"
            }`}
          >
            <Eye className="h-3.5 w-3.5" />
            <span>AI Overlays</span>
          </button>

          <Link
            to="/dashboard/detection"
            className={`${buttonVariants({ variant: "ghost", size: "sm" })} h-7 px-2 text-xs`}
          >
            <Maximize2 className="h-3.5 w-3.5 mr-1" />
            Inspect
          </Link>
        </div>
      </div>

      {/* CCTV Viewfinder Container */}
      <div className="relative aspect-[16/9] w-full overflow-hidden bg-black/90">
        <img
          src={sampleFrame}
          alt="Live Industrial Camera Feed"
          className="h-full w-full object-cover object-center filter brightness-95 contrast-105"
        />

        {/* Scan line effect */}
        <div className="pointer-events-none absolute inset-0 bg-blueprint-fine opacity-20" />
        <div className="pointer-events-none absolute inset-0 animate-safeops-scan bg-gradient-to-b from-transparent via-primary/10 to-transparent h-24" />

        {/* CCTV HUD Telemetry Watermarks */}
        <div className="pointer-events-none absolute left-4 top-4 flex items-center gap-2 rounded bg-black/60 px-2.5 py-1 font-mono text-[11px] text-white/90 backdrop-blur-md">
          <Radio className="h-3 w-3 text-critical animate-pulse" />
          <span>{activeCam} FOUNDRY BLAST CORRIDOR</span>
          <span className="text-white/40">|</span>
          <span className="text-amber-300">{clock} UTC</span>
        </div>

        <div className="pointer-events-none absolute right-4 top-4 rounded bg-black/60 px-2 py-1 font-mono text-[10px] text-white/70 backdrop-blur-md">
          1080P · 30.0 FPS · CV MODEL: v2.4-FAST
        </div>

        {/* Real-time Bounding Box Overlays */}
        {showOverlays && (
          <>
            {/* Worker 1: Arun Singh - Compliant */}
            <div
              className="absolute border-2 border-success bg-success/15 transition-all duration-300 rounded-sm"
              style={{ left: "29.5%", top: "12%", width: "27%", height: "82%" }}
            >
              <div className="absolute -top-7 left-0 flex items-center gap-1 rounded bg-black/80 px-1.5 py-0.5 font-mono text-[10px] font-medium text-success backdrop-blur-sm border border-success/40">
                <span>W-074 · SAFE [97%]</span>
              </div>
              <div
                className="absolute border border-success/80 bg-success/30 rounded-xs"
                style={{ left: "18%", top: "1%", width: "48%", height: "17%" }}
              >
                <span className="absolute -bottom-4 right-0 font-mono text-[8px] text-success bg-black/70 px-1">
                  HELMET OK
                </span>
              </div>
              <div
                className="absolute border border-success/80 bg-success/30 rounded-xs"
                style={{ left: "11%", top: "28%", width: "78%", height: "49%" }}
              >
                <span className="absolute -bottom-4 right-0 font-mono text-[8px] text-success bg-black/70 px-1">
                  VEST OK
                </span>
              </div>
            </div>

            {/* Worker 2: Unverified - Critical Violation */}
            <div
              className="absolute border-2 border-critical bg-critical/20 transition-all duration-300 rounded-sm animate-pulse"
              style={{ left: "61.5%", top: "20%", width: "17%", height: "66%" }}
            >
              <div className="absolute -top-7 left-0 flex items-center gap-1 rounded bg-critical px-1.5 py-0.5 font-mono text-[10px] font-bold text-white shadow-md">
                <ShieldAlert className="h-3 w-3" />
                <span>NO HELMET · 91%</span>
              </div>
              {/* Missing helmet dashed highlight */}
              <div
                className="absolute border border-dashed border-critical bg-critical/40 rounded-xs animate-ping"
                style={{ left: "22%", top: "0%", width: "55%", height: "16%" }}
              />
            </div>

            {/* Worker 3: Joseph Mathew - Low Confidence Review */}
            <div
              className="absolute border-2 border-warning bg-warning/15 transition-all duration-300 rounded-sm"
              style={{ left: "5.5%", top: "24%", width: "16%", height: "62%" }}
            >
              <div className="absolute -top-7 left-0 flex items-center gap-1 rounded bg-warning/90 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-black backdrop-blur-sm">
                <span>W-089 · AUDIT [58%]</span>
              </div>
            </div>
          </>
        )}

        {/* Viewfinder crosshairs and corner brackets */}
        <div className="pointer-events-none absolute bottom-4 left-4 flex items-center gap-2">
          <span className="rounded bg-black/60 px-2 py-1 font-mono text-[10px] text-white/80 backdrop-blur-sm">
            3 WORKERS TRACKED · 1 CRITICAL BREACH
          </span>
        </div>

        <div className="absolute bottom-4 right-4">
          <Link
            to="/dashboard/detection"
            className="flex items-center gap-1.5 rounded-lg bg-primary/95 px-3 py-1.5 font-medium text-xs text-primary-foreground shadow-lg backdrop-blur-sm hover:bg-primary transition-all duration-200"
          >
            <Sparkles className="h-3.5 w-3.5" />
            Inspect Full Geometry
          </Link>
        </div>
      </div>

      {/* Stream Footer Telemetry */}
      <div className="border-t border-border/80 bg-secondary/20 px-5 py-2.5 flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
        <div className="flex items-center gap-2 font-mono text-[11px]">
          <span className="h-2 w-2 rounded-full bg-success" />
          <span>Inference Latency: 42ms</span>
          <span className="text-border">|</span>
          <span>NVIDIA TensorRT Acceleration</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-foreground font-medium">Automatic Snapshot Trigger:</span>
          <span className="rounded bg-secondary px-1.5 py-0.5 font-mono text-[10px] text-foreground">
            On Severity &ge; High
          </span>
        </div>
      </div>
    </div>
  );
}
