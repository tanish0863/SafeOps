import { useState, useEffect } from "react";
import { Cpu, Activity, Zap, RefreshCw, CheckCircle2 } from "lucide-react";
import { toast } from "sonner";

export function EdgeInferenceCard() {
  const [latency, setLatency] = useState(38.2);
  const [fps, setFps] = useState(30.4);
  const [isPinging, setIsPinging] = useState(false);
  const [activeTab, setActiveTab] = useState<"telemetry" | "sectors">("telemetry");

  // Subtle real-time hardware telemetry micro-fluctuations
  useEffect(() => {
    const timer = setInterval(() => {
      setLatency(prev => +(38 + (Math.random() * 3.4 - 1.7)).toFixed(1));
      setFps(prev => +(30 + (Math.random() * 0.8 - 0.4)).toFixed(1));
    }, 2500);
    return () => clearInterval(timer);
  }, []);

  const handlePing = () => {
    setIsPinging(true);
    setTimeout(() => {
      setIsPinging(false);
      toast.success("AI Inference Pipeline Nominal", {
        description: `YOLO-v8x PPE Engine · TensorRT FP16 · Latency: ${latency}ms · 8 Edge Nodes in sync.`,
      });
    }, 600);
  };

  return (
    <div className="panel-raised hover-lift relative overflow-hidden rounded-xl p-5">
      {/* Top accent gradient bar */}
      <span className="absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r from-emerald-500 via-primary to-cyan-400" />

      {/* Header with Title and Mode Switcher */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
              Neural Vision Pipeline
            </span>
            <span className="inline-flex items-center gap-1 rounded bg-emerald-500/10 px-1.5 py-0.5 font-mono text-[9px] font-semibold text-emerald-400 border border-emerald-500/30">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
              CUDA ACTIVE
            </span>
          </div>

          <div className="mt-1 flex items-baseline gap-2">
            <span className="text-3xl font-bold tracking-tight text-foreground font-mono">
              {latency}
            </span>
            <span className="text-xs font-medium text-muted-foreground">
              ms latency · {fps} FPS
            </span>
          </div>
        </div>

        {/* Action Button: Pipeline Ping */}
        <button
          onClick={handlePing}
          title="Run Pipeline Diagnostic Ping"
          className="flex h-9 w-9 items-center justify-center rounded-lg border border-primary/30 bg-primary/10 text-primary hover:bg-primary/20 hover:border-primary/50 transition-all active:scale-95 shadow-sm"
        >
          <Cpu className={`h-4 w-4 ${isPinging ? "animate-spin" : ""}`} />
        </button>
      </div>

      {/* Visual Telemetry Stream Waveform */}
      <div className="mt-3.5 space-y-2">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <div className="flex items-center gap-1.5">
            <Activity className="h-3.5 w-3.5 text-primary animate-pulse" />
            <span className="font-medium text-foreground">Real-Time Inference Signal</span>
          </div>
          <span className="font-mono text-[10px] text-muted-foreground">
            YOLOv8-PPE-Pro · 99.2%
          </span>
        </div>

        {/* Dynamic Waveform SVG */}
        <div className="relative h-9 w-full overflow-hidden rounded-lg bg-secondary/30 border border-border/70 px-2 py-1 flex items-center">
          <svg className="h-full w-full" preserveAspectRatio="none" viewBox="0 0 200 40">
            <defs>
              <linearGradient id="infGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.8" />
                <stop offset="50%" stopColor="#3b82f6" stopOpacity="0.9" />
                <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.8" />
              </linearGradient>
            </defs>
            <path
              d="M0,20 Q15,6 30,20 T60,20 T90,8 T110,32 T130,20 T160,11 T180,26 T200,20"
              fill="none"
              stroke="url(#infGradient)"
              strokeWidth="2.5"
              strokeLinecap="round"
              className="animate-pulse"
            />
          </svg>
          <span className="absolute right-2.5 top-1 font-mono text-[8px] text-muted-foreground">
            EDGE-TPU · 148 VECTORS
          </span>
        </div>

        {/* Edge Infrastructure Health Matrix */}
        <div className="grid grid-cols-4 gap-1.5 pt-1 text-center">
          <div className="rounded-md border border-border/60 bg-secondary/30 py-1.5 px-1">
            <span className="block font-mono text-xs font-bold text-foreground">8 / 8</span>
            <span className="block text-[9px] text-muted-foreground truncate">Edge Nodes</span>
          </div>
          <div className="rounded-md border border-border/60 bg-secondary/30 py-1.5 px-1">
            <span className="block font-mono text-xs font-bold text-foreground">148</span>
            <span className="block text-[9px] text-muted-foreground truncate">Tracked IDs</span>
          </div>
          <div className="rounded-md border border-border/60 bg-secondary/30 py-1.5 px-1">
            <span className="block font-mono text-xs font-bold text-emerald-400">99.2%</span>
            <span className="block text-[9px] text-muted-foreground truncate">Confidence</span>
          </div>
          <div className="rounded-md border border-border/60 bg-secondary/30 py-1.5 px-1">
            <span className="block font-mono text-xs font-bold text-cyan-400">0</span>
            <span className="block text-[9px] text-muted-foreground truncate">Dropouts</span>
          </div>
        </div>
      </div>
    </div>
  );
}
