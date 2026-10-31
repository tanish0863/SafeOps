import { useState } from "react";
import { UserPlus, X, ShieldCheck, HardHat, CheckCircle2, AlertTriangle, AlertOctagon } from "lucide-react";
import type { Worker, ComplianceStatus } from "@/data/demoData";
import { addRealtimeWorker } from "@/lib/workerStore";
import { toast } from "sonner";

interface AddWorkerModalProps {
  isOpen: boolean;
  onClose: () => void;
  onWorkerAdded: (worker: Worker) => void;
}

const defaultZones = [
  "Assembly Line A",
  "Assembly Line B",
  "Foundry Floor",
  "Welding Bay",
  "Loading Dock",
  "Quality Control",
];

export function AddWorkerModal({ isOpen, onClose, onWorkerAdded }: AddWorkerModalProps) {
  const [name, setName] = useState("");
  const [workerId, setWorkerId] = useState(() => `W-${Math.floor(100 + Math.random() * 900)}`);
  const [zone, setZone] = useState("Assembly Line A");
  const [customZone, setCustomZone] = useState("");
  const [shift, setShift] = useState<"A" | "B" | "C">("A");
  const [helmet, setHelmet] = useState(true);
  const [vest, setVest] = useState(true);
  const [overrideStatus, setOverrideStatus] = useState<ComplianceStatus | "auto">("auto");

  if (!isOpen) return null;

  // Compute status: if auto, compliant if both helmet and vest present, else violation
  const effectiveStatus: ComplianceStatus =
    overrideStatus === "auto"
      ? helmet && vest
        ? "compliant"
        : "violation"
      : overrideStatus;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedName = name.trim();
    if (!trimmedName) {
      toast.error("Please enter a worker name");
      return;
    }

    const assignedZone = zone === "custom" ? customZone.trim() || "General Floor" : zone;
    const now = new Date();
    const timestamp = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")} ${now.toTimeString().split(" ")[0].slice(0, 5)}`;

    const newWorker: Worker = {
      id: workerId.trim() || `W-${Math.floor(100 + Math.random() * 900)}`,
      name: trimmedName,
      zone: assignedZone,
      shift,
      lastScan: timestamp,
      scans: 1,
      helmet,
      vest,
      status: effectiveStatus,
      needsReview: effectiveStatus === "review",
    };

    addRealtimeWorker(newWorker);
    onWorkerAdded(newWorker);
    toast.success("Worker added to real-time registry", {
      description: `${newWorker.name} (${newWorker.id}) registered in ${newWorker.zone}.`,
    });
    setName("");
    setWorkerId(`W-${Math.floor(100 + Math.random() * 900)}`);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="panel w-full max-w-lg rounded-xl overflow-hidden shadow-2xl border border-border">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-border/80 px-6 py-4 bg-secondary/30">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-primary/40 bg-primary/10 text-primary">
              <UserPlus className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-foreground">Register Personnel Data</h2>
              <p className="text-[11px] text-muted-foreground">Enter real-time worker details into the live registry</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-border text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Form Body */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Full Name */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-foreground">Worker Full Name *</label>
              <input
                type="text"
                required
                placeholder="e.g. Aarav Sharma"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full rounded-lg border border-border bg-secondary/30 px-3 py-2 text-xs text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none"
              />
            </div>

            {/* Worker ID */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-foreground">Worker ID</label>
              <input
                type="text"
                placeholder="W-101"
                value={workerId}
                onChange={(e) => setWorkerId(e.target.value)}
                className="w-full rounded-lg border border-border bg-secondary/30 px-3 py-2 text-xs font-mono text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Zone Selection */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-foreground">Facility Zone</label>
              <select
                value={zone}
                onChange={(e) => setZone(e.target.value)}
                className="w-full rounded-lg border border-border bg-secondary/30 px-3 py-2 text-xs text-foreground focus:border-primary focus:outline-none"
              >
                {defaultZones.map((z) => (
                  <option key={z} value={z} className="bg-background text-foreground">
                    {z}
                  </option>
                ))}
                <option value="custom" className="bg-background text-foreground">
                  Custom Sector…
                </option>
              </select>

              {zone === "custom" && (
                <input
                  type="text"
                  placeholder="Enter custom zone name"
                  value={customZone}
                  onChange={(e) => setCustomZone(e.target.value)}
                  className="mt-1.5 w-full rounded-lg border border-border bg-secondary/30 px-3 py-1.5 text-xs text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none"
                />
              )}
            </div>

            {/* Shift Selection */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-foreground">Assigned Shift</label>
              <div className="flex rounded-lg border border-border bg-secondary/30 p-1">
                {(["A", "B", "C"] as const).map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => setShift(s)}
                    className={`flex-1 rounded-md py-1 text-xs font-medium transition-colors ${
                      shift === s
                        ? "bg-primary text-primary-foreground shadow-sm font-semibold"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    Shift {s}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* PPE Compliance Checkboxes */}
          <div className="space-y-2 pt-1 border-t border-border/60">
            <label className="text-xs font-semibold text-foreground">PPE Verification State</label>
            <div className="grid grid-cols-2 gap-3">
              {/* Helmet */}
              <label
                onClick={() => setHelmet(!helmet)}
                className={`flex items-center gap-2.5 p-3 rounded-lg border cursor-pointer transition-all ${
                  helmet
                    ? "border-success/40 bg-success/10 text-success"
                    : "border-critical/40 bg-critical/10 text-critical"
                }`}
              >
                <HardHat className="h-4 w-4 shrink-0" />
                <div className="min-w-0">
                  <span className="block text-xs font-semibold text-foreground">Safety Helmet</span>
                  <span className="block text-[10px]">
                    {helmet ? "Detected / Present" : "Missing / Non-compliant"}
                  </span>
                </div>
              </label>

              {/* Safety Vest */}
              <label
                onClick={() => setVest(!vest)}
                className={`flex items-center gap-2.5 p-3 rounded-lg border cursor-pointer transition-all ${
                  vest
                    ? "border-success/40 bg-success/10 text-success"
                    : "border-critical/40 bg-critical/10 text-critical"
                }`}
              >
                <ShieldCheck className="h-4 w-4 shrink-0" />
                <div className="min-w-0">
                  <span className="block text-xs font-semibold text-foreground">Hi-Vis Vest</span>
                  <span className="block text-[10px]">
                    {vest ? "Detected / Present" : "Missing / Non-compliant"}
                  </span>
                </div>
              </label>
            </div>
          </div>

          {/* Status Override */}
          <div className="space-y-1.5 pt-1">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold text-foreground">Calculated Status</label>
              <span
                className={`inline-flex items-center gap-1 rounded px-2 py-0.5 font-mono text-[10px] font-semibold uppercase ${
                  effectiveStatus === "compliant"
                    ? "border border-success/40 bg-success/10 text-success"
                    : effectiveStatus === "violation"
                    ? "border border-critical/40 bg-critical/10 text-critical"
                    : "border border-warning/40 bg-warning/10 text-warning"
                }`}
              >
                {effectiveStatus === "compliant" && <CheckCircle2 className="h-3 w-3" />}
                {effectiveStatus === "violation" && <AlertOctagon className="h-3 w-3" />}
                {effectiveStatus === "review" && <AlertTriangle className="h-3 w-3" />}
                {effectiveStatus}
              </span>
            </div>
          </div>

          {/* Form Actions */}
          <div className="flex items-center justify-end gap-3 pt-3 border-t border-border/80">
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-border px-4 py-2 text-xs font-medium text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="rounded-lg bg-primary px-5 py-2 text-xs font-semibold text-primary-foreground shadow-md hover:bg-primary/90 transition-all active:scale-[0.99]"
            >
              Save Worker
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
