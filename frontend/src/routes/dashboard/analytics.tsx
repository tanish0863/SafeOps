import { useEffect, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";

import { PageHeader } from "@/components/common/PageHeader";
import { TechLabel } from "@/components/common/TechLabel";
import { PpeChart, TrendChart, ZoneChart } from "@/components/analytics/Charts";
import {
  complianceTrend as demoComplianceTrend,
  ppeBreakdown as demoPpeBreakdown,
  zoneStats as demoZoneStats,
  type ComplianceTrendPoint,
  type PpeClassBreakdown,
  type ZoneStat,
} from "@/data/demoData";
import {
  fetchComplianceTrend,
  fetchPpeBreakdown,
  fetchZoneStats,
} from "@/services/detectionService";

export const Route = createFileRoute("/dashboard/analytics")({
  head: () => ({
    meta: [
      { title: "Analytics — SafeOps Dashboard" },
      { name: "description", content: "Compliance trends, zone risk and violation analytics across monitored industrial sites." },
      { property: "og:title", content: "Analytics — SafeOps Dashboard" },
      { property: "og:description", content: "Compliance trends, zone risk and violation analytics across monitored industrial sites." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: AnalyticsPage,
});

/** Safety analytics: trend, zone and PPE breakdown charts plus zone table. */
function AnalyticsPage() {
  const [trendData, setTrendData] = useState<ComplianceTrendPoint[]>(demoComplianceTrend);
  const [zoneData, setZoneData] = useState<ZoneStat[]>(demoZoneStats);
  const [ppeData, setPpeData] = useState<PpeClassBreakdown[]>(demoPpeBreakdown);

  useEffect(() => {
    let isMounted = true;
    async function loadAnalytics() {
      try {
        const [trendRes, zoneRes, ppeRes] = await Promise.allSettled([
          fetchComplianceTrend(),
          fetchZoneStats(),
          fetchPpeBreakdown(),
        ]);
        if (!isMounted) return;
        if (trendRes.status === "fulfilled" && trendRes.value.length > 0) {
          setTrendData(trendRes.value);
        }
        if (zoneRes.status === "fulfilled" && zoneRes.value.length > 0) {
          setZoneData(zoneRes.value);
        }
        if (ppeRes.status === "fulfilled" && ppeRes.value.length > 0) {
          setPpeData(ppeRes.value);
        }
      } catch (err) {
        console.warn("Using fallback analytics data:", err);
      }
    }
    loadAnalytics();
    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Safety Intelligence"
        title="Analytics"
        description="Aggregated compliance analytics across zones and equipment classes. Connected directly to SafeOps backend /api/v1/analytics endpoints."
      />

      <div className="panel rounded-md p-5">
        <TechLabel>Compliance rate — last 12 weeks</TechLabel>
        <div className="mt-4">
          <TrendChart data={trendData} />
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="panel rounded-md p-5">
          <TechLabel>Violations by zone</TechLabel>
          <div className="mt-4">
            <ZoneChart data={zoneData} />
          </div>
        </div>
        <div className="panel rounded-md p-5">
          <TechLabel>Violations by PPE class</TechLabel>
          <div className="mt-4">
            <PpeChart data={ppeData} />
          </div>
        </div>
      </div>

      <div className="panel overflow-x-auto rounded-md">
        <table className="w-full min-w-[560px] border-collapse bg-transparent text-left">
          <thead>
            <tr className="border-b border-border bg-white/[0.03]">
              {["Zone", "Violations", "Compliance"].map((heading) => (
                <th
                  key={heading}
                  className="px-4 py-3 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground"
                >
                  {heading}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {zoneData.map((zone) => (
              <tr key={zone.zone} className="border-b border-border transition-colors last:border-0 hover:bg-secondary/40">
                <td className="px-4 py-3 text-sm text-foreground">{zone.zone}</td>
                <td className="px-4 py-3 font-mono text-xs text-foreground">{zone.violations}</td>
                <td className="px-4 py-3 font-mono text-xs text-muted-foreground">{zone.compliance}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
