import { useEffect, useState } from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { PpeBreakdownItem, TrendPoint, ZoneStat } from "@/data/demoData";

const tooltipStyle = {
  backgroundColor: "var(--color-popover)",
  border: "1px solid oklch(0.86 0.02 252 / 18%)",
  borderRadius: 6,
  boxShadow: "0 12px 28px -14px oklch(0 0 0 / 60%)",
  fontFamily: "var(--font-mono)",
  fontSize: 11,
} as const;

const tickStyle = {
  fill: "var(--color-muted-foreground)",
  fontSize: 10,
  fontFamily: "var(--font-mono)",
} as const;

/** Recharts renders only after hydration to avoid SSR width/height mismatches. */
function useMounted() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted;
}

export function TrendChart({ data }: { data: TrendPoint[] }) {
  const mounted = useMounted();
  if (!mounted) return <div className="h-64" aria-hidden />;

  return (
    <ResponsiveContainer width="100%" height={256}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -14 }}>
        <defs>
          <linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--color-chart-1)" stopOpacity={0.28} />
            <stop offset="100%" stopColor="var(--color-chart-1)" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="label" tick={tickStyle} axisLine={{ stroke: "var(--color-border)" }} tickLine={false} />
        <YAxis domain={[75, 100]} tick={tickStyle} axisLine={false} tickLine={false} />
        <Tooltip
          contentStyle={tooltipStyle}
          labelStyle={{ color: "var(--color-foreground)" }}
          itemStyle={{ color: "var(--color-chart-1)" }}
          formatter={(value) => `${value}% compliance`}
        />
        <Area
          type="monotone"
          dataKey="rate"
          stroke="none"
          fill="url(#trendFill)"
          activeDot={false}
          legendType="none"
        />
        <Line
          type="monotone"
          dataKey="rate"
          stroke="var(--color-chart-1)"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 3, fill: "var(--color-chart-1)" }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function ZoneChart({ data }: { data: ZoneStat[] }) {
  const mounted = useMounted();
  if (!mounted) return <div className="h-64" aria-hidden />;

  return (
    <ResponsiveContainer width="100%" height={256}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -22 }}>
        <defs>
          <linearGradient id="zoneFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--color-chart-5)" stopOpacity={0.95} />
            <stop offset="100%" stopColor="var(--color-chart-5)" stopOpacity={0.45} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="zone" tick={tickStyle} axisLine={{ stroke: "var(--color-border)" }} tickLine={false} />
        <YAxis tick={tickStyle} axisLine={false} tickLine={false} allowDecimals={false} />
        <Tooltip
          contentStyle={tooltipStyle}
          labelStyle={{ color: "var(--color-foreground)" }}
          itemStyle={{ color: "var(--color-chart-5)" }}
          cursor={{ fill: "var(--color-secondary)", opacity: 0.4 }}
        />
        <Bar dataKey="violations" fill="url(#zoneFill)" radius={[3, 3, 0, 0]} maxBarSize={36} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function PpeChart({ data }: { data: PpeBreakdownItem[] }) {
  const mounted = useMounted();
  if (!mounted) return <div className="h-64" aria-hidden />;

  return (
    <ResponsiveContainer width="100%" height={256}>
      <BarChart data={data} layout="vertical" margin={{ top: 8, right: 16, bottom: 0, left: 24 }}>
        <defs>
          <linearGradient id="ppeFill" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="var(--color-chart-4)" stopOpacity={0.5} />
            <stop offset="100%" stopColor="var(--color-chart-4)" stopOpacity={0.95} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" tick={tickStyle} axisLine={{ stroke: "var(--color-border)" }} tickLine={false} allowDecimals={false} />
        <YAxis type="category" dataKey="ppe" tick={tickStyle} axisLine={false} tickLine={false} width={80} />
        <Tooltip
          contentStyle={tooltipStyle}
          labelStyle={{ color: "var(--color-foreground)" }}
          itemStyle={{ color: "var(--color-chart-4)" }}
          cursor={{ fill: "var(--color-secondary)", opacity: 0.4 }}
        />
        <Bar dataKey="violations" fill="url(#ppeFill)" radius={[0, 3, 3, 0]} maxBarSize={18} />
      </BarChart>
    </ResponsiveContainer>
  );
}
