import { Outlet, createFileRoute } from "@tanstack/react-router";

import { DashboardLayout } from "@/components/layouts/DashboardLayout";

export const Route = createFileRoute("/dashboard")({
  component: DashboardRoute,
});

/** Shared dashboard shell: sidebar, top bar and content outlet. */
function DashboardRoute() {
  return <DashboardLayout />;
}
