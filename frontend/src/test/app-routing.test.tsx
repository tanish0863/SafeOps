import { QueryClient } from "@tanstack/react-query";
import { createRouter, rootRouteId } from "@tanstack/react-router";
import { describe, expect, it } from "vitest";

import { routeTree } from "@/routeTree.gen";
import { dashboardNav } from "@/components/dashboard/Sidebar";

// Match routes without running loaders or rendering: loaders may need a server or
// network the test run lacks, and jsdom never loads the stylesheets React waits on.
function matchedRouteId(path: string): string | undefined {
  const router = createRouter({ routeTree, context: { queryClient: new QueryClient() } });
  return router.matchRoutes(path).at(-1)?.routeId;
}

describe("App routing", () => {
  it("matches a page for / instead of falling back to not found", () => {
    expect(matchedRouteId("/")).not.toBe(rootRouteId);
  });

  it("matches a page for every dashboard sidebar link", () => {
    for (const item of dashboardNav) {
      const routeId = matchedRouteId(item.to);
      expect(routeId, `no route matched ${item.to}`).toBeDefined();
      expect(routeId, `${item.to} fell back to not found`).not.toBe(rootRouteId);
    }
  });
});
