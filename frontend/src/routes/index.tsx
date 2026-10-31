import { createFileRoute } from "@tanstack/react-router";

import { LandingLayout } from "@/components/layouts/LandingLayout";
import { LandingHero } from "@/components/landing/LandingHero";
import { LandingSections } from "@/components/landing/LandingSections";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    meta: [
      { title: "SafeOps — Industrial PPE Compliance Monitoring" },
      {
        name: "description",
        content:
          "SafeOps watches the factory floor with industrial computer vision — detecting workers, associating helmets and safety vests, and flagging missing PPE the moment it happens.",
      },
      { property: "og:title", content: "SafeOps — See the risk. Before it becomes an incident." },
      {
        property: "og:description",
        content:
          "Industrial computer vision for workplace safety: worker detection, PPE association and live compliance monitoring.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
});

/** Cinematic landing experience: hero + system narrative sections. */
function Index() {
  return (
    <LandingLayout>
      <LandingHero />
      <LandingSections />
    </LandingLayout>
  );
}
