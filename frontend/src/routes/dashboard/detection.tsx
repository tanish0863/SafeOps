import { useEffect } from "react";
import { createFileRoute } from "@tanstack/react-router";

import { PageHeader } from "@/components/common/PageHeader";
import { DetectionCanvas } from "@/components/detection/DetectionCanvas";
import { DetectionControls } from "@/components/detection/DetectionControls";
import { DetectionSummary } from "@/components/detection/DetectionSummary";
import { ImageUploader } from "@/components/detection/ImageUploader";
import { InvestigationDetails } from "@/components/detection/InvestigationDetails";
import { WorkerInspectionPanel } from "@/components/detection/WorkerInspectionPanel";
import { useDetectionFlow } from "@/hooks/useDetectionFlow";
import sampleFrame from "@/assets/detection-frame.jpg";

export const Route = createFileRoute("/dashboard/detection")({
  head: () => ({
    meta: [
      { title: "Detection Workspace — SafeOps" },
      { name: "description", content: "Analyze workplace images and inspect worker-level PPE compliance in the investigation canvas." },
      { property: "og:title", content: "Detection Workspace — SafeOps" },
      { property: "og:description", content: "Analyze workplace images and inspect worker-level PPE compliance in the investigation canvas." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: DetectionPage,
});

/**
 * Detection Workspace: upload an image, run the (demo) analysis and inspect
 * worker-level PPE compliance on a synchronized investigation canvas. The
 * service layer is swapped for the real backend without any UI change — see
 * src/services/detectionService.ts.
 */
function DetectionPage() {
  const flow = useDetectionFlow();

  // Listen for media forwarded from the dashboard QuickUploadHub
  useEffect(() => {
    if (typeof window === "undefined") return;

    const pending = (window as unknown as { __SAFEOPS_PENDING_FILE__?: File }).__SAFEOPS_PENDING_FILE__;
    if (pending) {
      delete (window as unknown as { __SAFEOPS_PENDING_FILE__?: File }).__SAFEOPS_PENDING_FILE__;
      if (pending.type.startsWith("video/")) {
        flow.setVideo(pending);
      } else {
        flow.setImage(pending);
      }
      return;
    }

    if (sessionStorage.getItem("safeops_load_sample_video")) {
      sessionStorage.removeItem("safeops_load_sample_video");
      flow.loadSampleVideo(sampleFrame);
      return;
    }

    if (sessionStorage.getItem("safeops_load_sample")) {
      sessionStorage.removeItem("safeops_load_sample");
      flow.loadSample(sampleFrame);
    }
  }, []);

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Vision Console"
        title="Detection Workspace"
        description="Analyze workplace images and inspect worker-level PPE compliance."
      />

      <div className="grid items-start gap-5 lg:grid-cols-[320px_minmax(0,1fr)] 2xl:grid-cols-[320px_minmax(0,1fr)_360px]">
        {/* Left panel — input & controls */}
        <div className="order-1 space-y-5">
          <ImageUploader
            state={flow.state}
            image={flow.image}
            video={flow.video}
            webcamStream={flow.webcamStream}
            facingMode={flow.facingMode}
            activeMediaType={flow.mediaType}
            onSelectMode={flow.setMediaType}
            onFile={flow.setImage}
            onVideoFile={flow.setVideo}
            onStartWebcam={() => void flow.startWebcamStream()}
            onStopWebcam={flow.stopWebcamStream}
            onToggleFacingMode={flow.toggleFacingMode}
            onSample={() => void flow.loadSample(sampleFrame)}
            onSampleVideo={() => void flow.loadSampleVideo(sampleFrame)}
            onRemove={flow.removeMedia}
            onAnalyze={() => void flow.analyze()}
          />
          <DetectionControls
            state={flow.state}
            hasImage={Boolean(flow.image || flow.video || flow.webcamStream)}
            rules={flow.rules}
            onToggleRule={flow.toggleRule}
            onAnalyze={() => void flow.analyze()}
            onReset={flow.reset}
          />
        </div>

        {/* Center panel — visual investigation canvas */}
        <div className="order-2 min-w-0 space-y-4">
          <DetectionCanvas
            state={flow.state}
            mediaType={flow.mediaType}
            imageUrl={flow.image?.url ?? null}
            videoUrl={flow.video?.url ?? null}
            webcamStream={flow.webcamStream}
            facingMode={flow.facingMode}
            isVideoPlaying={flow.isVideoPlaying}
            continuousAnalysis={flow.continuousAnalysis}
            onTogglePlay={flow.toggleVideoPlay}
            onToggleContinuousAnalysis={flow.toggleContinuousAnalysis}
            result={flow.result}
            error={flow.error}
            selectedWorkerId={flow.selectedWorkerId}
            onSelectWorker={flow.setSelectedWorkerId}
          />
          {flow.result && <DetectionSummary result={flow.result} />}
          <InvestigationDetails
            image={flow.image}
            result={flow.result}
            state={flow.state}
            rules={flow.rules}
          />
        </div>

        {/* Right panel — worker inspection */}
        <div className="order-3 lg:col-span-2 2xl:col-span-1">
          <WorkerInspectionPanel
            result={flow.result}
            selectedWorkerId={flow.selectedWorkerId}
            onSelectWorker={flow.setSelectedWorkerId}
          />
        </div>
      </div>
    </div>
  );
}
