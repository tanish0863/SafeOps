"""
LOGGING FORMAT SPECIFICATION

This file defines the exact format your vision pipeline should output predictions
so they can be parsed and compared with ground truth.

======= REQUIRED OUTPUT FORMAT =======

Your system must print to console (or log file) one entry per frame in this format:

  PREDICTION: frame={FRAME_ID} person={PERSON} helmet={HELMET} zone={ZONE}

Where:
  FRAME_ID = integer frame number (0-indexed or 1-indexed, must be consistent)
  PERSON   = "Yes" | "No" (detected person in frame)
  HELMET   = "Yes" | "No" | "N/A" (helmet visible; N/A if no person)
  ZONE     = "Yes" | "No" | "N/A" (person in restricted zone; N/A if no person)

======= OUTPUT LOCATION =======

Log to:
  1. Console (stderr or stdout) for real-time monitoring
  2. File (optional): log_predictions.txt for post-analysis

======= EXAMPLE OUTPUT =======

Frame processing loop:
  for frame_id, frame in enumerate(video_source):
      # ... run detection, tracking, zone check, PPE check ...
      person_detected = len(tracked_persons) > 0
      person_str = "Yes" if person_detected else "No"
      helmet_str = "N/A" if not person_detected else ("Yes" if all_helmets_on else "No")
      zone_str = "N/A" if not person_detected else ("Yes" if in_zone else "No")
      
      # Print prediction
      print(f"PREDICTION: frame={frame_id} person={person_str} helmet={helmet_str} zone={zone_str}")

Real console output (30 frames at ~30 FPS):
  PREDICTION: frame=0 person=No helmet=N/A zone=N/A
  PREDICTION: frame=1 person=No helmet=N/A zone=N/A
  PREDICTION: frame=2 person=No helmet=N/A zone=N/A
  PREDICTION: frame=3 person=No helmet=N/A zone=N/A
  PREDICTION: frame=4 person=No helmet=N/A zone=N/A
  PREDICTION: frame=5 person=Yes helmet=Yes zone=No
  PREDICTION: frame=6 person=Yes helmet=Yes zone=No
  PREDICTION: frame=7 person=Yes helmet=No zone=No
  PREDICTION: frame=8 person=Yes helmet=No zone=No
  PREDICTION: frame=9 person=Yes helmet=No zone=No
  PREDICTION: frame=10 person=Yes helmet=No zone=Yes
  PREDICTION: frame=11 person=Yes helmet=No zone=Yes
  PREDICTION: frame=12 person=Yes helmet=Yes zone=Yes
  PREDICTION: frame=13 person=Yes helmet=Yes zone=Yes
  PREDICTION: frame=14 person=No helmet=N/A zone=N/A

======= CONVERSION TO CSV =======

After collecting console output, extract into predictions.csv:
  1. Capture all lines starting with "PREDICTION:"
  2. Parse frame_id, person, helmet, zone
  3. Convert to CSV format for evaluation_metrics.py

Conversion script:

  import re

  with open("predictions.log") as infile, open("predictions.csv", "w") as outfile:
      outfile.write("frame_id,timestamp_seconds,person_present,helmet_visible,in_restricted_zone\\n")
      for line in infile:
          if "PREDICTION:" not in line:
              continue
          match = re.search(r"frame=(\d+) person=(\w+) helmet=(\w+) zone=(\w+)", line)
          if match:
              frame_id, person, helmet, zone = match.groups()
              timestamp = float(frame_id) / 30.0  # Assume 30 FPS
              outfile.write(f"{frame_id},{ timestamp:.2f},{person},{helmet},{zone}\\n")

======= VALIDATION CHECKLIST =======

Before running evaluation_metrics.py, ensure:
  ☐ prediction.csv exists with all required columns
  ☐ All person_present values are "Yes" or "No"
  ☐ All helmet_visible values are "Yes", "No", or "N/A"
  ☐ All in_restricted_zone values are "Yes", "No", or "N/A"
  ☐ frame_id column is populated (at least for non-skipped frames)
  ☐ No missing values for person_present (it's always detectable)
  ☐ Helmet and zone are N/A exactly when person_present is No

======= IMPLEMENTATION IN main.py =======

Add this to your SafetyPipeline class:

  class SafetyPipeline:
      def __init__(self, config, output_log="predictions.log"):
          # ... existing init ...
          self.output_log = open(output_log, "w")

      def process_frame(self, frame):
          # ... detection, tracking, zone, PPE, alert logic ...
          
          # Log prediction
          person_str = "Yes" if self.tracked_persons else "No"
          helmet_str = "N/A"
          zone_str = "N/A"
          
          if self.tracked_persons:
              # Helmet: check if ANY person has no helmet
              helmet_violations = sum(1 for p in self.tracked_persons if p.has_no_helmet)
              helmet_str = "No" if helmet_violations > 0 else "Yes"
              
              # Zone: check if ANY person in zone
              zone_violations = sum(1 for p in self.tracked_persons if p.in_restricted_zone)
              zone_str = "Yes" if zone_violations > 0 else "No"
          
          log_line = f"PREDICTION: frame={self.frame_count} person={person_str} helmet={helmet_str} zone={zone_str}"
          print(log_line)
          self.output_log.write(log_line + "\\n")
          self.frame_count += 1

      def cleanup(self):
          # ... existing cleanup ...
          self.output_log.close()

"""

# Example logging implementation
def example_logging():
    """
    Shows how to integrate logging into the SafetyPipeline.
    This is just pseudocode—adapt to your actual class structure.
    """
    frame_count = 0
    output_log = open("predictions.log", "w")

    # Simulated pipeline loop
    for frame_id in range(15):
        tracked_persons = []  # Would come from tracking
        person_detected = len(tracked_persons) > 0

        person_str = "Yes" if person_detected else "No"
        helmet_str = "N/A"
        zone_str = "N/A"

        # Build prediction
        if person_detected:
            helmet_violations = sum(
                1 for p in tracked_persons if p.has_no_helmet
            )  # pseudocode
            helmet_str = "No" if helmet_violations > 0 else "Yes"

            zone_violations = sum(
                1 for p in tracked_persons if p.in_restricted_zone
            )  # pseudocode
            zone_str = "Yes" if zone_violations > 0 else "No"

        # Log
        log_line = f"PREDICTION: frame={frame_id} person={person_str} helmet={helmet_str} zone={zone_str}"
        print(log_line)
        output_log.write(log_line + "\n")

    output_log.close()
    print("\n[INFO] Predictions logged to predictions.log")


if __name__ == "__main__":
    example_logging()
