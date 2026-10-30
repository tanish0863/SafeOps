"""
Ground Truth Definition Template for YOLOv8 Vision System Testing

This is the format to manually define expected outputs for a test video.
Instructions:
1. Play your test video frame-by-frame or at slow speed
2. For each key event/frame, record what you EXPECT to see:
   - Frame number or timestamp
   - Is a person present?
   - If yes, is helmet visible?
   - If yes, is the person in a restricted zone?
3. Save as ground_truth.csv in your project
4. Use this with evaluation_metrics.py to measure accuracy

CSV Format (copy this table into ground_truth.csv):

frame_id,timestamp_seconds,person_present,helmet_visible,in_restricted_zone
1,0.0,No,N/A,N/A
2,0.03,No,N/A,N/A
3,0.06,No,N/A,N/A
4,0.1,No,N/A,N/A
5,0.13,No,N/A,N/A
6,0.16,Yes,Yes,No
7,0.2,Yes,Yes,No
8,0.23,Yes,No,No
9,0.26,Yes,No,No
10,0.3,Yes,No,No
11,0.33,Yes,No,Yes
12,0.36,Yes,No,Yes
13,0.4,Yes,Yes,Yes
14,0.43,Yes,Yes,Yes
15,0.46,No,N/A,N/A


RULES:
- person_present: 'Yes' or 'No'
- helmet_visible: 'Yes', 'No', or 'N/A' (N/A if no person)
- in_restricted_zone: 'Yes', 'No', or 'N/A' (N/A if no person)
- frame_id: Sequential frame number (0, 1, 2, ...)
- timestamp_seconds: (Optional) Human-readable time reference


REAL EXAMPLE WITH COMMENTARY:

frame_id,timestamp_seconds,person_present,helmet_visible,in_restricted_zone
1,0.0,No,N/A,N/A          # Empty frame before person appears
2,0.03,No,N/A,N/A         # Still empty
3,0.06,No,N/A,N/A         # Still empty
4,0.1,Yes,Yes,No          # Person enters frame, wearing helmet, outside zone
5,0.13,Yes,Yes,No         # Same person, still has helmet, outside zone
6,0.16,Yes,No,No          # VIOLATION: Person removes helmet, still outside zone
7,0.2,Yes,No,No           # Person still without helmet, outside zone
8,0.23,Yes,No,Yes         # VIOLATION: Person enters restricted zone WITHOUT helmet
9,0.26,Yes,No,Yes         # Continues in zone without helmet (multiple violations)
10,0.3,Yes,Yes,Yes        # Person puts helmet back ON, still in zone
11,0.33,Yes,Yes,Yes       # Person compliant (helmet + in zone = acceptable)
12,0.36,Yes,Yes,No        # Person exits zone, helmet still on = compliant
13,0.4,Yes,Yes,No         # Compliant
14,0.43,No,N/A,N/A        # Person leaves frame completely
15,0.46,No,N/A,N/A        # Empty again


HOW TO CREATE YOUR OWN GROUND TRUTH:

Step 1: Understand your test video
  - How long is it?
  - What violations occur?
  - Frame rate (30 fps? 24 fps?)

Step 2: Play at slow speed or frame-by-frame
  - Use VLC Media Player: Tools > Preferences > Hotkeys > Space = frame advance
  - Or use FFmpeg to extract frames as images

Step 3: For each significant frame:
  - Write down: Frame number, person present/absent, helmet yes/no, zone yes/no

Step 4: Fill CSV with at least:
  - All frames where person enters/exits
  - All frames where helmet changes state
  - All frames where zone changes state
  - Optional: every frame (more granular = more accurate evaluation)

Step 5: Save as 'ground_truth.csv'

Step 6: Define your system's output format (see logging_format.md)

Step 7: Run evaluation_metrics.py to compare


COMMON MISTAKES TO AVOID:

❌ Not recording frame numbers accurately
   → Use media player frame counter or FFmpeg

❌ Leaving 'helmet_visible' as Yes when person is absent
   → Set to N/A if person_present='No'

❌ Recording too few samples
   → Aim for at least 20-30 key events for good evaluation

❌ Inconsistent criteria for "in zone"
   → Decide: Does foot have to be inside? Or just bounding box overlap?
   → Document this and apply consistently


TIPS FOR ACCURACY:

- Record helmet state at the MOMENT it's visible, not future frames
- For zones: Be precise about boundaries (use pixel coords if needed)
- If person partially visible, count as "present"
- If object detection fails (person missing entirely), mark as absence
"""
