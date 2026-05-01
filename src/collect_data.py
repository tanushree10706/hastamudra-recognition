"""
collect_data.py
---------------
Phase 2 — Data Collection

What this does:
  - Opens webcam with hand skeleton (like test script)
  - You choose a mudra label
  - Press S to start capturing 200 frames
  - Saves all landmark coordinates + label to dataset/landmarks.csv

Run: python src\collect_data.py
"""

import cv2
import mediapipe as mp
import csv
import os
import time

# ── MediaPipe setup (same as test script) ────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ── Config ────────────────────────────────────────────────────────────────────
MUDRAS = ["Pataka", "Tripataka", "Mushti", "Arala", "Shikara"]
SAMPLES_PER_MUDRA = 200
DATASET_PATH = "dataset/landmarks.csv"

# ── CSV setup ─────────────────────────────────────────────────────────────────
# Create dataset folder if it doesn't exist
os.makedirs("dataset", exist_ok=True)

# Build CSV header: x0,y0,z0,x1,y1,z1,...,x20,y20,z20,label
header = []
for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]
header.append("label")

# Create CSV file with header if it doesn't exist yet
if not os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    print(f"Created new dataset file: {DATASET_PATH}")
else:
    print(f"Appending to existing dataset: {DATASET_PATH}")

# ── Main collection function ──────────────────────────────────────────────────
def collect_mudra(mudra_name, landmarker, cap):
    """
    Collects SAMPLES_PER_MUDRA frames for a given mudra.
    Shows live webcam feed with instructions.
    Press S to start capturing.
    """
    samples_collected = 0
    capturing = False

    print(f"\n{'='*50}")
    print(f"  Next mudra: {mudra_name}")
    print(f"  Get your hand ready, then press S to start.")
    print(f"{'='*50}\n")

    while samples_collected < SAMPLES_PER_MUDRA:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        h, w, _ = frame.shape

        # ── Draw skeleton if hand detected ────────────────────────────────────
        hand_detected = False
        row = None

        if result.hand_landmarks:
            hand_detected = True
            hand = result.hand_landmarks[0]
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            for start, end in CONNECTIONS:
                cv2.line(frame, points[start], points[end], (0, 200, 255), 2)
            for cx, cy in points:
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Build the data row: 63 landmark values
           # Build normalized data row
            # Subtract wrist (landmark 0) from all points
            # This makes position-invariant features
           # Normalize by wrist position AND hand scale
            # Step 1: subtract wrist → position invariant
            # Step 2: divide by hand scale → distance invariant
            wrist = hand[0]
            
            # Scale = distance from wrist to middle finger base (landmark 9)
            # This stays consistent regardless of how far hand is from camera
            scale = max(
                abs(hand[9].x - wrist.x) + abs(hand[9].y - wrist.y),
                1e-6  # avoid division by zero
            )
            
            row = []
            for lm in hand:
                row += [
                    (lm.x - wrist.x) / scale,
                    (lm.y - wrist.y) / scale,
                    (lm.z - wrist.z) / scale
                ]
        # ── Capture frame if recording ────────────────────────────────────────
        if capturing and hand_detected and row:
            row.append(mudra_name)
            with open(DATASET_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            samples_collected += 1

        # ── UI text on frame ──────────────────────────────────────────────────
        # Status bar at top
        status_color = (0, 255, 0) if capturing else (0, 165, 255)
        status_text = f"CAPTURING: {samples_collected}/{SAMPLES_PER_MUDRA}" if capturing else "Press S to start"
        cv2.putText(frame, status_color and status_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        # Mudra name
        cv2.putText(frame, f"Mudra: {mudra_name}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Hand detection indicator
        det_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        det_text = "Hand: DETECTED" if hand_detected else "Hand: NOT DETECTED"
        cv2.putText(frame, det_text, (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, det_color, 2)

        # Progress bar
        if capturing:
            bar_width = int((samples_collected / SAMPLES_PER_MUDRA) * (w - 20))
            cv2.rectangle(frame, (10, h-30), (w-10, h-10), (50,50,50), -1)
            cv2.rectangle(frame, (10, h-30), (10+bar_width, h-10), (0,255,0), -1)

        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and not capturing and hand_detected:
            capturing = True
            print(f"  Started capturing {mudra_name}...")
        elif key == ord('q'):
            print("  Quit by user.")
            return False  # Signal to stop everything

    print(f"  ✓ Collected {samples_collected} samples for {mudra_name}")
    return True  # Signal to continue


# ── Run collection for all mudras ─────────────────────────────────────────────
model_path = "hand_landmarker.task"
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("\n🙏 Hasta Mudra Data Collection")
print("================================")
print(f"Mudras to collect: {MUDRAS}")
print(f"Samples per mudra: {SAMPLES_PER_MUDRA}")
print(f"Total samples: {SAMPLES_PER_MUDRA * len(MUDRAS)}")
print("\nInstructions:")
print("  - Hold the mudra steady in front of camera")
print("  - Press S when ready to start capturing")
print("  - Stay still while capturing (progress bar fills up)")
print("  - Press Q anytime to quit\n")

with HandLandmarker.create_from_options(options) as landmarker:
    for mudra in MUDRAS:
        should_continue = collect_mudra(mudra, landmarker, cap)
        if not should_continue:
            break
        # Small pause between mudras
        time.sleep(1)

cap.release()
cv2.destroyAllWindows()

# ── Show final count ──────────────────────────────────────────────────────────
import pandas as pd
if os.path.exists(DATASET_PATH):
    try:
        import pandas as pd
        df = pd.read_csv(DATASET_PATH)
        print("\n📊 Dataset Summary:")
        print(df["label"].value_counts())
        print(f"\nTotal rows: {len(df)}")
        print(f"Saved to: {DATASET_PATH}")
    except:
        print(f"\nDataset saved to: {DATASET_PATH}")