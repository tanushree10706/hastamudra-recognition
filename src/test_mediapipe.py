import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── MediaPipe NEW API setup ───────────────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Download the model file first
import urllib.request
import os

model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        model_path
    )
    print("Downloaded!")

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

print("Webcam opened. Show your hand. Press Q to quit.\n")

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                print("── Landmarks detected ──")
                for idx, lm in enumerate(hand):
                    print(f"  Point {idx:02d}: x={lm.x:.3f}  y={lm.y:.3f}  z={lm.z:.3f}")

                # Draw dots manually
                # Draw dots + skeleton
                h, w, _ = frame.shape

                # Convert landmarks to pixel coordinates
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

                # MediaPipe's 21 landmark connections (which point connects to which)
                CONNECTIONS = [
                    (0,1),(1,2),(2,3),(3,4),        # Thumb
                    (0,5),(5,6),(6,7),(7,8),         # Index finger
                    (0,9),(9,10),(10,11),(11,12),     # Middle finger
                    (0,13),(13,14),(14,15),(15,16),   # Ring finger
                    (0,17),(17,18),(18,19),(19,20),   # Pinky
                    (5,9),(9,13),(13,17)              # Palm base
                ]

                # Draw lines (skeleton)
                for start, end in CONNECTIONS:
                    cv2.line(frame, points[start], points[end], (0, 200, 255), 2)

                # Draw dots on top of lines
                for cx, cy in points:
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        cv2.imshow("MediaPipe Hand Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Done.")