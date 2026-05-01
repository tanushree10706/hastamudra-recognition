"""
predict_live.py
---------------
Phase 4 — Live Prediction

What this does:
  - Loads the trained Random Forest model
  - Opens webcam with hand skeleton
  - Predicts mudra name in real time
  - Shows prediction + confidence score on screen

Run: python src\predict_live.py
"""

import cv2
import mediapipe as mp
import pickle
import numpy as np
import collections

# ── Load trained model ────────────────────────────────────────────────────────
print("Loading model...")
with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)
print(f"  Classes: {model.classes_}")
print("  Model loaded!\n")

# ── MediaPipe setup ───────────────────────────────────────────────────────────
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

# ── Smoothing setup ───────────────────────────────────────────────────────────
# Instead of showing raw prediction every frame (flickery),
# we keep last 10 predictions and show the most common one.
# This is called a "majority vote buffer" — makes display stable.
BUFFER_SIZE = 10
prediction_buffer = collections.deque(maxlen=BUFFER_SIZE)

# ── MediaPipe model ───────────────────────────────────────────────────────────
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7
)

# ── Webcam ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Live prediction started!")
print("Show a mudra to the camera.")
print("Press Q to quit.\n")

# ── Confidence color helper ───────────────────────────────────────────────────
def confidence_color(conf):
    """Green if confident, yellow if unsure, red if low."""
    if conf >= 0.85:
        return (0, 255, 0)    # Green
    elif conf >= 0.60:
        return (0, 165, 255)  # Orange
    else:
        return (0, 0, 255)    # Red

# ── Main loop ─────────────────────────────────────────────────────────────────
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        prediction  = "No hand"
        confidence  = 0.0
        hand_detected = False

        # ── If hand detected → predict ────────────────────────────────────────
        if result.hand_landmarks:
            hand_detected = True
            hand = result.hand_landmarks[0]

            # Draw skeleton
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for start, end in CONNECTIONS:
                cv2.line(frame, points[start], points[end], (0, 200, 255), 2)
            for cx, cy in points:
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Build feature vector (same 63 numbers as training)
           # Build normalized feature vector (must match training)
           # Must match training normalization exactly
            wrist = hand[0]
            scale = max(
                abs(hand[9].x - wrist.x) + abs(hand[9].y - wrist.y),
                1e-6
            )
            features = []
            for lm in hand:
                features += [
                    (lm.x - wrist.x) / scale,
                    (lm.y - wrist.y) / scale,
                    (lm.z - wrist.z) / scale
                ]
            features = np.array(features).reshape(1, -1)

            # Predict
            raw_pred   = model.predict(features)[0]
            proba      = model.predict_proba(features)[0]
            confidence = np.max(proba)

            # Add to smoothing buffer
            prediction_buffer.append(raw_pred)

            # Majority vote from buffer
            prediction = collections.Counter(prediction_buffer).most_common(1)[0][0]

        # ── Draw UI ───────────────────────────────────────────────────────────
        # Dark overlay bar at top for readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Prediction text (big)
        pred_color = confidence_color(confidence) if hand_detected else (128, 128, 128)
        cv2.putText(frame, prediction, (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, pred_color, 3)

        # Confidence score
        conf_text = f"Confidence: {confidence*100:.1f}%" if hand_detected else ""
        cv2.putText(frame, conf_text, (15, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, pred_color, 2)

        # Confidence bar
        if hand_detected:
            bar_w = int(confidence * (w - 30))
            cv2.rectangle(frame, (15, 115), (w-15, 133), (50,50,50), -1)
            cv2.rectangle(frame, (15, 115), (15+bar_w, 133), pred_color, -1)

        # Press Q hint
        cv2.putText(frame, "Press Q to quit", (w-200, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

        cv2.imshow("Hasta Mudra Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Done.")