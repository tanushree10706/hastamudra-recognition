"""
api.py
------
FastAPI backend

What this does:
  - Exposes a POST /predict endpoint
  - Receives a webcam frame from React frontend
  - Runs MediaPipe + ML model on it
  - Returns predicted mudra + confidence score

Run: uvicorn src.api:app --reload
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import mediapipe as mp
from PIL import Image
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor

# -- Load model ----------------------------------------------------------------
print("Loading model...")
with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)
print(f"  Classes: {model.classes_}")
print("  Model ready!\n")

# -- MediaPipe setup -----------------------------------------------------------
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)
executor = ThreadPoolExecutor(max_workers=2)
print("MediaPipe ready!\n")

# -- Mudra metadata ------------------------------------------------------------
MUDRA_DATA = {
    "Pataka": {
        "meaning"  : "Flag",
        "usage"    : "Represents clouds, forest, river, blessing",
        "mythology": "Used to depict Lord Vishnu's Sudarshana Chakra",
        "rasa"     : "Shanta (Peace), Vira (Heroism)",
        "emoji"    : "🚩"
    },
    "Tripataka": {
        "meaning"  : "Three parts of a flag",
        "usage"    : "Represents a crown, tree, flame of a lamp",
        "mythology": "Associated with Lord Shiva's trident",
        "rasa"     : "Adbhuta (Wonder), Vira (Heroism)",
        "emoji"    : "🔱"
    },
    "Mushti": {
        "meaning"  : "Clenched fist",
        "usage"    : "Represents holding something, combat, strength",
        "mythology": "Symbolizes the strength of Hanuman",
        "rasa"     : "Raudra (Fury), Vira (Heroism)",
        "emoji"    : "✊"
    },
    "Arala": {
        "meaning"  : "Curved / Bent",
        "usage"    : "Represents drinking nectar, the wind god Vayu",
        "mythology": "Associated with Vayu the wind deity",
        "rasa"     : "Sringara (Love), Karuna (Compassion)",
        "emoji"    : "🌬️"
    },
    "Shikara": {
        "meaning"  : "Peak / Mountain",
        "usage"    : "Represents a bow, pillar, husband",
        "mythology": "Symbolizes Mount Meru, the cosmic mountain",
        "rasa"     : "Adbhuta (Wonder), Shanta (Peace)",
        "emoji"    : "🏔️"
    },
}

# -- FastAPI app ---------------------------------------------------------------
app = FastAPI(title="Hasta Mudra Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Normalize landmarks -------------------------------------------------------
def normalize(hand):
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
            (lm.z - wrist.z) / scale,
        ]
    return features

# -- Health check --------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Hasta Mudra API is running"}

# -- Predict endpoint ----------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        def process():
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
            pil_image = pil_image.resize((320, 240))
            frame = np.array(pil_image)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            return landmarker.detect(mp_image)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, process)

        if not result.hand_landmarks:
            return {
                "detected"  : False,
                "mudra"     : None,
                "confidence": 0.0,
                "landmarks" : [],
                "info"      : None,
            }

        hand         = result.hand_landmarks[0]
        features     = normalize(hand)
        features_arr = np.array(features).reshape(1, -1)
        prediction   = model.predict(features_arr)[0]
        proba        = model.predict_proba(features_arr)[0]
        confidence   = float(np.max(proba))
        landmarks    = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand]

        return {
            "detected"  : True,
            "mudra"     : prediction,
            "confidence": round(confidence, 4),
            "landmarks" : landmarks,
            "info"      : MUDRA_DATA.get(prediction, {}),
        }

    except Exception as e:
        return {"detected": False, "error": str(e)}