# 🙏 MUDRA_VISION — Kathak Hasta Mudra Detection

> Real-time Kathak hand gesture recognition using Computer Vision and Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Next.js](https://img.shields.io/badge/Next.js-16-black)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen)


## 🎯 What is this?

MUDRA_VISION is a full-stack AI application that detects classical Kathak hand gestures (Hasta Mudras) in real-time using your webcam.

Built by a Kathak dancer — this project combines cultural authenticity with modern ML engineering.


## ✨ Features

- 🎥 **Full-screen immersive UI** — camera feed as background with overlay UI
- 🤖 **Real-time ML detection** — Random Forest classifier trained on self-collected data
- 🖐️ **Hand skeleton overlay** — 21 landmark points drawn live on screen
- 📊 **Confidence scoring** — live confidence percentage per prediction
- 📚 **Mudra library** — Sanskrit names, meanings, mythology, and Rasa
- ⚡ **Fast inference** — async backend with frame skipping for smooth performance

## 🖐️ Supported Mudras

| Mudra | Sanskrit | Meaning | Rasa |
|-------|----------|---------|------|
| Pataka | पताका | Flag | Shanta, Vira |
| Tripataka | त्रिपताका | Three parts of flag | Adbhuta, Vira |
| Mushti | मुष्टि | Clenched fist | Raudra, Vira |
| Arala | अराल | Curved / Bent | Sringara, Karuna |
| Shikara | शिखर | Peak / Mountain | Adbhuta, Shanta |


## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest (100 estimators) |
| Test Accuracy | 100% |
| Features | 63 normalized landmark coordinates |
| Training samples | 1,200 per class (6,000 total) |
| Augmentation | 2x noise augmentation per sample |
| Normalization | Wrist-relative + scale-invariant |

## 🏗️ Project Structure
hastamudra-recognition/
├── src/
│   ├── collect_data.py      # Data collection with augmentation
│   ├── train_model.py       # Model training + evaluation
│   ├── predict_live.py      # Python live prediction (OpenCV)
│   └── api.py               # FastAPI backend
├── mudra-frontend/          # Next.js frontend
│   └── src/app/
│       ├── page.js          # Main UI component
│       ├── layout.js        # App layout
│       └── globals.css      # Global styles
├── dataset/
│   └── landmarks.csv        # 6000 self-collected samples
├── models/
│   ├── random_forest.pkl    # Trained model
│   └── confusion_matrix.png # Evaluation results
└── requirements.txt


## 🚀 Setup & Run

### Prerequisites
- Python 3.11+
- Node.js 18+
- Webcam

### Backend Setup
```bash
git clone https://github.com/tanushree10706/hastamudra-recognition.git
cd hastamudra-recognition

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
uvicorn src.api:app --reload
```

### Frontend Setup
```bash
cd mudra-frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.


## 🧠 How It Works
Webcam frame captured every 300ms
↓
Sent to FastAPI backend as JPEG
↓
MediaPipe extracts 21 hand landmarks
↓
Coordinates normalized (wrist-relative + scale-invariant)
↓
Random Forest predicts mudra from 63 features
↓
Result + confidence sent back to frontend
↓
UI updates with mudra name + skeleton overlay

## 🛠️ Tech Stack

**ML Pipeline**
- MediaPipe — hand landmark detection
- Scikit-learn — Random Forest classifier
- NumPy / Pandas — data processing
- OpenCV — image processing

**Backend**
- FastAPI — REST API
- Uvicorn — ASGI server
- PIL — image handling

**Frontend**
- Next.js 16 — React framework
- Tailwind CSS — styling
- Canvas API — skeleton drawing

## 👩‍💻 About

Built by **Tanushree** — Kathak dancer.

This project was born from a desire to bridge classical Indian dance with modern AI — making traditional art forms accessible and recognizable through technology.


## 📈 Future Work

- [ ] Expand to 25 Asamyukta Hasta Mudras
- [ ] Add Samyukta (two-handed) mudra detection
- [ ] Deploy to Vercel + Railway
- [ ] Mobile responsive UI
- [ ] Audio pronunciation of Sanskrit names
