\# 🙏 Hastamudra Recognition System



Real-time Kathak hand gesture (mudra) detection using Computer Vision and Machine Learning.



\## 🎯 Project Overview

This system detects 5 Asamyukta Hasta Mudras in real-time using:

\- \*\*MediaPipe\*\* for hand landmark detection (21 points)

\- \*\*Random Forest\*\* classifier trained on self-collected data

\- \*\*FastAPI\*\* backend for serving predictions

\- \*\*Next.js + Tailwind\*\* frontend with glassmorphism UI



\## 🖐️ Supported Mudras

| Mudra | Meaning |

|-------|---------|

| Pataka | Flag |

| Tripataka | Three parts of a flag |

| Mushti | Clenched fist |

| Arala | Curved/Bent |

| Shikara | Peak/Mountain |



\## 📊 Model Performance

\- \*\*Algorithm:\*\* Random Forest (100 estimators)

\- \*\*Test Accuracy:\*\* 100%

\- \*\*Features:\*\* 63 normalized landmark coordinates

\- \*\*Training samples:\*\* 1200 per class (6000 total)



\## 🏗️ Project Structure
hastamudra-recognition/

├── src/

│   ├── collect\_data.py      # Data collection script

│   ├── train\_model.py       # Model training

│   ├── predict\_live.py      # Live prediction (Python UI)

│   └── api.py               # FastAPI backend

├── mudra-frontend/          # Next.js frontend

├── dataset/

│   └── landmarks.csv        # Collected landmark data

├── models/

│   ├── random\_forest.pkl    # Trained model

│   └── confusion\_matrix.png # Evaluation results

└── requirements.txt

## 🚀 Setup \& Run



\### Backend

```bash

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

uvicorn src.api:app --reload

```



\### Frontend

```bash

cd mudra-frontend

npm install

npm run dev

```



Open `http://localhost:3000`



\## 🧠 How It Works

1\. Webcam captures live video frames

2\. MediaPipe extracts 21 hand landmarks (63 coordinates)

3\. Coordinates normalized relative to wrist position and hand scale

4\. Random Forest predicts mudra from normalized features

5\. Result displayed in real-time with confidence score



\## 🛠️ Built With

\- Python, OpenCV, MediaPipe

\- Scikit-learn (Random Forest)

\- FastAPI, Uvicorn

\- Next.js, Tailwind CSS, Framer Motion

