"use client";

import { useEffect, useRef, useState, useCallback } from "react";

const MUDRAS = [
  { name: "Pataka", emoji: "🚩", meaning: "Flag", sanskrit: "पताका", usage: "Represents clouds, forest, river, blessing", mythology: "Used to depict Lord Vishnu's Sudarshana Chakra", rasa: "Shanta (Peace), Vira (Heroism)" },
  { name: "Tripataka", emoji: "🔱", meaning: "Three parts of a flag", sanskrit: "त्रिपताका", usage: "Represents a crown, tree, flame of a lamp", mythology: "Associated with Lord Shiva's trident", rasa: "Adbhuta (Wonder), Vira (Heroism)" },
  { name: "Mushti", emoji: "✊", meaning: "Clenched fist", sanskrit: "मुष्टि", usage: "Represents holding something, combat, strength", mythology: "Symbolizes the strength of Hanuman", rasa: "Raudra (Fury), Vira (Heroism)" },
  { name: "Arala", emoji: "🌬️", meaning: "Curved / Bent", sanskrit: "अराल", usage: "Represents drinking nectar, the wind god Vayu", mythology: "Associated with Vayu the wind deity", rasa: "Sringara (Love), Karuna (Compassion)" },
  { name: "Shikara", emoji: "🏔️", meaning: "Peak / Mountain", sanskrit: "शिखर", usage: "Represents a bow, pillar, husband", mythology: "Symbolizes Mount Meru, the cosmic mountain", rasa: "Adbhuta (Wonder), Shanta (Peace)" },
];

const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

export default function Home() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const intervalRef = useRef(null);
  const isProcessing = useRef(false);
  const streamRef = useRef(null);

  const [started, setStarted] = useState(false);
  const [detected, setDetected] = useState(false);
  const [mudra, setMudra] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [info, setInfo] = useState(null);
  const [landmarks, setLandmarks] = useState([]);
  const [showGallery, setShowGallery] = useState(false);
  const [error, setError] = useState(null);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      setStarted(true);
      setError(null);
    } catch(err) {
      setError("Camera error: " + err.message);
    }
  }, []);

  useEffect(() => {
    if (started && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [started]);

  const sendFrame = useCallback(async () => {
    if (isProcessing.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;
    isProcessing.current = true;

    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {
      if (!blob) { isProcessing.current = false; return; }
      const form = new FormData();
      form.append("file", blob, "frame.jpg");
      try {
        const res = await fetch("http://127.0.0.1:8000/predict", { method: "POST", body: form });
        const data = await res.json();
        if (data.detected) {
          setDetected(true);
          setMudra(data.mudra);
          setConfidence(data.confidence);
          setInfo(data.info);
          setLandmarks(data.landmarks);
        } else {
          setDetected(false);
          setLandmarks([]);
        }
      } catch {
        setError("Backend not running");
      } finally {
        isProcessing.current = false;
      }
    }, "image/jpeg", 0.5);
  }, []);

  useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay || !landmarks.length) {
      if (overlay) overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
      return;
    }
    overlay.width = overlay.offsetWidth;
    overlay.height = overlay.offsetHeight;
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const W = overlay.width;
    const H = overlay.height;

    ctx.strokeStyle = "rgba(183, 109, 255, 0.7)";
    ctx.lineWidth = 1.5;
    CONNECTIONS.forEach(([s, e]) => {
      const a = landmarks[s], b = landmarks[e];
      if (!a || !b) return;
      ctx.beginPath();
      ctx.moveTo((1 - a.x) * W, a.y * H);
      ctx.lineTo((1 - b.x) * W, b.y * H);
      ctx.stroke();
    });
    landmarks.forEach((lm) => {
      ctx.beginPath();
      ctx.arc((1 - lm.x) * W, lm.y * H, 3, 0, Math.PI * 2);
      ctx.fillStyle = "#b76dff";
      ctx.fill();
    });
  }, [landmarks]);

  useEffect(() => {
    if (started) intervalRef.current = setInterval(sendFrame, 300);
    return () => clearInterval(intervalRef.current);
  }, [started, sendFrame]);

  const currentMudraInfo = mudra ? MUDRAS.find(m => m.name === mudra) : null;

  return (
    <div style={{ position: "fixed", inset: 0, background: "#000", overflow: "hidden", fontFamily: "'Space Mono', monospace" }}>

      {/* Full screen video */}
      {started && (
        <video
          ref={videoRef}
          autoPlay muted playsInline
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)", filter: "brightness(0.55) grayscale(0.15)" }}
        />
      )}

      {/* Skeleton overlay */}
      <canvas
        ref={overlayRef}
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none", zIndex: 2 }}
      />

      {/* Corner brackets */}
      {["tl","tr","bl","br"].map(pos => (
        <div key={pos} style={{
          position: "absolute", width: "32px", height: "32px", zIndex: 10,
          ...(pos === "tl" ? { top: "24px", left: "24px", borderTop: "2px solid rgba(255,255,255,0.25)", borderLeft: "2px solid rgba(255,255,255,0.25)" } : {}),
          ...(pos === "tr" ? { top: "24px", right: "24px", borderTop: "2px solid rgba(255,255,255,0.25)", borderRight: "2px solid rgba(255,255,255,0.25)" } : {}),
          ...(pos === "bl" ? { bottom: "24px", left: "24px", borderBottom: "2px solid rgba(255,255,255,0.25)", borderLeft: "2px solid rgba(255,255,255,0.25)" } : {}),
          ...(pos === "br" ? { bottom: "24px", right: "24px", borderBottom: "2px solid rgba(255,255,255,0.25)", borderRight: "2px solid rgba(255,255,255,0.25)" } : {}),
        }} />
      ))}

      {/* Scanning line */}
      {started && detected && (
        <div style={{
          position: "absolute", left: 0, right: 0, height: "2px", zIndex: 3, pointerEvents: "none",
          background: "linear-gradient(90deg, transparent, rgba(183,109,255,0.6), transparent)",
          animation: "scan 3s linear infinite",
        }} />
      )}

      {/* Top bar */}
      <header style={{ position: "absolute", top: 0, left: 0, right: 0, zIndex: 10, display: "flex", justifyContent: "space-between", alignItems: "center", padding: "20px 32px", background: "linear-gradient(to bottom, rgba(0,0,0,0.7), transparent)" }}>
        <div style={{ fontSize: "18px", fontWeight: 700, letterSpacing: "3px", color: "#b76dff", textTransform: "uppercase" }}>
          MUDRA_VISION
        </div>
        <div style={{ display: "flex", gap: "32px" }}>
          <span style={{ color: "#b76dff", fontSize: "12px", letterSpacing: "1px", cursor: "pointer" }}>CAPTURE</span>
          <span
            onClick={() => setShowGallery(!showGallery)}
            style={{ color: showGallery ? "#b76dff" : "rgba(255,255,255,0.4)", fontSize: "12px", letterSpacing: "1px", cursor: "pointer" }}
          >
            LIBRARY
          </span>
        </div>
        {/* Confidence badge */}
        {detected && (
          <div style={{ background: "rgba(0,0,0,0.6)", backdropFilter: "blur(8px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "4px", padding: "8px 14px" }}>
            <div style={{ fontSize: "9px", letterSpacing: "2px", color: "rgba(255,255,255,0.4)", marginBottom: "2px" }}>CONFIDENCE</div>
            <div style={{ fontSize: "20px", fontWeight: 600, color: "#4ae176" }}>{(confidence * 100).toFixed(1)}%</div>
          </div>
        )}
      </header>

      {/* Center title */}
      <div style={{ position: "absolute", top: "80px", left: 0, right: 0, textAlign: "center", zIndex: 10 }}>
        <h1 style={{ fontSize: "clamp(24px, 4vw, 42px)", fontWeight: 600, color: "white", margin: 0 }}>
        kathak <span style={{ color: "#b76dff" }}>मुद्राः</span>
        </h1>
      </div>

      {/* Not started — center prompt */}
      {!started && (
        <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", zIndex: 10 }}>
          <div style={{ fontSize: "48px", marginBottom: "24px" }}>🙏</div>
          <p style={{ color: "rgba(255,255,255,0.5)", fontSize: "12px", letterSpacing: "3px", marginBottom: "32px", textTransform: "uppercase" }}>
            enable audio and camera access
          </p>
          <button
            onClick={startCamera}
            style={{ padding: "14px 48px", background: "transparent", border: "1px solid rgba(183,109,255,0.6)", borderRadius: "4px", color: "#b76dff", fontSize: "12px", letterSpacing: "3px", textTransform: "uppercase", cursor: "pointer" }}
          >
            BEGIN
          </button>
          {error && <p style={{ color: "#ff6b6b", marginTop: "16px", fontSize: "12px" }}>{error}</p>}
        </div>
      )}

      {/* Left panel — frame rate / latency */}
      {started && (
        <div style={{ position: "absolute", left: "32px", top: "50%", transform: "translateY(-50%)", zIndex: 10, width: "140px" }}>
          <div style={{ background: "rgba(0,0,0,0.4)", backdropFilter: "blur(8px)", borderRadius: "8px", padding: "16px" }}>
            <div style={{ marginBottom: "16px" }}>
              <div style={{ fontSize: "9px", letterSpacing: "2px", color: "rgba(255,255,255,0.4)", marginBottom: "6px" }}>FRAME_RATE</div>
              <div style={{ height: "2px", background: "rgba(255,255,255,0.1)", borderRadius: "2px" }}>
                <div style={{ height: "100%", width: "60%", background: "#b76dff", borderRadius: "2px" }} />
              </div>
            </div>
            <div>
              <div style={{ fontSize: "9px", letterSpacing: "2px", color: "rgba(255,255,255,0.4)", marginBottom: "6px" }}>LATENCY</div>
              <div style={{ height: "2px", background: "rgba(255,255,255,0.1)", borderRadius: "2px" }}>
                <div style={{ height: "100%", width: "20%", background: "#b76dff", borderRadius: "2px" }} />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Status indicator */}
      {started && (
        <div style={{ position: "absolute", bottom: "80px", left: "50%", transform: "translateX(-50%)", zIndex: 10, textAlign: "center" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "8px", marginBottom: "8px" }}>
            <div style={{
              width: "8px", height: "8px", borderRadius: "50%",
              background: detected ? "#4ae176" : "rgba(255,255,255,0.3)",
              boxShadow: detected ? "0 0 8px #4ae176" : "none",
            }} />
            <span style={{ fontSize: "10px", letterSpacing: "2px", color: "rgba(255,255,255,0.4)" }}>
              {detected ? "DETECTING" : "ALIGN YOUR HAND"}
            </span>
          </div>
          <div style={{ fontSize: "10px", letterSpacing: "1px", color: "rgba(255,255,255,0.25)" }}>
            current mudra
          </div>
          <div style={{ fontSize: "14px", letterSpacing: "2px", color: detected ? "white" : "rgba(255,255,255,0.3)", marginTop: "4px" }}>
            {detected && mudra ? mudra.toUpperCase() : "UNKNOWN"}
          </div>
        </div>
      )}

      {/* Bottom left — mudra info card */}
      {detected && currentMudraInfo && (
        <div style={{ position: "absolute", bottom: "40px", left: "32px", zIndex: 10, maxWidth: "280px" }}>
          <div style={{ background: "rgba(0,0,0,0.65)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "8px", padding: "20px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
              <div style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#4ae176" }} />
              <h3 style={{ margin: 0, fontSize: "16px", fontWeight: 700, color: "white", letterSpacing: "1px" }}>
                {currentMudraInfo.name} <span style={{ color: "#b76dff" }}>{currentMudraInfo.sanskrit}</span>
              </h3>
            </div>
            <p style={{ margin: "0 0 8px", fontSize: "11px", color: "rgba(255,255,255,0.5)", letterSpacing: "1px" }}>
              "{currentMudraInfo.meaning}"
            </p>
            <p style={{ margin: 0, fontSize: "12px", color: "rgba(255,255,255,0.7)", lineHeight: 1.6 }}>
              {currentMudraInfo.usage}
            </p>
            <div style={{ marginTop: "12px", paddingTop: "12px", borderTop: "1px solid rgba(255,255,255,0.1)" }}>
              <span style={{ fontSize: "9px", letterSpacing: "2px", color: "rgba(255,255,255,0.3)" }}>RASA • </span>
              <span style={{ fontSize: "11px", color: "#b76dff" }}>{currentMudraInfo.rasa}</span>
            </div>
          </div>
        </div>
      )}

      {/* Library panel */}
      {showGallery && (
        <div style={{ position: "absolute", top: 0, right: 0, bottom: 0, width: "280px", zIndex: 20, background: "rgba(0,0,0,0.85)", backdropFilter: "blur(16px)", borderLeft: "1px solid rgba(255,255,255,0.1)", padding: "80px 20px 20px", overflowY: "auto" }}>
          <div style={{ fontSize: "9px", letterSpacing: "3px", color: "rgba(255,255,255,0.3)", marginBottom: "16px" }}>MUDRA LIBRARY</div>
          {MUDRAS.map(m => (
            <div key={m.name} style={{ padding: "14px", borderRadius: "8px", marginBottom: "8px", background: mudra === m.name && detected ? "rgba(183,109,255,0.15)" : "rgba(255,255,255,0.03)", border: mudra === m.name && detected ? "1px solid rgba(183,109,255,0.4)" : "1px solid rgba(255,255,255,0.06)", cursor: "pointer" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "4px" }}>
                <span style={{ fontSize: "18px" }}>{m.emoji}</span>
                <div>
                  <div style={{ fontSize: "13px", fontWeight: 600, color: "white" }}>{m.name}</div>
                  <div style={{ fontSize: "10px", color: "#b76dff" }}>{m.sanskrit}</div>
                </div>
                {mudra === m.name && detected && (
                  <div style={{ marginLeft: "auto", width: "6px", height: "6px", borderRadius: "50%", background: "#4ae176", boxShadow: "0 0 6px #4ae176" }} />
                )}
              </div>
              <p style={{ margin: 0, fontSize: "11px", color: "rgba(255,255,255,0.4)", lineHeight: 1.5 }}>{m.meaning}</p>
            </div>
          ))}
        </div>
      )}

      {/* Hidden canvas */}
      <canvas ref={canvasRef} style={{ display: "none" }} />

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
        @keyframes scan {
          0% { top: 0%; }
          100% { top: 100%; }
        }
      `}</style>
    </div>
  );
}