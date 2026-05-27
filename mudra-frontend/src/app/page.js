"use client";

import { useEffect, useRef, useState, useCallback } from "react";

const MUDRAS = [
  {
    name: "Pataka",
    emoji: "🚩",
    meaning: "Flag",
    usage: "Represents clouds, forest, river, blessing",
    mythology: "Used to depict Lord Vishnu's Sudarshana Chakra",
    rasa: "Shanta (Peace), Vira (Heroism)",
  },
  {
    name: "Tripataka",
    emoji: "🔱",
    meaning: "Three parts of a flag",
    usage: "Represents a crown, tree, flame of a lamp",
    mythology: "Associated with Lord Shiva's trident",
    rasa: "Adbhuta (Wonder), Vira (Heroism)",
  },
  {
    name: "Mushti",
    emoji: "✊",
    meaning: "Clenched fist",
    usage: "Represents holding something, combat, strength",
    mythology: "Symbolizes the strength of Hanuman",
    rasa: "Raudra (Fury), Vira (Heroism)",
  },
  {
    name: "Arala",
    emoji: "🌬️",
    meaning: "Curved / Bent",
    usage: "Represents drinking nectar, the wind god Vayu",
    mythology: "Associated with Vayu the wind deity",
    rasa: "Sringara (Love), Karuna (Compassion)",
  },
  {
    name: "Shikara",
    emoji: "🏔️",
    meaning: "Peak / Mountain",
    usage: "Represents a bow, pillar, husband",
    mythology: "Symbolizes Mount Meru, the cosmic mountain",
    rasa: "Adbhuta (Wonder), Shanta (Peace)",
  },
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
  const intervalRef = useRef(null);

  const [started, setStarted] = useState(false);
  const [detected, setDetected] = useState(false);
  const [mudra, setMudra] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [info, setInfo] = useState(null);
  const [landmarks, setLandmarks] = useState([]);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState(null);

const streamRef = useRef(null);

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
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {
      if (!blob) return;
      const form = new FormData();
      form.append("file", blob, "frame.jpg");
      try {
        const res = await fetch("http://127.0.0.1:8000/predict", {
          method: "POST",
          body: form,
        });
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
        setError("Cannot connect to backend. Is it running?");
      }
    }, "image/jpeg", 0.8);
  }, []);

  useEffect(() => {
    const overlay = document.getElementById("overlay-canvas");
    if (!overlay || !landmarks.length) {
      if (overlay) {
        const ctx = overlay.getContext("2d");
        ctx.clearRect(0, 0, overlay.width, overlay.height);
      }
      return;
    }
    const ctx = overlay.getContext("2d");
    overlay.width = overlay.offsetWidth;
    overlay.height = overlay.offsetHeight;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const W = overlay.width;
    const H = overlay.height;

    ctx.strokeStyle = "rgba(168, 85, 247, 0.6)";
    ctx.lineWidth = 2;
    CONNECTIONS.forEach(([s, e]) => {
      const a = landmarks[s];
      const b = landmarks[e];
      if (!a || !b) return;
      ctx.beginPath();
      ctx.moveTo((1 - a.x) * W, a.y * H);
      ctx.lineTo((1 - b.x) * W, b.y * H);
      ctx.stroke();
    });

    landmarks.forEach((lm) => {
      ctx.beginPath();
      ctx.arc((1 - lm.x) * W, lm.y * H, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#f0abfc";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1;
      ctx.fill();
      ctx.stroke();
    });
  }, [landmarks]);

  useEffect(() => {
    if (started) {
      intervalRef.current = setInterval(sendFrame, 200);
    }
    return () => clearInterval(intervalRef.current);
  }, [started, sendFrame]);

  const displayInfo = selected
    ? MUDRAS.find((m) => m.name === selected.name)
    : info
    ? { ...info, name: mudra, emoji: info.emoji }
    : null;

  const displayName = selected ? selected.name : mudra;

  return (
    <main style={{ minHeight: "100vh", color: "white", background: "linear-gradient(135deg, #0a0a0f 0%, #0f0a1e 50%, #0a0f1e 100%)", fontFamily: "system-ui, sans-serif" }}>

      {/* Gradient orbs */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", overflow: "hidden" }}>
        <div style={{ position: "absolute", top: 0, left: 0, width: "500px", height: "500px", borderRadius: "50%", opacity: 0.2, background: "radial-gradient(circle, #7c3aed, transparent)" }} />
        <div style={{ position: "absolute", bottom: 0, right: 0, width: "400px", height: "400px", borderRadius: "50%", opacity: 0.15, background: "radial-gradient(circle, #db2777, transparent)" }} />
      </div>

      <div style={{ position: "relative", zIndex: 10, maxWidth: "1100px", margin: "0 auto", padding: "48px 24px" }}>

        {/* Hero */}
        <div style={{ textAlign: "center", marginBottom: "48px" }}>
          <p style={{ color: "#a855f7", fontSize: "12px", fontWeight: 600, letterSpacing: "4px", textTransform: "uppercase", marginBottom: "12px" }}>
            Bharatanatyam • Kathak • Classical Dance
          </p>
          <h1 style={{ fontSize: "clamp(2rem, 5vw, 4rem)", fontWeight: 800, marginBottom: "16px", background: "linear-gradient(135deg, #fff 0%, #c084fc 50%, #f472b6 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            Hastamudra Detection
          </h1>
          <p style={{ color: "#9ca3af", fontSize: "1.1rem" }}>
            Recognizing classical dance gestures in real-time using AI
          </p>
        </div>

        {/* Main grid */}
        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: "24px" }}>

          {/* Camera */}
          <div>
            <div style={{ position: "relative", borderRadius: "16px", overflow: "hidden", aspectRatio: "16/9", background: "rgba(255,255,255,0.05)", backdropFilter: "blur(12px)", border: detected ? "1px solid rgba(168,85,247,0.6)" : "1px solid rgba(255,255,255,0.1)", boxShadow: detected ? "0 0 30px rgba(168,85,247,0.4)" : "none", transition: "all 0.3s ease" }}>

              {!started ? (
                <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: "linear-gradient(135deg, #1a0533, #0d1533)" }}>
                  <div style={{ fontSize: "4rem", marginBottom: "24px" }}>🙏</div>
                  <h2 style={{ fontSize: "1.5rem", fontWeight: 600, marginBottom: "8px" }}>Ready to detect</h2>
                  <p style={{ color: "#9ca3af", marginBottom: "32px", fontSize: "0.9rem" }}>Allow camera access to begin</p>
                  <button onClick={startCamera} style={{ padding: "12px 32px", borderRadius: "50px", fontWeight: 600, color: "white", border: "none", cursor: "pointer", fontSize: "1rem", background: "linear-gradient(135deg, #7c3aed, #db2777)" }}>
                    Enable Camera
                  </button>
                  {error && <p style={{ color: "#f87171", marginTop: "16px", fontSize: "0.85rem" }}>{error}</p>}
                </div>
              ) : (
                <>
                  <video
                    ref={videoRef}
                    autoPlay
                    muted
                    playsInline
                    style={{ width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)" }}
                  />
                  <canvas id="overlay-canvas" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }} />

                  {/* Status badge */}
                  <div style={{ position: "absolute", top: "16px", left: "16px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", padding: "6px 12px", borderRadius: "50px", fontSize: "12px", fontWeight: 500, background: "rgba(0,0,0,0.5)", backdropFilter: "blur(8px)", border: "1px solid rgba(255,255,255,0.1)", color: detected ? "#4ade80" : "#9ca3af" }}>
                      <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: detected ? "#4ade80" : "#6b7280" }} />
                      {detected ? "Detecting" : "No hand"}
                    </div>
                  </div>

                  {/* Bottom label */}
                  {detected && mudra ? (
                    <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, padding: "24px", background: "linear-gradient(to top, rgba(0,0,0,0.85), transparent)" }}>
                      <p style={{ fontSize: "2.5rem", fontWeight: 800, background: "linear-gradient(135deg, #fff, #c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                        {mudra}
                      </p>
                      <div style={{ display: "flex", alignItems: "center", gap: "12px", marginTop: "8px" }}>
                        <div style={{ flex: 1, height: "4px", borderRadius: "4px", background: "rgba(255,255,255,0.1)" }}>
                          <div style={{ height: "100%", borderRadius: "4px", background: "linear-gradient(90deg, #7c3aed, #f472b6)", width: `${confidence * 100}%`, transition: "width 0.3s ease" }} />
                        </div>
                        <span style={{ fontSize: "0.85rem", color: "#d1d5db" }}>{(confidence * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  ) : (
                    <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <p style={{ color: "#6b7280", fontSize: "1rem" }}>Show a mudra to begin</p>
                    </div>
                  )}
                </>
              )}
            </div>
            <canvas ref={canvasRef} style={{ display: "none" }} />
          </div>

          {/* Right panel */}
          <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>

            {/* Info card */}
            <div style={{ borderRadius: "16px", padding: "24px", background: "rgba(255,255,255,0.05)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)", minHeight: "200px" }}>
              {displayInfo ? (
                <>
                  <div style={{ fontSize: "2.5rem", marginBottom: "12px" }}>{displayInfo.emoji}</div>
                  <h2 style={{ fontSize: "1.5rem", fontWeight: 700, marginBottom: "4px" }}>{displayName}</h2>
                  <p style={{ color: "#a855f7", fontSize: "0.85rem", marginBottom: "16px" }}>"{displayInfo.meaning}"</p>
                  <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                    <div>
                      <p style={{ fontSize: "10px", color: "#6b7280", textTransform: "uppercase", letterSpacing: "2px", marginBottom: "4px" }}>Usage</p>
                      <p style={{ fontSize: "0.85rem", color: "#d1d5db" }}>{displayInfo.usage}</p>
                    </div>
                    <div>
                      <p style={{ fontSize: "10px", color: "#6b7280", textTransform: "uppercase", letterSpacing: "2px", marginBottom: "4px" }}>Mythology</p>
                      <p style={{ fontSize: "0.85rem", color: "#d1d5db" }}>{displayInfo.mythology}</p>
                    </div>
                    <div>
                      <p style={{ fontSize: "10px", color: "#6b7280", textTransform: "uppercase", letterSpacing: "2px", marginBottom: "4px" }}>Rasa</p>
                      <p style={{ fontSize: "0.85rem", color: "#d1d5db" }}>{displayInfo.rasa}</p>
                    </div>
                  </div>
                </>
              ) : (
                <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center" }}>
                  <div style={{ fontSize: "2rem", marginBottom: "12px" }}>🙏</div>
                  <p style={{ color: "#6b7280", fontSize: "0.85rem" }}>
                    {started ? "Detecting mudra..." : "Start camera to see mudra info"}
                  </p>
                </div>
              )}
            </div>

            {/* Mudra gallery */}
            <div style={{ borderRadius: "16px", padding: "16px", background: "rgba(255,255,255,0.05)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)" }}>
              <p style={{ fontSize: "10px", color: "#6b7280", textTransform: "uppercase", letterSpacing: "2px", marginBottom: "12px", paddingLeft: "4px" }}>
                Supported Mudras
              </p>
              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                {MUDRAS.map((m) => (
                  <button
                    key={m.name}
                    onClick={() => setSelected(selected?.name === m.name ? null : m)}
                    style={{ display: "flex", alignItems: "center", gap: "12px", padding: "10px 12px", borderRadius: "12px", textAlign: "left", border: mudra === m.name && detected ? "1px solid rgba(168,85,247,0.4)" : "1px solid transparent", background: mudra === m.name && detected ? "rgba(168,85,247,0.2)" : selected?.name === m.name ? "rgba(255,255,255,0.1)" : "transparent", cursor: "pointer", color: "white", width: "100%", transition: "all 0.2s ease" }}
                  >
                    <span style={{ fontSize: "1.2rem" }}>{m.emoji}</span>
                    <div>
                      <p style={{ fontSize: "0.85rem", fontWeight: 500, margin: 0 }}>{m.name}</p>
                      <p style={{ fontSize: "0.75rem", color: "#6b7280", margin: 0 }}>{m.meaning}</p>
                    </div>
                    {mudra === m.name && detected && (
                      <div style={{ marginLeft: "auto", width: "8px", height: "8px", borderRadius: "50%", background: "#a855f7" }} />
                    )}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div style={{ textAlign: "center", marginTop: "48px", color: "#4b5563", fontSize: "0.85rem" }}>
          Built with AI + Computer Vision • Random Forest + MediaPipe
        </div>
      </div>
    </main>
  );
}