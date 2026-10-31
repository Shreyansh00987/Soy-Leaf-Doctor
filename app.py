"""
FastAPI Triage Application and Mobile Web Interface for Soy Leaf Doctor (HARITVA).
Features:
- Dual Input: High-Resolution Photo Upload OR Field Video Canopy Scan (MP4/WebM)
- Trilingual UI: English, Hindi (हिंदी), and Marathi (मराठी)
- Pre-inference Optical Quality Gate (Blur & Exposure verification)
- Split-conformal Risk-Controlled Triage (|C|=1, 2, >=3)
- Explainable Grad-CAM Lesion Heatmap Overlays
"""
import json
import os
import sys
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import torch
import uvicorn

from src.model import CLASSES, build_model
from src.dataset import get_eval_transforms
from src.quality import check_image_quality
from src.conformal import ConformalCalibrator
from src.explainability import GradCAM, apply_heatmap_to_image, image_to_base64
from src.i18n import get_advisory

app = FastAPI(title="Soy Leaf Doctor (HARITVA)", version="1.0.0")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/mobilenet_v3.pt"
CONFORMAL_PATH = "models/conformal.json"

Q_HAT = 0.85
if os.path.exists(CONFORMAL_PATH):
    try:
        with open(CONFORMAL_PATH, "r") as f:
            cfg = json.load(f)
            Q_HAT = float(cfg.get("q_hat", 0.85))
    except Exception:
        pass

CALIBRATOR = ConformalCalibrator(alpha=0.10, q_hat=Q_HAT)
MODEL = None
GRAD_CAM = None
TRANSFORM = get_eval_transforms()


def get_model():
    global MODEL, GRAD_CAM
    if MODEL is None:
        MODEL = build_model(pretrained=False, checkpoint_path=MODEL_PATH if os.path.exists(MODEL_PATH) else None)
        MODEL.to(DEVICE)
        MODEL.eval()
        GRAD_CAM = GradCAM(MODEL)
    return MODEL, GRAD_CAM


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Soy Leaf Doctor · HARITVA</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap">
  <style>
    :root {
      --primary: #15803D;
      --primary-dark: #14532D;
      --bg: #F8FAFC;
      --surface: #FFFFFF;
      --text: #0F172A;
      --text-muted: #64748B;
      --border: #E2E8F0;
      --urgent: #DC2626;
      --warning: #D97706;
      --info: #2563EB;
      --radius: 12px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 16px;
    }
    .container {
      max-width: 680px;
      margin: 0 auto;
      background: var(--surface);
      border-radius: var(--radius);
      box-shadow: 0 4px 20px -2px rgba(0,0,0,0.06);
      border: 1px solid var(--border);
      overflow: hidden;
    }
    header {
      background: linear-gradient(135deg, #14532D 0%, #15803D 100%);
      color: white;
      padding: 20px 24px;
    }
    .header-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .badge {
      display: inline-block;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      background: rgba(255,255,255,0.2);
      padding: 3px 8px;
      border-radius: 6px;
    }
    .lang-toggle {
      display: flex;
      background: rgba(0,0,0,0.25);
      border-radius: 8px;
      padding: 2px;
      gap: 2px;
    }
    .lang-btn {
      background: none;
      border: none;
      color: rgba(255,255,255,0.75);
      padding: 4px 10px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      border-radius: 6px;
      transition: all 0.2s;
    }
    .lang-btn.active {
      background: white;
      color: var(--primary-dark);
    }
    h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.5px; }
    p.sub { font-size: 13px; opacity: 0.9; }

    .content { padding: 24px; }
    .upload-box {
      border: 2px dashed #CBD5E1;
      border-radius: 12px;
      padding: 26px 16px;
      text-align: center;
      cursor: pointer;
      background: #F8FAFC;
      transition: all 0.2s;
      margin-bottom: 20px;
    }
    .upload-box:hover {
      border-color: var(--primary);
      background: #F0FDF4;
    }
    .upload-icon { font-size: 36px; margin-bottom: 8px; }
    .upload-text { font-size: 14px; font-weight: 600; color: var(--text); }
    .upload-sub { font-size: 12px; color: var(--text-muted); margin-top: 4px; }
    input[type="file"] { display: none; }

    .btn {
      display: inline-block;
      width: 100%;
      background: var(--primary);
      color: white;
      border: none;
      padding: 14px 20px;
      font-size: 15px;
      font-weight: 600;
      border-radius: 10px;
      cursor: pointer;
      text-align: center;
      transition: background 0.2s;
    }
    .btn:hover { background: var(--primary-dark); }
    .btn:disabled { background: #94A3B8; cursor: not-allowed; }

    .preview-section {
      display: none;
      margin: 16px 0;
      text-align: center;
    }
    .preview-img, .preview-video {
      max-width: 100%;
      max-height: 240px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }

    .card {
      background: #FFFFFF;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
      margin-top: 16px;
    }
    .card.urgent { border-left: 4px solid var(--urgent); }
    .card.warning { border-left: 4px solid var(--warning); }
    .card.info { border-left: 4px solid var(--info); }
    .card.success { border-left: 4px solid var(--primary); }

    .triage-badge {
      display: inline-block;
      padding: 4px 10px;
      font-size: 12px;
      font-weight: 700;
      border-radius: 6px;
      margin-bottom: 8px;
    }
    .triage-badge.CONFIDENT { background: #DCFCE7; color: #166534; }
    .triage-badge.AMBIGUOUS { background: #FEF3C7; color: #92400E; }
    .triage-badge.ESCALATE { background: #F3E8FF; color: #6B21A8; }

    .prob-bar {
      height: 8px;
      background: #E2E8F0;
      border-radius: 4px;
      overflow: hidden;
      margin: 6px 0 12px;
    }
    .prob-fill {
      height: 100%;
      background: var(--primary);
      border-radius: 4px;
    }

    .advisory-row { margin-top: 10px; font-size: 13px; }
    .advisory-row b { display: block; color: var(--text-muted); font-size: 11px; text-transform: uppercase; margin-bottom: 2px; }

    .tab-btn {
      width: 100%;
      padding: 10px;
      background: #0F172A;
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      margin-top: 14px;
    }
    .tab-btn:hover { background: #1E293B; }

    #camContainer { display: none; margin-top: 12px; text-align: center; }
    #camImg { max-width: 100%; border-radius: 8px; border: 1px solid var(--border); }
    .spinner { display: none; text-align: center; padding: 20px; font-size: 14px; color: var(--text-muted); }

    .video-frames-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 12px;
    }
    .frame-card {
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px;
      font-size: 11px;
      text-align: center;
      background: #F8FAFC;
    }
    .frame-card img { width: 100%; height: 75px; object-fit: cover; border-radius: 4px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-top">
        <span class="badge">OFFLINE TRIAGE v1.0</span>
        <div class="lang-toggle">
          <button class="lang-btn" onclick="setLang('en')" id="btnEn">EN</button>
          <button class="lang-btn active" onclick="setLang('hi')" id="btnHi">हिन्दी</button>
          <button class="lang-btn" onclick="setLang('mr')" id="btnMr">मराठी</button>
        </div>
      </div>
      <h1>Soy Leaf Doctor · हरित्वा</h1>
      <p class="sub" id="hdrSub">सोयाबीन पत्ती व फसल वीडियो रोग पहचान प्रणाली</p>
    </header>

    <div class="content">
      <div class="upload-box" onclick="document.getElementById('fileInput').click()">
        <div class="upload-icon">📷 🎥</div>
        <div class="upload-text" id="uploadTitle">पत्ती का फोटो या खेत का वीडियो अपलोड करें</div>
        <div class="upload-sub" id="uploadSub">JPEG, PNG, MP4, WebM (Live Camera & Field Video)</div>
        <input type="file" id="fileInput" accept="image/*,video/*" capture="environment" onchange="handleFile(event)">
      </div>

      <div class="preview-section" id="previewSection">
        <img id="previewImg" class="preview-img" style="display:none;" alt="Leaf Photo">
        <video id="previewVideo" class="preview-video" style="display:none;" controls playsinline></video>
      </div>

      <button class="btn" id="analyzeBtn" onclick="startAnalysis()" disabled>निरीक्षण करें (Analyze)</button>
      <div class="spinner" id="spinner">प्रक्रियाधीन है... कृपया प्रतीक्षा करें...</div>

      <div id="resultContainer"></div>
    </div>
  </div>

  <canvas id="hiddenCanvas" style="display:none;"></canvas>

  <script>
    let currentLang = 'hi';
    let selectedFile = null;
    let isVideo = false;

    const UI_TEXT = {
      en: {
        sub: "Soybean Leaf & Video Canopy Diagnostic Triage System",
        upload: "Upload Leaf Photo or Field Video",
        subUpload: "JPEG, PNG, MP4, WebM (Live Camera & Field Video)",
        btn: "Analyze (Photo or Video)",
        analyzing: "Analyzing media... inspecting canopy frames...",
        confBadge: "CONFIDENT DIAGNOSIS",
        ambigBadge: "AMBIGUOUS / DIFFERENTIAL DIAGNOSIS",
        escBadge: "HIGH UNCERTAINTY / ESCALATE TO PATHOLOGIST",
        qFailTitle: "Image Quality Alert",
        organic: "Organic / Bio-control Advisory",
        chemical: "Emergency Chemical Treatment",
        symptoms: "Characteristic Symptoms",
        gradCamBtn: "Show Lesion Saliency Heatmap (Grad-CAM)",
        videoSummary: "Field Video Canopy Analysis Summary"
      },
      hi: {
        sub: "सोयाबीन पत्ती व फसल वीडियो रोग पहचान प्रणाली",
        upload: "पत्ती का फोटो या खेत का वीडियो अपलोड करें",
        subUpload: "JPEG, PNG, MP4, WebM (कैमरा या फील्ड वीडियो)",
        btn: "निरीक्षण करें (Analyze)",
        analyzing: "प्रक्रियाधीन है... वीडियो फ्रेम्स का विश्लेषण हो रहा है...",
        confBadge: "सटीक निदान (CONFIDENT)",
        ambigBadge: "संदिग्ध स्थिति (2 संभावित रोग)",
        escBadge: "अनिश्चित स्थिति / विशेषज्ञ को भेजें",
        qFailTitle: "फोटो गुणवत्ता चेतावनी",
        organic: "जैविक / देशी उपचार",
        chemical: "आपातकालीन रासायनिक छिड़काव",
        symptoms: "रोग के मुख्य लक्षण",
        gradCamBtn: "रोगग्रस्त भाग का हीटमैप देखें (Grad-CAM)",
        videoSummary: "खेत वीडियो कैनोपी निरीक्षण परिणाम"
      },
      mr: {
        sub: "सोयाबीन पान व पीक व्हिडिओ रोग निदान प्रणाली",
        upload: "पानाचा फोटो किंवा शेताचा व्हिडिओ अपलोड करा",
        subUpload: "JPEG, PNG, MP4, WebM (कॅमेरा किंवा व्हिडिओ)",
        btn: "तपासा (Analyze)",
        analyzing: "तपासणी सुरू आहे... व्हिडिओ फ्रेम्स तपासत आहे...",
        confBadge: "निश्चित निदान (CONFIDENT)",
        ambigBadge: "संशयास्पद स्थिती (२ संभाव्य आजार)",
        escBadge: "अनिश्चित स्थिती / तज्ज्ञांकडे पाठवा",
        qFailTitle: "फोटो गुणवत्ता सूचना",
        organic: "जैविक / घरगुती उपाय",
        chemical: "तातडीची रासायनिक फवारणी",
        symptoms: "रोगाची मुख्य लक्षणे",
        gradCamBtn: "रोगग्रस्त भागाचा हीटमॅप पहा (Grad-CAM)",
        videoSummary: "शेतातील व्हिडिओ तपासणी निकाल"
      }
    };

    function setLang(lang) {
      currentLang = lang;
      document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
      document.getElementById('btn' + lang.charAt(0).toUpperCase() + lang.slice(1)).classList.add('active');

      const t = UI_TEXT[lang];
      document.getElementById('hdrSub').innerText = t.sub;
      document.getElementById('uploadTitle').innerText = t.upload;
      document.getElementById('uploadSub').innerText = t.subUpload;
      document.getElementById('analyzeBtn').innerText = t.btn;
      document.getElementById('spinner').innerText = t.analyzing;
    }

    function handleFile(e) {
      if (e.target.files && e.target.files[0]) {
        selectedFile = e.target.files[0];
        isVideo = selectedFile.type.startsWith('video');

        const pImg = document.getElementById('previewImg');
        const pVid = document.getElementById('previewVideo');
        const pSec = document.getElementById('previewSection');

        pSec.style.display = 'block';
        document.getElementById('analyzeBtn').disabled = false;

        if (isVideo) {
          pImg.style.display = 'none';
          pVid.style.display = 'inline-block';
          pVid.src = URL.createObjectURL(selectedFile);
        } else {
          pVid.style.display = 'none';
          pImg.style.display = 'inline-block';
          const reader = new FileReader();
          reader.onload = (evt) => { pImg.src = evt.target.result; };
          reader.readAsDataURL(selectedFile);
        }
      }
    }

    async function startAnalysis() {
      if (!selectedFile) return;

      const btn = document.getElementById('analyzeBtn');
      const spinner = document.getElementById('spinner');
      const resultDiv = document.getElementById('resultContainer');

      btn.disabled = true;
      spinner.style.display = 'block';
      resultDiv.innerHTML = '';

      try {
        if (isVideo) {
          await analyzeVideo();
        } else {
          await analyzeSingleImage(selectedFile);
        }
      } catch (err) {
        resultDiv.innerHTML = `<div class="card urgent"><b>Error:</b> ${err.message}</div>`;
      } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
      }
    }

    async function analyzeSingleImage(blob) {
      const formData = new FormData();
      formData.append('file', blob, 'leaf.jpg');
      formData.append('lang', currentLang);

      const resp = await fetch('/api/triage', { method: 'POST', body: formData });
      const data = await resp.json();
      renderResult(data);
    }

    async function analyzeVideo() {
      const video = document.getElementById('previewVideo');
      const canvas = document.getElementById('hiddenCanvas');
      const ctx = canvas.getContext('2d');

      // Extract 3 representative keyframes at 20%, 50%, 80% duration
      const duration = video.duration || 3.0;
      const timestamps = [duration * 0.25, duration * 0.50, duration * 0.75];
      const frameBlobs = [];

      for (let t of timestamps) {
        video.currentTime = t;
        await new Promise(r => video.onseeked = r);

        canvas.width = video.videoWidth || 384;
        canvas.height = video.videoHeight || 384;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.85));
        frameBlobs.push({ time: t.toFixed(1), blob: blob });
      }

      // Analyze each frame
      const frameResults = [];
      for (let f of frameBlobs) {
        const formData = new FormData();
        formData.append('file', f.blob, `frame_${f.time}s.jpg`);
        formData.append('lang', currentLang);
        const resp = await fetch('/api/triage', { method: 'POST', body: formData });
        const res = await resp.json();
        frameResults.push({ time: f.time, data: res });
      }

      renderVideoResults(frameResults);
    }

    function renderResult(data) {
      const t = UI_TEXT[currentLang];
      const res = document.getElementById('resultContainer');
      const q = data.quality;
      const tr = data.triage;
      const adv = data.advisory;

      let html = '';
      if (!q.passed) {
        html += `
          <div class="card warning">
            <span class="triage-badge AMBIGUOUS">${t.qFailTitle}</span>
            <p><b>${q.issue}</b></p>
            <p style="font-size:13px; color:var(--text-muted);">${q.recommendation}</p>
          </div>
        `;
      }

      let cardClass = 'success';
      let badgeText = t.confBadge;
      if (tr.triage_state === 'AMBIGUOUS') {
        cardClass = 'warning';
        badgeText = t.ambigBadge;
      } else if (tr.triage_state === 'ESCALATE') {
        cardClass = 'urgent';
        badgeText = t.escBadge;
      }

      const topProb = Math.round((tr.probabilities[tr.label] || 0) * 100);

      html += `
        <div class="card ${cardClass}">
          <span class="triage-badge ${tr.triage_state}">${badgeText}</span>
          <h2 style="font-size:18px; margin-bottom:4px;">${adv.name}</h2>
          <div style="font-size:12px; color:var(--text-muted);">
            Confidence: <b>${tr.confidence_level}</b> (${topProb}%) · Prediction Set: [${tr.prediction_set.join(', ')}]
          </div>
          <div class="prob-bar"><div class="prob-fill" style="width: ${topProb}%;"></div></div>

          <div class="advisory-row">
            <b>${t.symptoms}</b>
            <div>${adv.symptoms}</div>
          </div>
          <div class="advisory-row" style="color:#166534;">
            <b>${t.organic}</b>
            <div>🌱 ${adv.organic}</div>
          </div>
          <div class="advisory-row" style="color:#991B1B;">
            <b>${t.chemical}</b>
            <div>🧪 ${adv.chemical}</div>
          </div>

          <button class="tab-btn" onclick="toggleCam()">${t.gradCamBtn}</button>
          <div id="camContainer">
            <img id="camImg" src="data:image/png;base64,${data.cam_base64}" alt="Grad-CAM Lesion Heatmap">
            <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">Heatmap indicates foliar lesions that activated the model.</div>
          </div>
        </div>
      `;

      res.innerHTML = html;
    }

    function renderVideoResults(frameResults) {
      const t = UI_TEXT[currentLang];
      const res = document.getElementById('resultContainer');

      // Pick dominant disease across sampled frames
      const counts = {};
      for (let f of frameResults) {
        const lbl = f.data.triage.label;
        counts[lbl] = (counts[lbl] || 0) + 1;
      }
      let dominant = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
      const repData = frameResults[0].data;

      let html = `
        <div class="card success">
          <span class="triage-badge CONFIDENT">🎥 ${t.videoSummary}</span>
          <h2 style="font-size:18px; margin-bottom:4px;">${repData.advisory.name}</h2>
          <p style="font-size:12px; color:var(--text-muted);">Scanned ${frameResults.length} keyframes across video canopy stream.</p>
          
          <div class="video-frames-grid">
            ${frameResults.map(f => `
              <div class="frame-card">
                <img src="data:image/png;base64,${f.data.cam_base64}" alt="Keyframe ${f.time}s">
                <div style="font-weight:600; margin-top:4px;">${f.time}s: ${f.data.triage.label}</div>
              </div>
            `).join('')}
          </div>

          <div class="advisory-row" style="color:#166534; margin-top:14px;">
            <b>${t.organic}</b>
            <div>🌱 ${repData.advisory.organic}</div>
          </div>
          <div class="advisory-row" style="color:#991B1B;">
            <b>${t.chemical}</b>
            <div>🧪 ${repData.advisory.chemical}</div>
          </div>
        </div>
      `;

      res.innerHTML = html;
    }

    function toggleCam() {
      const box = document.getElementById('camContainer');
      box.style.display = box.style.display === 'block' ? 'none' : 'block';
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_TEMPLATE


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "device": str(DEVICE),
        "model_loaded": os.path.exists(MODEL_PATH),
        "classes": CLASSES,
        "conformal_q_hat": Q_HAT,
        "supports_video": True,
    }


@app.post("/api/triage")
async def triage(file: UploadFile = File(...), lang: str = Form("hi")):
    model, grad_cam = get_model()

    image = Image.open(file.file).convert("RGB")
    quality = check_image_quality(image)

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = model.predict_tta(tensor).squeeze().cpu().numpy()

    triage_res = CALIBRATOR.predict(probs, CLASSES)
    advisory = get_advisory(triage_res.label, lang=lang)

    cam_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    heatmap = grad_cam.generate_heatmap(cam_tensor, target_class_idx=CLASSES.index(triage_res.label))
    blended = apply_heatmap_to_image(image, heatmap, alpha=0.45)
    cam_base64 = image_to_base64(blended)

    return {
        "quality": {
            "passed": quality.passed,
            "blur_score": quality.blur_score,
            "brightness": quality.brightness,
            "issue": quality.issue,
            "recommendation": quality.recommendation,
        },
        "triage": {
            "label": triage_res.label,
            "prediction_set": triage_res.prediction_set,
            "set_size": triage_res.set_size,
            "triage_state": triage_res.triage_state,
            "confidence_level": triage_res.confidence_level,
            "probabilities": triage_res.probabilities,
        },
        "advisory": advisory,
        "cam_base64": cam_base64,
    }


@app.get("/health")
def health_check():
    """Health check endpoint for cloud load balancers and Render health probes."""
    return {
        "status": "healthy",
        "service": "Soy Leaf Doctor (HARITVA)",
        "model": "MobileNetV3-Large",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
