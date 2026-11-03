#!/usr/bin/env python3
"""
SignalWell — Facial Emotion Recognition Session Evaluator
==========================================================
Manages evaluation sessions (1-minute and 2-minute durations), continuous data
collection, metrics aggregation, quality auditing, and multi-format reporting
(Structured JSON, Tabular CSV, and Standalone Interactive HTML Dashboard).

Strict Design Principles:
  - Missing faces are NEVER counted as Neutral.
  - Emotion distribution percentages are computed exclusively over valid classified frames.
  - Separate raw predictions from temporally smoothed display predictions.
  - Automatically assesses data sufficiency (flags sessions with < 50% face detection).
"""

import os
import csv
import json
import time
from datetime import datetime
from collections import Counter, defaultdict

import cv2
import numpy as np

try:
    from emotion_classifier import DISPLAY_CLASSES, EMOTION_EMOJIS
except ImportError:
    from SignalWell.emotion_classifier import DISPLAY_CLASSES, EMOTION_EMOJIS

# Emotion Color Palette (BGR for OpenCV, Hex for HTML)
EMOTION_COLORS_BGR = {
    'Happy':    (154, 227, 95),   # Emerald Green
    'Sad':      (225, 120, 85),   # Slate Blue
    'Disgust':  (85, 199, 122),   # Moss Green
    'Surprise': (247, 85, 168),   # Magenta / Violet
    'Fear':     (60, 150, 240),   # Warning Amber / Orange
    'Angry':    (68, 68, 239),    # Crimson Red
    'Neutral':  (145, 155, 165)   # Slate Gray
}

EMOTION_COLORS_HEX = {
    'Happy':    '#5FE39A',
    'Sad':      '#567AE1',
    'Disgust':  '#7AC755',
    'Surprise': '#F755A8',
    'Fear':     '#F0963C',
    'Angry':    '#EF4444',
    'Neutral':  '#919BA5'
}

class SessionEvaluator:
    """
    Manages session lifecycle, real-time sample storage, and final evaluation analytics.
    """
    def __init__(self, selected_duration_sec: int = 60):
        self.selected_duration_sec = selected_duration_sec
        self.state = "IDLE"  # "IDLE", "RUNNING", "PAUSED", "COMPLETED"

        self.session_id = ""
        self.start_wall_time = 0.0
        self.start_iso_time = ""
        self.end_iso_time = ""
        self.actual_duration_sec = 0.0

        # Frame Accounting
        self.total_captured_frames = 0
        self.total_analyzed_frames = 0
        self.total_valid_faces = 0
        self.total_classification_failures = 0
        self.frames_without_face = 0

        # Prediction Records
        self.samples = []  # List of dicts matching Section 5
        self.emotion_counter = Counter()
        self.confidence_by_emotion = defaultdict(list)

        # Cache of completed report
        self.completed_report = None

    def start_session(self, duration_sec: int = None):
        """Initializes and begins a new evaluation session."""
        if duration_sec is not None:
            self.selected_duration_sec = int(duration_sec)

        now = datetime.now()
        self.session_id = f"SES-{now.strftime('%Y%m%d-%H%M%S')}"
        self.start_wall_time = time.time()
        self.start_iso_time = now.isoformat()
        self.end_iso_time = ""
        self.actual_duration_sec = 0.0

        self.total_captured_frames = 0
        self.total_analyzed_frames = 0
        self.total_valid_faces = 0
        self.total_classification_failures = 0
        self.frames_without_face = 0

        self.samples = []
        self.emotion_counter = Counter()
        self.confidence_by_emotion = defaultdict(list)
        self.completed_report = None

        self.state = "RUNNING"
        print(f"\n[SessionEvaluator] [>] Started Session {self.session_id} (Target Duration: {self.selected_duration_sec}s)")

    def reset_session(self):
        """Resets the evaluator back to IDLE state."""
        self.state = "IDLE"
        self.session_id = ""
        self.start_wall_time = 0.0
        self.start_iso_time = ""
        self.end_iso_time = ""
        self.actual_duration_sec = 0.0
        self.samples.clear()
        self.emotion_counter.clear()
        self.confidence_by_emotion.clear()
        self.completed_report = None
        print("[SessionEvaluator] [R] Session reset to IDLE.")

    def get_elapsed_seconds(self) -> float:
        """Returns elapsed session seconds."""
        if self.state in ("RUNNING", "PAUSED"):
            return time.time() - self.start_wall_time
        elif self.state == "COMPLETED":
            return self.actual_duration_sec
        return 0.0

    def get_remaining_seconds(self) -> float:
        """Returns remaining seconds in the session."""
        elapsed = self.get_elapsed_seconds()
        return max(0.0, self.selected_duration_sec - elapsed)

    def is_time_up(self) -> bool:
        """Checks if session reached the selected duration."""
        if self.state == "RUNNING" and self.get_elapsed_seconds() >= self.selected_duration_sec:
            return True
        return False

    def record_frame(self,
                     has_face: bool,
                     raw_emotion: str = None,
                     raw_conf: float = 0.0,
                     smoothed_emotion: str = None,
                     probabilities: dict = None,
                     cls_error: bool = False):
        """
        Records per-frame observations during an active session.
        Missing faces are tracked for quality but NEVER recorded as Neutral.
        """
        self.total_captured_frames += 1

        if self.state != "RUNNING":
            return

        self.total_analyzed_frames += 1
        elapsed = self.get_elapsed_seconds()

        if not has_face:
            self.frames_without_face += 1
            return

        self.total_valid_faces += 1

        if cls_error or raw_emotion is None or probabilities is None:
            self.total_classification_failures += 1
            return

        # Valid classification sample
        sample_record = {
            "timestamp": datetime.now().isoformat(),
            "elapsedSeconds": round(elapsed, 2),
            "emotion": raw_emotion,
            "confidence": round(float(raw_conf), 4),
            "smoothedEmotion": smoothed_emotion or raw_emotion,
            "probabilities": {k: round(float(probabilities.get(k, 0.0)), 4) for k in DISPLAY_CLASSES}
        }
        self.samples.append(sample_record)

        # Update accumulators
        self.emotion_counter[raw_emotion] += 1
        self.confidence_by_emotion[raw_emotion].append(float(raw_conf))

    def finalize_session(self) -> dict:
        """
        Stops collecting predictions, calculates comprehensive evaluation metrics,
        and generates the final report.
        """
        if self.state == "COMPLETED" and self.completed_report is not None:
            return self.completed_report

        now = datetime.now()
        self.end_iso_time = now.isoformat()
        self.actual_duration_sec = round(time.time() - self.start_wall_time, 2)
        self.state = "COMPLETED"

        total_valid = len(self.samples)

        # ----------------------------------------------------
        # B. Expression Distribution (Strictly valid samples)
        # ----------------------------------------------------
        distribution = {}
        for c in DISPLAY_CLASSES:
            cnt = self.emotion_counter[c]
            pct = round((cnt / total_valid * 100.0), 2) if total_valid > 0 else 0.0
            distribution[c] = {
                "count": cnt,
                "percentage": pct
            }

        # ----------------------------------------------------
        # C. Most Frequently Detected Expression
        # ----------------------------------------------------
        if self.emotion_counter:
            most_freq_name, most_freq_count = self.emotion_counter.most_common(1)[0]
            most_freq_pct = round((most_freq_count / total_valid * 100.0), 2) if total_valid > 0 else 0.0
        else:
            most_freq_name = "None"
            most_freq_count = 0
            most_freq_pct = 0.0

        # ----------------------------------------------------
        # D. Confidence Analysis
        # ----------------------------------------------------
        all_confs = [s["confidence"] for s in self.samples]
        overall_avg_conf = round(float(np.mean(all_confs)), 4) if all_confs else 0.0

        confidence_analysis = {}
        for c in DISPLAY_CLASSES:
            confs = self.confidence_by_emotion[c]
            avg_c = round(float(np.mean(confs)), 4) if confs else 0.0
            confidence_analysis[c] = {
                "count": len(confs),
                "averageConfidence": avg_c
            }

        # ----------------------------------------------------
        # E. Temporal Analysis (Timeline Progression)
        # ----------------------------------------------------
        # Timeline step events: show expression changes or regular 5-second intervals
        timeline_events = []
        last_em = None
        for s in self.samples:
            cur_em = s["smoothedEmotion"]
            if cur_em != last_em:
                mins = int(s["elapsedSeconds"] // 60)
                secs = int(s["elapsedSeconds"] % 60)
                timeline_events.append({
                    "timeLabel": f"{mins:02d}:{secs:02d}",
                    "elapsedSeconds": s["elapsedSeconds"],
                    "emotion": cur_em,
                    "confidence": s["confidence"]
                })
                last_em = cur_em

        # ----------------------------------------------------
        # F. Data Quality & Validity Assessment
        # ----------------------------------------------------
        face_detection_success_rate = (
            round((self.total_valid_faces / self.total_analyzed_frames * 100.0), 2)
            if self.total_analyzed_frames > 0 else 0.0
        )
        missing_face_pct = round(100.0 - face_detection_success_rate, 2)

        # Thresholds: at least 40% of session frames must have a detected face,
        # and at least 20 valid predictions collected.
        is_sufficient = (face_detection_success_rate >= 40.0) and (total_valid >= 20)
        quality_status = (
            "SUFFICIENT DATA" if is_sufficient
            else "INSUFFICIENT DATA: Face detected in only a small portion of the session"
        )

        report = {
            "sessionInformation": {
                "sessionId": self.session_id,
                "startTime": self.start_iso_time,
                "endTime": self.end_iso_time,
                "selectedDurationSeconds": self.selected_duration_sec,
                "actualDurationSeconds": self.actual_duration_sec,
                "totalCapturedFrames": self.total_captured_frames,
                "totalAnalyzedFrames": self.total_analyzed_frames,
                "totalValidFaceDetections": self.total_valid_faces,
                "totalClassificationFailures": self.total_classification_failures,
                "totalValidPredictions": total_valid
            },
            "expressionDistribution": distribution,
            "mostFrequentlyDetectedExpression": {
                "emotion": most_freq_name,
                "count": most_freq_count,
                "percentage": most_freq_pct
            },
            "confidenceAnalysis": {
                "overallAverageConfidence": overall_avg_conf,
                "byExpression": confidence_analysis,
                "disclaimer": "Confidence reflects neural network softmax certainty, not verified ground truth."
            },
            "temporalAnalysis": {
                "significantEvents": timeline_events,
                "totalEvents": len(timeline_events)
            },
            "dataQuality": {
                "faceDetectionSuccessRatePct": face_detection_success_rate,
                "missingFaceRatePct": missing_face_pct,
                "missingOrInvalidPredictions": self.frames_without_face + self.total_classification_failures,
                "isDataSufficient": is_sufficient,
                "status": quality_status
            },
            "rawPredictions": self.samples
        }

        self.completed_report = report
        print(f"[SessionEvaluator] [OK] Session {self.session_id} finalized with {total_valid} valid predictions.")
        return report

    def export_json(self, filepath: str = None) -> str:
        """Exports the session report and raw samples to a formatted JSON file."""
        report = self.finalize_session()
        if not filepath:
            os.makedirs("logs", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join("logs", f"SignalWell_Session_{self.session_id}_{ts}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"[SessionEvaluator] [JSON] Exported JSON report to: {os.path.abspath(filepath)}")
        return filepath

    def export_csv(self, filepath: str = None) -> str:
        """Exports timestamped raw prediction samples to a CSV file."""
        report = self.finalize_session()
        if not filepath:
            os.makedirs("logs", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join("logs", f"SignalWell_Session_{self.session_id}_{ts}.csv")

        fieldnames = [
            "timestamp", "elapsedSeconds", "emotion", "confidence", "smoothedEmotion",
            "Happy", "Sad", "Disgust", "Surprise", "Fear", "Angry", "Neutral"
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in self.samples:
                row = {
                    "timestamp": s["timestamp"],
                    "elapsedSeconds": s["elapsedSeconds"],
                    "emotion": s["emotion"],
                    "confidence": s["confidence"],
                    "smoothedEmotion": s["smoothedEmotion"],
                }
                for c in DISPLAY_CLASSES:
                    row[c] = s["probabilities"].get(c, 0.0)
                writer.writerow(row)

        print(f"[SessionEvaluator] [CSV] Exported CSV predictions to: {os.path.abspath(filepath)}")
        return filepath

    def export_html_dashboard(self, filepath: str = None) -> str:
        """
        Generates an interactive HTML Visual Analytics Dashboard with Chart.js charts,
        summary cards, distribution breakdown, and temporal event log.
        """
        report = self.finalize_session()
        if not filepath:
            os.makedirs("reports", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join("reports", f"SignalWell_Evaluation_Report_{self.session_id}_{ts}.html")

        info = report["sessionInformation"]
        dist = report["expressionDistribution"]
        mf = report["mostFrequentlyDetectedExpression"]
        conf_ana = report["confidenceAnalysis"]
        quality = report["dataQuality"]
        events = report["temporalAnalysis"]["significantEvents"]

        # Chart.js Data
        labels_json = json.dumps(DISPLAY_CLASSES)
        counts_json = json.dumps([dist[c]["count"] for c in DISPLAY_CLASSES])
        pcts_json = json.dumps([dist[c]["percentage"] for c in DISPLAY_CLASSES])
        colors_json = json.dumps([EMOTION_COLORS_HEX[c] for c in DISPLAY_CLASSES])

        # Timeline chart data (downsampled if > 150 points)
        step = max(1, len(self.samples) // 100)
        chart_samples = self.samples[::step]
        time_labels_json = json.dumps([f"{s['elapsedSeconds']}s" for s in chart_samples])
        time_emotions_json = json.dumps([s['smoothedEmotion'] for s in chart_samples])
        time_conf_json = json.dumps([round(s['confidence'] * 100, 1) for s in chart_samples])

        quality_color = "#5FE39A" if quality["isDataSufficient"] else "#F97316"
        quality_badge = "RELIABLE DATA" if quality["isDataSufficient"] else "INSUFFICIENT SAMPLES"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SignalWell — Facial Emotion Evaluation Report ({info['sessionId']})</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  :root {{
    --bg: #0B0E14;
    --card: #141A23;
    --border: #232B39;
    --text: #EDE7DA;
    --muted: #8A93A6;
    --accent: #5FE39A;
    --accent-dim: rgba(95, 227, 154, 0.12);
    --warn: #E5A84B;
    --danger: #EF4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Plus Jakarta Sans', sans-serif;
    padding: 32px 24px;
    line-height: 1.5;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 28px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 20px;
    flex-wrap: wrap;
    gap: 16px;
  }}
  .header-left h1 {{
    font-size: 1.75rem;
    font-weight: 800;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .header-left p {{ color: var(--muted); font-size: 0.92rem; margin-top: 4px; }}
  .badge-session {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    background: var(--accent-dim);
    color: var(--accent);
    padding: 6px 12px;
    border-radius: 6px;
    border: 1px solid rgba(95, 227, 154, 0.3);
  }}
  .print-btn {{
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-family: inherit;
    font-weight: 600;
    font-size: 0.88rem;
    transition: all 0.2s;
  }}
  .print-btn:hover {{ background: #1B2230; border-color: var(--accent); }}

  /* KPI Grid */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
  }}
  .kpi-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
  }}
  .kpi-label {{
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 8px;
  }}
  .kpi-val {{
    font-size: 1.6rem;
    font-weight: 800;
    color: #fff;
    font-family: 'JetBrains Mono', monospace;
  }}
  .kpi-sub {{ font-size: 0.82rem; color: var(--muted); margin-top: 4px; }}

  /* Main Grid */
  .charts-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 28px;
  }}
  @media(max-width: 900px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}

  .chart-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px;
    display: flex;
    flex-direction: column;
  }}
  .chart-title {{
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .chart-wrap {{ position: relative; height: 280px; }}

  /* Full-width Timeline Card */
  .timeline-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 28px;
  }}
  .timeline-wrap {{ position: relative; height: 240px; }}

  /* Tables */
  .table-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 28px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }}
  th, td {{
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    color: var(--muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}
  td.mono {{ font-family: 'JetBrains Mono', monospace; }}

  /* Status Banner */
  .quality-banner {{
    padding: 14px 20px;
    border-radius: 10px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border: 1px solid;
    font-size: 0.94rem;
  }}

  @media print {{
    body {{ background: #fff; color: #000; padding: 0; }}
    .kpi-card, .chart-card, .timeline-card, .table-card {{ border: 1px solid #ccc; background: #fff; }}
    .print-btn {{ display: none; }}
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="header-left">
      <h1>SignalWell — Evaluation Report <span class="badge-session">{info['sessionId']}</span></h1>
      <p>Automated Real-Time Facial Emotion Session Analysis • Models: YOLO11n-face & InceptionResnetV1 (v6)</p>
    </div>
    <button class="print-btn" onclick="window.print()">🖨️ Print / Save PDF</button>
  </header>

  <!-- Data Quality Banner -->
  <div class="quality-banner" style="border-color: {quality_color}; background: rgba(95,227,154,0.06); color: {quality_color};">
    <div><strong>STATUS:</strong> {quality['status']}</div>
    <div style="font-family:'JetBrains Mono',monospace; font-weight:700;">{quality_badge} ({quality['faceDetectionSuccessRatePct']}% Face Coverage)</div>
  </div>

  <!-- KPI Grid -->
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Session Duration</div>
      <div class="kpi-val">{info['actualDurationSeconds']:.1f}s</div>
      <div class="kpi-sub">Target: {info['selectedDurationSeconds']}s ({info['selectedDurationSeconds']//60} min)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Valid Predictions</div>
      <div class="kpi-val">{info['totalValidPredictions']}</div>
      <div class="kpi-sub">Total Analyzed: {info['totalAnalyzedFrames']} frames</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Dominant Emotion</div>
      <div class="kpi-val" style="color: {EMOTION_COLORS_HEX.get(mf['emotion'], '#fff')}">{EMOTION_EMOJIS.get(mf['emotion'], '')} {mf['emotion']}</div>
      <div class="kpi-sub">{mf['percentage']}% of valid samples ({mf['count']} frames)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Average Confidence</div>
      <div class="kpi-val">{conf_ana['overallAverageConfidence']*100:.1f}%</div>
      <div class="kpi-sub">Face Track Rate: {quality['faceDetectionSuccessRatePct']}%</div>
    </div>
  </div>

  <!-- Charts Grid -->
  <div class="charts-grid">
    <div class="chart-card">
      <div class="chart-title">Emotion Distribution <span>Doughnut View</span></div>
      <div class="chart-wrap">
        <canvas id="donutChart"></canvas>
      </div>
    </div>
    <div class="chart-card">
      <div class="chart-title">Emotion Frequencies <span>Count Comparison</span></div>
      <div class="chart-wrap">
        <canvas id="barChart"></canvas>
      </div>
    </div>
  </div>

  <!-- Temporal Progression -->
  <div class="timeline-card">
    <div class="chart-title">Temporal Expression Timeline <span>Inference Stream</span></div>
    <div class="timeline-wrap">
      <canvas id="timeChart"></canvas>
    </div>
  </div>

  <!-- Detailed Statistics Table -->
  <div class="table-card">
    <div class="chart-title">Detailed 7-Category Affective Statistics</div>
    <table>
      <thead>
        <tr>
          <th>Category</th>
          <th>Predictions Count</th>
          <th>Distribution Percentage</th>
          <th>Average Confidence</th>
          <th>Visual Color</th>
        </tr>
      </thead>
      <tbody>
        {"".join([f'''<tr>
          <td><strong>{EMOTION_EMOJIS.get(c, '')} {c}</strong></td>
          <td class="mono">{dist[c]["count"]}</td>
          <td class="mono">{dist[c]["percentage"]:.1f}%</td>
          <td class="mono">{conf_ana['byExpression'][c]['averageConfidence']*100:.1f}%</td>
          <td><span style="display:inline-block;width:14px;height:14px;border-radius:3px;background:{EMOTION_COLORS_HEX[c]};"></span></td>
        </tr>''' for c in DISPLAY_CLASSES])}
      </tbody>
    </table>
  </div>

  <!-- Significant Temporal Transitions -->
  <div class="table-card">
    <div class="chart-title">Chronological Expression Transitions (Timeline)</div>
    <table>
      <thead>
        <tr>
          <th>Timestamp</th>
          <th>Elapsed Seconds</th>
          <th>Predicted Expression</th>
          <th>Confidence</th>
        </tr>
      </thead>
      <tbody>
        {"".join([f'''<tr>
          <td class="mono">{ev['timeLabel']}</td>
          <td class="mono">{ev['elapsedSeconds']:.1f}s</td>
          <td><strong>{EMOTION_EMOJIS.get(ev['emotion'], '')} {ev['emotion']}</strong></td>
          <td class="mono">{ev['confidence']*100:.1f}%</td>
        </tr>''' for ev in events]) if events else '<tr><td colspan="4" style="color:var(--muted);text-align:center;">No transitions recorded.</td></tr>'}
      </tbody>
    </table>
  </div>
</div>

<script>
// 1. Donut Chart
const ctxDonut = document.getElementById('donutChart').getContext('2d');
new Chart(ctxDonut, {{
  type: 'doughnut',
  data: {{
    labels: {labels_json},
    datasets: [{{
      data: {pcts_json},
      backgroundColor: {colors_json},
      borderWidth: 2,
      borderColor: '#141A23'
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'right', labels: {{ color: '#EDE7DA', font: {{ family: 'Plus Jakarta Sans' }} }} }},
      tooltip: {{ callbacks: {{ label: (ctx) => `${{ctx.label}}: ${{ctx.raw}}%` }} }}
    }}
  }}
}});

// 2. Bar Chart
const ctxBar = document.getElementById('barChart').getContext('2d');
new Chart(ctxBar, {{
  type: 'bar',
  data: {{
    labels: {labels_json},
    datasets: [{{
      label: 'Frame Count',
      data: {counts_json},
      backgroundColor: {colors_json},
      borderRadius: 6
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8A93A6' }}, grid: {{ display: false }} }},
      y: {{ ticks: {{ color: '#8A93A6' }}, grid: {{ color: '#232B39' }} }}
    }}
  }}
}});

// 3. Temporal Line Chart
const ctxTime = document.getElementById('timeChart').getContext('2d');
new Chart(ctxTime, {{
  type: 'line',
  data: {{
    labels: {time_labels_json},
    datasets: [{{
      label: 'Confidence (%)',
      data: {time_conf_json},
      borderColor: '#5FE39A',
      backgroundColor: 'rgba(95, 227, 154, 0.08)',
      fill: true,
      tension: 0.25,
      pointRadius: 2
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8A93A6', maxTicksLimit: 12 }}, grid: {{ color: '#232B39' }} }},
      y: {{ min: 0, max: 100, ticks: {{ color: '#8A93A6' }}, grid: {{ color: '#232B39' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"[SessionEvaluator] [HTML] Exported HTML Visual Dashboard to: {os.path.abspath(filepath)}")
        return filepath
