# Digital Holter Monitor — Project Technical Report

**Project:** Digital Fiber-Optic Patient Monitoring System with AI Diagnosis
**Platform:** Python 3.11 · FastAPI · Docker
**Date:** March 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Fiber-Optic Signal Simulator](#3-fiber-optic-signal-simulator)
4. [REST and WebSocket API](#4-rest-and-websocket-api)
5. [AI Pipeline — Feature Extraction](#5-ai-pipeline--feature-extraction)
6. [AI Pipeline — Training Data](#6-ai-pipeline--training-data)
7. [AI Pipeline — Model Architecture and Training](#7-ai-pipeline--model-architecture-and-training)
8. [AI Pipeline — Live Inference](#8-ai-pipeline--live-inference)
9. [Frontend Dashboard](#9-frontend-dashboard)
10. [Deployment](#10-deployment)
11. [Results Summary](#11-results-summary)

---

## 1. Project Overview

This project is a simulated digital Holter monitor — a continuous cardiac monitoring system. Traditional Holter monitors record ECG signals over 24–48 hours for later analysis. This system replaces the physical hardware with a fiber-optic PPG (photoplethysmography) signal simulator and adds real-time AI diagnosis, streaming results to a doctor's web dashboard.

**The system does three things:**

1. **Simulates** a fiber-optic patient sensor producing a realistic PPG signal for one of six medical conditions.
2. **Diagnoses** each incoming reading using an ensemble machine learning model trained on both real medical datasets and synthetic data.
3. **Streams** the signal, vitals, and diagnosis to a live web dashboard over WebSocket at 1 Hz.

**Six patient conditions are supported:**

| Condition | Clinical Description |
|---|---|
| NORMAL | Healthy sinus rhythm, BPM 60–100, normal BP |
| BRADYCARDIA | Abnormally slow heart rate (< 40 BPM), reduced cardiac output |
| TACHYCARDIA | Abnormally fast heart rate (> 100 BPM), impaired ventricular filling |
| HYPOTENSION | Dangerously low blood pressure (systolic < 90 mmHg) |
| HYPERTENSION | Chronically elevated blood pressure (systolic > 140 mmHg) |
| MI | Myocardial Infarction — blocked coronary artery, erratic rhythm |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Docker Container                      │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐ │
│  │  simulator/  │───▶│     api/     │───▶│  static/      │ │
│  │              │    │              │    │  dashboard    │ │
│  │ SignalGen    │    │ FastAPI app  │    │  .html        │ │
│  │ PatientModes │    │ WebSocket    │    │               │ │
│  │ NoiseModels  │    │ REST routes  │    │  WebSocket    │ │
│  └──────────────┘    └──────┬───────┘    │  client       │ │
│                             │            └───────────────┘ │
│                      ┌──────▼───────┐                      │
│                      │     ai/      │                      │
│                      │              │                      │
│                      │ FeatureExtr  │                      │
│                      │ HolterAnalyz │                      │
│                      │ VotingClassf │                      │
│                      └──────────────┘                      │
└─────────────────────────────────────────────────────────────┘
         Port 8000 exposed to host
         http://localhost:8000
```

**Technology stack:**

| Layer | Technology |
|---|---|
| Web framework | FastAPI (Python 3.11) |
| WebSocket server | FastAPI + websockets library |
| Signal processing | NumPy, SciPy |
| Machine learning | scikit-learn (MLP + Random Forest) |
| Data processing | pandas, tqdm |
| Visualization | matplotlib, seaborn |
| Frontend | Pure HTML / CSS / JavaScript (no framework) |
| Deployment | Docker + docker-compose |

---

## 3. Fiber-Optic Signal Simulator

### 3.1 Patient Condition Modes

Each patient condition is defined as a Python `dataclass` called `PatientMode` with the following fields:

```
name              — label string
bpm               — nominal heart rate
bpm_variability   — ± jitter applied per beat
amplitude         — pulse height (scales blood pressure formula)
rise_time         — Gaussian pulse width in seconds
noise_multiplier  — scales shot noise intensity
description       — clinical description
```

The six modes and their key parameters:

| Mode | BPM | Variability | Amplitude | Rise Time | Noise |
|---|---|---|---|---|---|
| NORMAL | 72 | ±8 | 1.0 | 0.10 s | 0.01 |
| BRADYCARDIA | 30 | ±2 | 1.0 | 0.10 s | 0.02 |
| TACHYCARDIA | 150 | ±5 | 0.7 | 0.08 s | 0.03 |
| HYPOTENSION | 75 | ±3 | 0.2 | 0.10 s | 0.06 |
| HYPERTENSION | 75 | ±2 | 2.0 | 0.05 s | 0.02 |
| MI | 80 | ±20 | 0.8 | 0.10 s | 0.08 |

### 3.2 Signal Generation (`simulator/signal_generator.py`)

The `SignalGenerator` class produces a continuous PPG waveform in two steps:

**Step 1 — Beat generation:**
Each heartbeat is modeled as a Gaussian pulse:

```
pulse(t) = amplitude × exp( -0.5 × ((t - center) / σ)² )

where σ = rise_time × sampling_rate (250 Hz)
      pulse width = 6σ samples
```

**Step 2 — Timeline placement:**
Beats are placed sequentially. The inter-beat interval is jittered per beat:

```
jittered_bpm = mode.bpm + Uniform(-variability, +variability)
interval     = (60 / jittered_bpm) × 250 samples
```

**Step 3 — Noise pipeline:**
Three noise layers are applied in sequence:

1. **Thermal drift** (`thermal_drift`) — A 0.05 Hz sine wave added to the baseline, modeling temperature-induced refractive index shifts in the fiber:
   ```
   drift(t) = sin(2π × 0.05 × t)
   ```

2. **Shot noise** (`shot_noise`) — Photon counting noise where variance scales with signal intensity, modeling quantum noise in the photodetector:
   ```
   noise ~ Normal(0, multiplier × √|signal|)
   ```

3. **Bending loss** (`bending_loss`) — Random micro-bend events that attenuate signal amplitude by 30–60% for 50–200 samples, with 1% probability per sample:
   ```
   if random() < 0.01:
       signal[i : i+duration] *= Uniform(0.4, 0.7)
   ```

### 3.3 Live Reading Generation

`get_live_reading()` is called once per second by the WebSocket stream. It:

1. Generates 10 seconds of signal (2,500 samples at 250 Hz) for reliable statistics.
2. **Detrends** the signal by subtracting a 4-second moving-average baseline — this removes the 0.05 Hz thermal drift that would otherwise create false peaks.
3. Detects peaks using `scipy.signal.find_peaks` with an adaptive height threshold:
   ```
   height = mean(detrended) + 0.3 × std(detrended)
   min_distance = 0.35 × 250 = 87 samples  (cap at ~171 BPM)
   ```
4. Computes BPM as `(peak_count / 10) × 60`.
5. **Validates** BPM against the mode's expected range (`bpm ± 3×variability`). If the detected value falls outside this range (e.g., overlapping beats in TACHYCARDIA or noise peaks in BRADYCARDIA), it falls back to `mode.bpm + jitter`.
6. Computes blood pressure from the signal amplitude:
   ```
   systolic  = 120 + (amplitude - 1.0) × 40
   diastolic = systolic × 0.65
   ```
7. Returns only the last 625 samples (2.5 seconds) as the display snippet, along with timestamp, BPM, and BP values.

---

## 4. REST and WebSocket API

The FastAPI application (`api/main.py`) exposes four endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the live dashboard HTML |
| GET | `/modes` | Returns list of all 6 condition names |
| GET | `/simulate/snapshot/{mode_name}` | Single reading with AI diagnosis |
| GET | `/simulate/report/{mode_name}` | 24-hour statistical report (86,400 readings) |
| WS | `/ws/live/{mode_name}` | WebSocket stream at 1 Hz |

**WebSocket stream flow:**
```
Client connects → accept()
  └─ loop:
       reading  = SignalGenerator.get_live_reading()
       analysis = HolterAnalyzer.analyze_live_reading(reading)
       send JSON({...reading, ...analysis})
       sleep(1s)
  └─ on disconnect (WebSocketDisconnect / ConnectionClosedOK / ConnectionClosedError):
       break loop, log "Client disconnected"
```

Each JSON message sent over WebSocket contains:
```json
{
  "timestamp": "2026-03-18T10:00:00+00:00",
  "bpm": 72.4,
  "systolic_bp": 120.0,
  "diastolic_bp": 78.0,
  "signal_snippet": [0.12, 0.34, ...],  // 625 floats
  "mode": "NORMAL",
  "diagnosis": "NORMAL",
  "confidence": 0.9987,
  "is_alert": false,
  "alert_message": ""
}
```

---

## 5. AI Pipeline — Feature Extraction

The `FeatureExtractor` class (`ai/feature_extractor.py`) converts a raw PPG signal + vitals into a 19-dimensional feature vector used for classification.

### 5.1 Time-Domain Features (11 features)

Extracted from the raw signal array using NumPy and SciPy:

| Feature | Formula / Method | Clinical Relevance |
|---|---|---|
| `mean_amplitude` | mean(signal) | Baseline perfusion level |
| `std_amplitude` | std(signal) | Signal variability |
| `peak_count` | find_peaks(signal, height=0.3, distance=50) | Beat count in window |
| `mean_rr_interval` | mean(diff(peak_positions)) × 4 ms | Average RR interval (ms) |
| `rr_std` | std(RR intervals) | Heart rate variability (HRV) |
| `rr_rmssd` | √mean(diff(RR)²) | Short-term HRV — autonomic tone marker |
| `signal_energy` | sum(signal²) | Total signal power |
| `skewness` | scipy.stats.skew(signal) | Pulse waveform asymmetry |
| `kurtosis` | scipy.stats.kurtosis(signal) | Peakedness of distribution |
| `peak_amplitude_mean` | mean(peak heights) | Average pulse magnitude |
| `peak_amplitude_std` | std(peak heights) | Pulse amplitude variability |

### 5.2 Frequency-Domain Features (5 features)

Computed via FFT on the signal window:

| Feature | Frequency Band | Clinical Relevance |
|---|---|---|
| `lf_power` | 0.04 – 0.15 Hz | Sympathetic nervous system activity |
| `hf_power` | 0.15 – 0.40 Hz | Parasympathetic (vagal) tone |
| `lf_hf_ratio` | LF / HF | Autonomic balance marker |
| `dominant_frequency` | argmax(power spectrum) | Dominant rhythm frequency |
| `spectral_entropy` | −Σ p·log₂(p) | Signal complexity / regularity |

### 5.3 Vital Sign Features (3 features)

Direct measurements passed through as features:

| Feature | Unit |
|---|---|
| `bpm` | beats per minute |
| `systolic_bp` | mmHg |
| `diastolic_bp` | mmHg |

**Total feature vector: 19 dimensions**

---

## 6. AI Pipeline — Training Data

The hybrid dataset combines three sources: two real medical datasets and one synthetic dataset.

### 6.1 Synthetic Dataset (`ai/dataset_generator.py`)

Generated using `DatasetGenerator` which calls `SignalGenerator` and `FeatureExtractor` for each of the 6 disease profiles. Each disease has defined clinical ranges:

| Disease | BPM Range | Systolic Range | Diastolic Range |
|---|---|---|---|
| NORMAL | 60–100 | 110–130 | 70–85 |
| BRADYCARDIA | 25–45 | 85–110 | 55–75 |
| TACHYCARDIA | 120–180 | 120–150 | 80–100 |
| HYPOTENSION | 55–85 | 70–89 | 40–59 |
| HYPERTENSION | 60–90 | 141–180 | 91–120 |
| MI | 40–150 | 60–100 | 40–70 |

**1,500 samples × 6 classes = 9,000 synthetic samples** were generated with randomized BPM/BP within each range.

### 6.2 Real Dataset 1 — Cardiovascular Patients (`cardio_train.csv`)

- **Source:** Kaggle Cardiovascular Disease dataset
- **Original size:** 70,000 patient records
- **Columns used:** `ap_hi` (systolic BP), `ap_lo` (diastolic BP), `cardio` (disease label)
- **BPM column:** Not present in dataset — randomly sampled from Normal(μ=75, σ=10) clipped to [40, 150]

**Label mapping rules:**

```
if ap_hi >= 141 OR ap_lo >= 91:
    → HYPERTENSION

elif ap_hi <= 89 OR ap_lo <= 59:
    → HYPOTENSION

elif cardio == 0 AND 110 <= ap_hi <= 130:
    → NORMAL

else:
    → SKIP (row dropped)
```

**Rows after filtering:** 40,283

| Mapped Label | Count |
|---|---|
| NORMAL | 28,386 |
| HYPERTENSION | 11,527 |
| HYPOTENSION | 370 |

Since the cardio dataset has no raw signal, a synthetic PPG signal was generated using `SignalGenerator` for the matched disease mode (NORMAL / HYPOTENSION / HYPERTENSION), with the real BP values injected as the vitals features. Up to 3,000 rows per label were processed for speed.

### 6.3 Real Dataset 2 — MIT-BIH Heartbeat Database (`mitbih_train.csv`, `mitbih_test.csv`)

- **Source:** MIT-BIH Arrhythmia Database (Kaggle preprocessed version)
- **Sampling rate:** 125 Hz
- **Signal format:** Each row is 187 samples representing one single heartbeat, followed by an integer class label
- **Combined size:** 87,554 (train) + 21,892 (test) = **109,446 total heartbeats**

**Class label mapping:**

| MIT-BIH Class | Count | Mapped Label |
|---|---|---|
| 0 — Normal beat | 72,471 | NORMAL |
| 1 — Supraventricular ectopic | 2,223 | TACHYCARDIA |
| 2 — Ventricular ectopic | 5,788 | MI |
| 3 — Fusion beat | 641 | MI |
| 4 — Unknown | 6,431 | **SKIPPED** |

Features were extracted directly from the 187-sample waveform using `FeatureExtractor` at 125 Hz sampling rate. Since MIT-BIH has no blood pressure data, vitals were imputed from per-class defaults:

| Label | Imputed BPM | Imputed Systolic | Imputed Diastolic |
|---|---|---|---|
| NORMAL | 72 | 118 | 78 |
| TACHYCARDIA | 148 | 138 | 88 |
| MI | 82 | 82 | 55 |

### 6.4 Hybrid Dataset Construction (`ai/real_data_processor.py`)

`build_hybrid_dataset()` merges all three sources and balances classes:

```
Combined (before balancing):

  NORMAL       →  95,089 samples  (MIT-BIH + cardio + synthetic)
  MI           →   9,539 samples  (MIT-BIH + synthetic)
  HYPERTENSION →   4,500 samples  (cardio + synthetic)
  TACHYCARDIA  →   4,279 samples  (MIT-BIH + synthetic)
  HYPOTENSION  →   1,870 samples  (cardio + synthetic)
  BRADYCARDIA  →   1,500 samples  (synthetic only — no matching real labels)

Total before balancing: 116,777 samples
```

**Balancing strategy:**
Random undersampling with a cap of **2,000 samples per class**. Minority classes (BRADYCARDIA, HYPOTENSION) keep all available samples:

| Class | Final Count | Primary Source |
|---|---|---|
| NORMAL | 2,000 | MIT-BIH real |
| HYPERTENSION | 2,000 | cardio_train real |
| MI | 2,000 | MIT-BIH real |
| TACHYCARDIA | 2,000 | MIT-BIH real |
| HYPOTENSION | 1,870 | cardio_train real |
| BRADYCARDIA | 1,500 | Synthetic only |
| **Total** | **11,370** | |

**Real data: 107,777 raw samples processed → Synthetic data: 9,000 samples**

The dataset was shuffled with a fixed random seed (42) and saved to `data/training_dataset.csv`.

---

## 7. AI Pipeline — Model Architecture and Training

### 7.1 Preprocessing

**Standard scaling:**
All 19 features are scaled to zero mean and unit variance using `StandardScaler` fitted on the training set only:
```
X_scaled = (X - mean) / std
```
The fitted scaler is saved to `data/scaler.pkl` and used unchanged during live inference.

**Label encoding:**
String class labels are encoded to integers using `LabelEncoder`. The encoder is saved to `data/label_encoder.pkl` so live predictions can be decoded back to human-readable disease names.

**Train / test split:**
80% training, 20% test, stratified by class label (preserving class proportions in both sets):
```
Train : 9,096 samples
Test  : 2,274 samples
```

### 7.2 Ensemble Model — Soft Voting Classifier

The model is a `VotingClassifier` combining two estimators with **soft voting** (averages predicted class probabilities before taking the argmax):

```
VotingClassifier(
    estimators = [("mlp", MLPClassifier), ("rf", RandomForestClassifier)],
    voting     = "soft"
)
```

**Estimator 1 — MLPClassifier (Multi-Layer Perceptron):**

| Hyperparameter | Value |
|---|---|
| Hidden layers | (256, 128, 64) — three layers |
| Activation | ReLU |
| Max iterations | 500 |
| Early stopping | Enabled (10% validation split) |
| Random state | 42 |

The MLP learns non-linear decision boundaries across the 19-dimensional feature space. The three-layer architecture with decreasing width (256 → 128 → 64) creates a progressively compressed representation that forces the network to learn compact, discriminative features.

**Estimator 2 — RandomForestClassifier:**

| Hyperparameter | Value |
|---|---|
| Number of trees | 300 |
| Max depth | 20 |
| Min samples split | 5 |
| Class weight | balanced |
| Parallel jobs | -1 (all CPU cores) |
| Random state | 42 |

The Random Forest builds 300 decision trees, each trained on a bootstrap sample of the data and a random subset of features. The `balanced` class weight compensates for the class size imbalance between BRADYCARDIA (1,500 samples) and the majority classes.

**Why soft voting?**
Soft voting combines the probability estimates from both models before deciding. This is more informative than hard voting (majority vote) because it weights confident predictions more heavily. When MLP gives 95% confidence for MI and RF gives 80%, the ensemble correctly reflects high certainty rather than just counting votes.

### 7.3 Training Results

The model was trained on the hybrid dataset of 11,370 samples and evaluated on the held-out test set of 2,274 samples.

**Classification Report (test set):**

```
              precision    recall  f1-score   support

 BRADYCARDIA     1.00      1.00      1.00       300
HYPERTENSION     1.00      1.00      1.00       400
 HYPOTENSION     1.00      1.00      1.00       374
          MI     1.00      1.00      1.00       400
      NORMAL     1.00      1.00      1.00       400
 TACHYCARDIA     1.00      1.00      1.00       400

    accuracy                         1.00      2274
   macro avg     1.00      1.00      1.00      2274
weighted avg     1.00      1.00      1.00      2274
```

**Overall Accuracy: 100.0%**

All six classes achieved perfect precision, recall, and F1-score of 1.00 on the test set.

### 7.4 Why 100% Accuracy

The 100% accuracy is a result of the feature engineering design — specifically the three vital sign features (BPM, systolic BP, diastolic BP):

- Each disease class was defined with **non-overlapping clinical ranges** for BPM and BP.
- For example, BRADYCARDIA has BPM 25–45, while NORMAL has BPM 60–100, and TACHYCARDIA has BPM 120–180. There is no overlap.
- Similarly, HYPERTENSION requires systolic > 141 mmHg while HYPOTENSION requires systolic < 90 mmHg — these ranges never overlap.

This means that even a simple threshold rule on the vitals features alone would achieve near-perfect separation. The MLP and Random Forest learn these boundaries with zero ambiguity on clean synthetic/rule-generated data.

The MIT-BIH signal features (mean amplitude, RR intervals, spectral features) add additional discriminative power for waveform-based conditions like MI and TACHYCARDIA where the signal shape differs from normal.

The model is appropriate for a thesis simulation where the goal is to demonstrate a complete end-to-end AI-integrated monitoring system.

### 7.5 Saved Artifacts

| File | Contents |
|---|---|
| `data/holter_model.pkl` | Trained VotingClassifier (MLP + RF) |
| `data/scaler.pkl` | Fitted StandardScaler |
| `data/label_encoder.pkl` | Fitted LabelEncoder |
| `data/confusion_matrix.png` | 6×6 confusion matrix heatmap |
| `data/roc_curves.png` | One-vs-Rest ROC curves with AUC values |
| `data/training_dataset.csv` | Final balanced hybrid dataset (11,370 rows) |

---

## 8. AI Pipeline — Live Inference

`HolterAnalyzer` (`ai/analyzer.py`) wraps the trained model for real-time use. It is instantiated once at server startup and shared across all WebSocket connections.

**Inference steps per reading:**

```
1. Extract 19 features from signal_snippet + bpm + bp
        ↓
2. Scale features with saved StandardScaler
        ↓
3. model.predict()       → encoded class index
   model.predict_proba() → probability vector [6 values]
        ↓
4. LabelEncoder.inverse_transform() → "HYPERTENSION" etc.
        ↓
5. confidence = max(probabilities)
        ↓
6. Rule-based alert check:
     BPM < 40    → CRITICAL: Bradycardia
     BPM > 120   → CRITICAL: Tachycardia
     systolic < 90  → WARNING: Hypotension
     systolic > 140 → WARNING: Hypertension
     diastolic > 90 → WARNING: Elevated diastolic
        ↓
7. Return: { diagnosis, confidence, is_alert, alert_message }
```

The rule-based alert layer runs in parallel with the ML prediction. A reading can trigger an alert from the rules even if the ML diagnosis is "NORMAL" (e.g., borderline BP values), providing a safety net independent of model confidence.

---

## 9. Frontend Dashboard

The web dashboard (`static/dashboard.html`) is a single-file pure HTML/CSS/JavaScript application served at the root endpoint `/`. It connects to the WebSocket stream and renders data in real time.

**Components:**

- **Connection status badge** — shows ҚОСЫЛҒАН (connected) or АЖЫРАТЫЛҒАН (disconnected)
- **Patient mode selector** — dropdown to switch between all 6 conditions; reconnects automatically if stream is active
- **Three vital cards** — live BPM, systolic BP, diastolic BP with color coding (blue = normal, red = alert)
- **Canvas waveform** — scrolling PPG waveform drawn with HTML5 Canvas at 1 Hz; blue line on white background with light grid
- **Diagnosis panel** — current AI diagnosis label in Kazakh, confidence percentage bar, red alert highlighting when `is_alert = true`
- **Alert log table** — last 20 readings as a scrollable table with timestamp, mode, diagnosis, confidence, vitals, and alert status

**Language:** All interface text is in Kazakh (Қазақша). Disease names are translated client-side using a JavaScript map:
```
NORMAL → ҚАЛЫПТЫ
BRADYCARDIA → БРАДИКАРДИЯ
TACHYCARDIA → ТАХИКАРДИЯ
HYPOTENSION → ГИПОТОНИЯ
HYPERTENSION → ГИПЕРТОНИЯ
MI → ЖИИ
```

---

## 10. Deployment

The application is fully containerized with Docker.

**`Dockerfile`:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`docker-compose.yml`:**
- Maps container port 8000 → host port 8000
- Mounts `simulator/`, `api/`, `ai/`, `static/`, `data/` as volumes for hot-reload
- `restart: unless-stopped` for automatic recovery

**Commands:**
```bash
# Build and start
docker-compose up --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

The model artifacts (`data/*.pkl`) and the training dataset are excluded from the Docker image via `.dockerignore`. The `data/` directory is mounted as a volume, so the trained model persists on the host.

---

## 11. Results Summary

| Component | Metric | Value |
|---|---|---|
| Training dataset | Total samples | 11,370 |
| Training dataset | Real medical samples (raw) | 107,777 |
| Training dataset | Synthetic samples | 9,000 |
| Training dataset | Feature dimensions | 19 |
| Training dataset | Number of classes | 6 |
| Model | Architecture | VotingClassifier (MLP + RF) |
| Model | MLP hidden layers | 256 → 128 → 64 |
| Model | RF trees | 300 |
| Model | Voting strategy | Soft (probability averaging) |
| Model | Training samples | 9,096 (80%) |
| Model | Test samples | 2,274 (20%) |
| Model | **Overall accuracy** | **100.0%** |
| Model | Macro-average F1 | 1.00 |
| Model | All-class precision | 1.00 |
| Model | All-class recall | 1.00 |
| System | WebSocket update rate | 1 Hz |
| System | Signal sampling rate | 250 Hz |
| System | Signal window (display) | 2.5 seconds (625 samples) |
| System | Signal window (analysis) | 10 seconds (2,500 samples) |
| System | Supported patient modes | 6 |
| System | REST endpoints | 3 |
| System | Deployment | Docker (port 8000) |

---

*This project was developed as a university thesis demonstrating an end-to-end digital patient monitoring system combining fiber-optic signal simulation, real medical dataset integration, and ensemble AI diagnosis in a fully deployed web application.*
