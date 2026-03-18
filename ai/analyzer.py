import pickle
from datetime import datetime, timezone

import numpy as np

from ai.feature_extractor import FeatureExtractor

ALERT_THRESHOLDS = {
    "BPM_LOW": 40,
    "BPM_HIGH": 120,
    "SYSTOLIC_LOW": 90,
    "SYSTOLIC_HIGH": 140,
    "DIASTOLIC_HIGH": 90,
}


class HolterAnalyzer:

    def __init__(self):
        self.ready = False
        self.extractor = FeatureExtractor()

        try:
            with open("data/holter_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open("data/scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            with open("data/label_encoder.pkl", "rb") as f:
                self.label_encoder = pickle.load(f)
            self.ready = True
            print("AI Analyzer ready")
        except FileNotFoundError:
            print("AI model not found, run training first")

    def analyze_live_reading(self, reading: dict) -> dict:
        signal = np.array(reading["signal_snippet"])
        bpm = reading["bpm"]
        systolic = reading["systolic_bp"]
        diastolic = reading["diastolic_bp"]

        # Extract and scale features
        features = self.extractor.extract_all(signal, bpm, systolic, diastolic)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Predict
        encoded_pred = self.model.predict(features_scaled)[0]
        proba = self.model.predict_proba(features_scaled)[0]
        confidence = float(np.max(proba))
        diagnosis = self.label_encoder.inverse_transform([encoded_pred])[0]

        # Build alerts
        alerts = []
        if bpm < ALERT_THRESHOLDS["BPM_LOW"]:
            alerts.append(f"CRITICAL: Heart rate {bpm:.0f} BPM - Bradycardia detected")
        if bpm > ALERT_THRESHOLDS["BPM_HIGH"]:
            alerts.append(f"CRITICAL: Heart rate {bpm:.0f} BPM - Tachycardia detected")
        if systolic < ALERT_THRESHOLDS["SYSTOLIC_LOW"]:
            alerts.append(f"WARNING: Systolic BP {systolic:.0f} mmHg - Hypotension detected")
        if systolic > ALERT_THRESHOLDS["SYSTOLIC_HIGH"]:
            alerts.append(f"WARNING: Systolic BP {systolic:.0f} mmHg - Hypertension detected")
        if diastolic > ALERT_THRESHOLDS["DIASTOLIC_HIGH"]:
            alerts.append(f"WARNING: Diastolic BP {diastolic:.0f} mmHg - Elevated diastolic pressure")

        is_alert = len(alerts) > 0
        alert_message = " | ".join(alerts) if alerts else ""

        return {
            "diagnosis": diagnosis,
            "confidence": round(confidence, 4),
            "is_alert": is_alert,
            "alert_message": alert_message,
            "bpm": bpm,
            "systolic_bp": systolic,
            "diastolic_bp": diastolic,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
