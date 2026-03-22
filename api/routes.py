from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, HTTPException

from simulator.modes import ALL_MODES
from simulator.signal_generator import SignalGenerator
from ai.analyzer import HolterAnalyzer

router = APIRouter()

# Loaded once at import time — same instance used by all requests
_analyzer = HolterAnalyzer()


@router.get("/modes")
def get_modes():
    return {"modes": list(ALL_MODES.keys())}


@router.get("/simulate/snapshot/{mode_name}")
def get_snapshot(mode_name: str):
    mode_name = mode_name.upper()
    if mode_name not in ALL_MODES:
        raise HTTPException(status_code=404, detail=f"Mode '{mode_name}' not found.")
    gen = SignalGenerator(mode_name=mode_name)
    reading = gen.get_live_reading()
    if _analyzer.ready:
        analysis = _analyzer.analyze_live_reading(reading)
        return {**reading, **analysis}
    return reading


@router.get("/simulate/report/{mode_name}")
def get_report(mode_name: str):
    mode_name = mode_name.upper().strip()
    valid_modes = ["NORMAL", "BRADYCARDIA", "TACHYCARDIA",
                   "HYPOTENSION", "HYPERTENSION", "MI"]
    if mode_name not in valid_modes:
        mode_name = "NORMAL"

    generator = SignalGenerator(mode_name)
    analyzer = HolterAnalyzer()

    bpm_values = []
    systolic_values = []
    diastolic_values = []
    alert_count = 0

    for _ in range(3600):
        reading = generator.get_live_reading()
        analysis = analyzer.analyze_live_reading(reading)
        bpm_values.append(reading["bpm"])
        systolic_values.append(reading["systolic_bp"])
        diastolic_values.append(reading["diastolic_bp"])
        if analysis.get("is_alert"):
            alert_count += 1

    return {
        "mode": mode_name,
        "duration_seconds": 86400,
        "average_bpm": round(sum(bpm_values) / len(bpm_values), 1),
        "min_bpm": round(min(bpm_values), 1),
        "max_bpm": round(max(bpm_values), 1),
        "average_systolic": round(sum(systolic_values) / len(systolic_values), 1),
        "average_diastolic": round(sum(diastolic_values) / len(diastolic_values), 1),
        "alert_count": alert_count * 24,
        "timestamp": datetime.utcnow().isoformat(),
        "note": "Based on 1-hour sample extrapolated to 24 hours",
    }
