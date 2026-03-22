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
    mode_name = mode_name.upper()
    if mode_name not in ALL_MODES:
        raise HTTPException(status_code=404, detail=f"Mode '{mode_name}' not found.")

    mode = ALL_MODES[mode_name]
    gen = SignalGenerator(mode_name=mode_name)

    # Sample 1 hour (3600 readings) and extrapolate to 24 hours.
    sample_size = 3600
    bpm_values = []
    systolic_values = []
    diastolic_values = []
    alert_count_per_hour = 0

    for _ in range(sample_size):
        reading = gen.get_live_reading()
        bpm = reading["bpm"]
        sys_bp = reading["systolic_bp"]
        dia_bp = reading["diastolic_bp"]

        bpm_values.append(bpm)
        systolic_values.append(sys_bp)
        diastolic_values.append(dia_bp)

        if bpm < 40 or bpm > 120 or sys_bp < 90 or sys_bp > 140:
            alert_count_per_hour += 1

    bpm_arr = np.array(bpm_values)
    sys_arr = np.array(systolic_values)
    dia_arr = np.array(diastolic_values)

    alert_count = alert_count_per_hour * 24

    return {
        "mode": mode_name,
        "description": mode.description,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_hours": 24,
        "average_bpm": round(float(bpm_arr.mean()), 1),
        "min_bpm": round(float(bpm_arr.min()), 1),
        "max_bpm": round(float(bpm_arr.max()), 1),
        "average_systolic": round(float(sys_arr.mean()), 1),
        "average_diastolic": round(float(dia_arr.mean()), 1),
        "alert_count": alert_count,
        "alert_percentage": round(alert_count / 86400 * 100, 2),
        "note": "Statistics based on 1-hour sample, extrapolated to 24 hours",
    }
