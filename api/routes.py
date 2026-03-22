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
    from api.simulator_state import simulator_state

    history = list(simulator_state.history)

    if len(history) == 0:
        return {
            "error": "No data yet. Connect to monitoring first.",
            "mode": "NONE",
            "duration_seconds": 0,
            "average_bpm": 0,
            "min_bpm": 0,
            "max_bpm": 0,
            "average_systolic": 0,
            "average_diastolic": 0,
            "alert_count": 0,
            "total_readings": 0,
            "timestamp": datetime.utcnow().isoformat(),
        }

    bpm_values = [r["bpm"] for r in history]
    systolic_values = [r["systolic_bp"] for r in history]
    diastolic_values = [r["diastolic_bp"] for r in history]
    alert_count = sum(1 for r in history if r.get("is_alert"))

    modes_used = {}
    for r in history:
        m = r.get("mode", "UNKNOWN")
        modes_used[m] = modes_used.get(m, 0) + 1

    dominant_mode = max(modes_used, key=modes_used.get)
    duration_minutes = len(history) // 60

    return {
        "mode": dominant_mode,
        "modes_breakdown": modes_used,
        "duration_seconds": len(history),
        "duration_minutes": duration_minutes,
        "total_readings": len(history),
        "average_bpm": round(sum(bpm_values) / len(bpm_values), 1),
        "min_bpm": round(min(bpm_values), 1),
        "max_bpm": round(max(bpm_values), 1),
        "average_systolic": round(sum(systolic_values) / len(systolic_values), 1),
        "average_diastolic": round(sum(diastolic_values) / len(diastolic_values), 1),
        "alert_count": alert_count,
        "first_reading_time": history[0]["timestamp"],
        "last_reading_time": history[-1]["timestamp"],
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/simulate/history")
def get_history():
    from api.simulator_state import simulator_state

    history = list(simulator_state.history)
    last_60 = history[-60:] if len(history) >= 60 else history
    return {
        "readings": last_60,
        "total_stored": len(history),
    }
