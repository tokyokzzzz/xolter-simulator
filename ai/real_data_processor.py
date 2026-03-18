"""
Real-data processor for the Holter Monitor project.

Combines three data sources into one hybrid training dataset:
  1. cardio_train.csv  — 70 000 cardiovascular patients (BP, demographics)
  2. mitbih_train/test — MIT-BIH annotated heartbeat waveforms (187 samples each)
  3. data/training_dataset.csv — existing synthetic data from DatasetGenerator

Saves the balanced result back to data/training_dataset.csv.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis as sp_kurtosis
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from tqdm import tqdm

from simulator.signal_generator import SignalGenerator
from ai.feature_extractor import FeatureExtractor
from ai.dataset_generator import FEATURE_NAMES, DISEASE_PROFILES

# ── File paths ────────────────────────────────────────────────────────────────
CARDIO_PATH        = "cardio_train.csv"
MITBIH_TRAIN_PATH  = "mitbih_train.csv"
MITBIH_TEST_PATH   = "mitbih_test.csv"
SYNTHETIC_PATH     = "data/training_dataset.csv"
OUTPUT_PATH        = "data/training_dataset.csv"

MAX_SAMPLES_PER_CLASS = 2000

# MIT-BIH integer → our label (class 4 = Unknown → skipped)
MITBIH_LABEL_MAP = {
    0: "NORMAL",
    1: "TACHYCARDIA",
    2: "MI",
    3: "MI",
}

# Default vitals injected for MIT-BIH rows (no BP in that dataset)
LABEL_VITALS = {
    "NORMAL":       {"bpm": 72.0,  "systolic": 118.0, "diastolic": 78.0},
    "BRADYCARDIA":  {"bpm": 45.0,  "systolic":  95.0, "diastolic": 62.0},
    "TACHYCARDIA":  {"bpm": 148.0, "systolic": 138.0, "diastolic": 88.0},
    "HYPOTENSION":  {"bpm": 68.0,  "systolic":  82.0, "diastolic": 52.0},
    "HYPERTENSION": {"bpm": 74.0,  "systolic": 158.0, "diastolic": 98.0},
    "MI":           {"bpm": 82.0,  "systolic":  82.0, "diastolic": 55.0},
}

# Cardio label → SignalGenerator mode name
CARDIO_MODE_MAP = {
    "NORMAL":       "NORMAL",
    "HYPOTENSION":  "HYPOTENSION",
    "HYPERTENSION": "HYPERTENSION",
}

_fe = FeatureExtractor()


def _build_feature_row(signal: np.ndarray, bpm: float,
                       systolic: float, diastolic: float,
                       sampling_rate: int = 250) -> list:
    """Extract the 19 features in FEATURE_NAMES order and return as a plain list."""
    td = _fe.extract_time_domain(signal)
    fd = _fe.extract_frequency_domain(signal, sampling_rate=sampling_rate)
    vitals = [bpm, systolic, diastolic]
    return list(td.values()) + list(fd.values()) + vitals


# ── Function 1 ────────────────────────────────────────────────────────────────

def process_cardio_data(filepath: str = CARDIO_PATH) -> pd.DataFrame:
    """
    Reads cardio_train.csv and maps rows to NORMAL / HYPOTENSION / HYPERTENSION.
    A synthetic 10-second PPG signal is generated for each row (matched to the
    disease mode) so that all 19 features can be extracted.  Real BP values are
    injected as the vitals features.
    """
    print("Loading cardio data...")
    df = pd.read_csv(filepath, sep=";")
    print(f"  Raw rows: {len(df)}")

    # Assign random BPM (population-level estimate)
    rng = np.random.default_rng(seed=0)
    df["bpm"] = rng.normal(loc=75.0, scale=10.0, size=len(df)).clip(40, 150)

    # Label mapping
    conditions = [
        (df["ap_hi"] >= 141) | (df["ap_lo"] >= 91),
        (df["ap_hi"] <= 89)  | (df["ap_lo"] <= 59),
        (df["cardio"] == 0)  & (df["ap_hi"] >= 110) & (df["ap_hi"] <= 130),
    ]
    labels = ["HYPERTENSION", "HYPOTENSION", "NORMAL"]

    df["label"] = np.select(conditions, labels, default="SKIP")
    df = df[df["label"] != "SKIP"].reset_index(drop=True)

    print(f"  After filtering: {len(df)} rows")
    print("  Label counts from cardio:")
    print(df["label"].value_counts().to_string(header=False))

    # Limit to 3000 per label before signal generation (speed)
    sampled_parts = []
    for lbl, grp in df.groupby("label"):
        sampled_parts.append(grp.sample(n=min(len(grp), 3000), random_state=42))
    df = pd.concat(sampled_parts, ignore_index=True)

    rows = []
    generators = {label: SignalGenerator(mode_name=CARDIO_MODE_MAP[label])
                  for label in CARDIO_MODE_MAP}

    for _, rec in tqdm(df.iterrows(), total=len(df), desc="cardio signals"):
        label     = rec["label"]
        systolic  = float(rec["ap_hi"])
        diastolic = float(rec["ap_lo"])
        bpm       = float(rec["bpm"])
        signal    = generators[label].generate_seconds(seconds=10)
        feats     = _build_feature_row(signal, bpm, systolic, diastolic, sampling_rate=250)
        rows.append(feats + [label])

    result = pd.DataFrame(rows, columns=FEATURE_NAMES + ["label"])
    print(f"  Cardio features extracted: {len(result)} rows")
    return result


# ── Function 2 ────────────────────────────────────────────────────────────────

def process_mitbih_data(train_path: str = MITBIH_TRAIN_PATH,
                        test_path: str  = MITBIH_TEST_PATH) -> pd.DataFrame:
    """
    Reads both MIT-BIH CSV files. Each row is 187 signal samples (one heartbeat)
    followed by an integer class label (0–4).  Class 4 is discarded.
    Features are extracted directly from the 187-sample waveform at 125 Hz.
    Vitals are imputed from per-class defaults (LABEL_VITALS).
    """
    print("Loading MIT-BIH data...")
    train = pd.read_csv(train_path, header=None)
    test  = pd.read_csv(test_path,  header=None)
    combined = pd.concat([train, test], ignore_index=True)
    print(f"  Total MIT-BIH rows: {len(combined)}")

    # Column 187 is the class label
    signal_cols = list(range(187))
    label_col   = 187

    # Drop class 4 (Unknown)
    combined = combined[combined[label_col] != 4.0].reset_index(drop=True)
    combined["label"] = combined[label_col].map(MITBIH_LABEL_MAP)

    print("  MIT-BIH class distribution (after mapping):")
    print(combined["label"].value_counts().to_string(header=False))

    rows = []
    for _, rec in tqdm(combined.iterrows(), total=len(combined), desc="mitbih signals"):
        label  = rec["label"]
        signal = rec[signal_cols].values.astype(float)
        vitals = LABEL_VITALS[label]
        feats  = _build_feature_row(
            signal,
            bpm       = vitals["bpm"],
            systolic  = vitals["systolic"],
            diastolic = vitals["diastolic"],
            sampling_rate = 125,
        )
        rows.append(feats + [label])

    result = pd.DataFrame(rows, columns=FEATURE_NAMES + ["label"])
    print(f"  MIT-BIH features extracted: {len(result)} rows")
    return result


# ── Function 3 ────────────────────────────────────────────────────────────────

def build_hybrid_dataset() -> pd.DataFrame:
    """
    Merges real (cardio + MIT-BIH) and synthetic data, balances each class to
    at most MAX_SAMPLES_PER_CLASS rows (minority classes keep all samples),
    shuffles, and saves to data/training_dataset.csv.
    """
    # ── Collect sources ───────────────────────────────────────────────────────
    cardio_df  = process_cardio_data()
    mitbih_df  = process_mitbih_data()

    import os
    if not os.path.exists(SYNTHETIC_PATH):
        print(f"\n{SYNTHETIC_PATH} not found — generating synthetic data first...")
        from ai.dataset_generator import DatasetGenerator
        DatasetGenerator().generate_dataset(samples_per_class=1500)

    print(f"\nLoading synthetic data from {SYNTHETIC_PATH}...")
    synthetic_df = pd.read_csv(SYNTHETIC_PATH)
    print(f"  Synthetic rows: {len(synthetic_df)}")

    real_count      = len(cardio_df) + len(mitbih_df)
    synthetic_count = len(synthetic_df)

    # ── Combine ───────────────────────────────────────────────────────────────
    combined = pd.concat([cardio_df, mitbih_df, synthetic_df], ignore_index=True)
    print(f"\nCombined (before balancing): {len(combined)} rows")
    print("Class counts before balancing:")
    print(combined["label"].value_counts().to_string(header=False))

    # ── Balance (undersample majority, keep all minority) ─────────────────────
    balanced_parts = []
    for label, group in combined.groupby("label"):
        n = min(len(group), MAX_SAMPLES_PER_CLASS)
        balanced_parts.append(group.sample(n=n, random_state=42))

    balanced = pd.concat(balanced_parts, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    balanced.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("HYBRID DATASET SAVED")
    print("=" * 60)
    print(f"Output file    : {OUTPUT_PATH}")
    print(f"Total samples  : {len(balanced)}")
    print(f"Features       : {len(FEATURE_NAMES)}")
    print(f"Real data      : {real_count} samples  |  Synthetic data: {synthetic_count} samples")
    print("\nFinal class distribution:")
    print(balanced["label"].value_counts().to_string(header=False))
    print("=" * 60)

    return balanced


if __name__ == "__main__":
    build_hybrid_dataset()
