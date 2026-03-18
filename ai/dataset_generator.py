import numpy as np
import pandas as pd
from tqdm import tqdm

from simulator.signal_generator import SignalGenerator
from ai.feature_extractor import FeatureExtractor

FEATURE_NAMES = [
    # Time domain (11)
    "mean_amplitude", "std_amplitude", "peak_count",
    "mean_rr_interval", "rr_std", "rr_rmssd",
    "signal_energy", "skewness", "kurtosis",
    "peak_amplitude_mean", "peak_amplitude_std",
    # Frequency domain (5)
    "lf_power", "hf_power", "lf_hf_ratio",
    "dominant_frequency", "spectral_entropy",
    # Vitals (3)
    "bpm", "systolic_bp", "diastolic_bp",
]

DISEASE_PROFILES = {
    "NORMAL": {
        "mode": "NORMAL",
        "bpm_range": (60, 100),
        "systolic_range": (110, 130),
        "diastolic_range": (70, 85),
    },
    "BRADYCARDIA": {
        "mode": "BRADYCARDIA",
        "bpm_range": (25, 45),
        "systolic_range": (85, 110),
        "diastolic_range": (55, 75),
    },
    "TACHYCARDIA": {
        "mode": "TACHYCARDIA",
        "bpm_range": (120, 180),
        "systolic_range": (120, 150),
        "diastolic_range": (80, 100),
    },
    "HYPOTENSION": {
        "mode": "HYPOTENSION",
        "bpm_range": (55, 85),
        "systolic_range": (70, 89),
        "diastolic_range": (40, 59),
    },
    "HYPERTENSION": {
        "mode": "HYPERTENSION",
        "bpm_range": (60, 90),
        "systolic_range": (141, 180),
        "diastolic_range": (91, 120),
    },
    "MI": {
        "mode": "MI",
        "bpm_range": (40, 150),
        "systolic_range": (60, 100),
        "diastolic_range": (40, 70),
    },
}


class DatasetGenerator:

    def __init__(self):
        self.extractor = FeatureExtractor()

    def generate_sample(self, disease_name: str) -> tuple:
        """
        Generates one labeled sample: a 10-second signal for the given disease,
        with BPM and blood pressure randomly drawn from the disease's clinical ranges.
        Returns (feature_vector: np.ndarray, label: str).
        """
        profile = DISEASE_PROFILES[disease_name]

        bpm = float(np.random.uniform(*profile["bpm_range"]))
        systolic = float(np.random.uniform(*profile["systolic_range"]))
        diastolic = float(np.random.uniform(*profile["diastolic_range"]))

        gen = SignalGenerator(mode_name=profile["mode"])
        signal = gen.generate_seconds(seconds=10)

        features = self.extractor.extract_all(signal, bpm, systolic, diastolic)
        return features, disease_name

    def generate_dataset(self, samples_per_class: int = 1500) -> pd.DataFrame:
        """
        Generates a balanced dataset across all 6 disease classes and saves it to CSV.
        Total samples = samples_per_class × 6 classes.
        """
        all_rows = []

        for disease_name in DISEASE_PROFILES:
            for _ in tqdm(range(samples_per_class), desc=f"{disease_name:12s}", unit="sample"):
                features, label = self.generate_sample(disease_name)
                row = list(features) + [label]
                all_rows.append(row)

        columns = FEATURE_NAMES + ["label"]
        df = pd.DataFrame(all_rows, columns=columns)

        output_path = "data/training_dataset.csv"
        df.to_csv(output_path, index=False)

        print(f"\nDataset saved to {output_path}")
        print(f"Total samples : {len(df)}")
        print(f"Features      : {len(FEATURE_NAMES)}")
        print("\nClass distribution:")
        print(df["label"].value_counts().to_string())

        return df


if __name__ == "__main__":
    DatasetGenerator().generate_dataset(samples_per_class=1500)
