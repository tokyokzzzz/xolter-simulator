import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq


class FeatureExtractor:

    def extract_time_domain(self, signal: np.ndarray) -> dict:
        """
        Time-domain features capture the statistical shape and rhythm of the PPG waveform.
        RR intervals and their variability (HRV) are gold-standard markers for autonomic
        nervous system function, arrhythmia detection, and cardiac stress assessment.
        """
        signal = np.asarray(signal, dtype=float)

        peaks, properties = find_peaks(signal, height=0.3, distance=50)
        peak_amplitudes = properties["peak_heights"] if len(peaks) > 0 else np.array([0.0])

        # RR intervals: time between successive peaks in milliseconds
        # Assumes the signal was sampled at 250 Hz (4 ms per sample)
        if len(peaks) >= 2:
            rr_intervals = np.diff(peaks) * 4.0  # convert samples → ms
            mean_rr = float(np.mean(rr_intervals))
            rr_std = float(np.std(rr_intervals))
            successive_diffs = np.diff(rr_intervals)
            rr_rmssd = float(np.sqrt(np.mean(successive_diffs ** 2))) if len(successive_diffs) > 0 else 0.0
        else:
            mean_rr = 0.0
            rr_std = 0.0
            rr_rmssd = 0.0

        return {
            "mean_amplitude": float(np.mean(signal)),
            "std_amplitude": float(np.std(signal)),
            "peak_count": int(len(peaks)),
            "mean_rr_interval": round(mean_rr, 3),
            "rr_std": round(rr_std, 3),
            "rr_rmssd": round(rr_rmssd, 3),
            "signal_energy": float(np.sum(signal ** 2)),
            "skewness": float(skew(signal)),
            "kurtosis": float(kurtosis(signal)),
            "peak_amplitude_mean": float(np.mean(peak_amplitudes)),
            "peak_amplitude_std": float(np.std(peak_amplitudes)),
        }

    def extract_frequency_domain(self, signal: np.ndarray, sampling_rate: int = 250) -> dict:
        """
        Frequency-domain features decompose the signal into autonomic nervous system bands.
        LF power (0.04–0.15 Hz) reflects sympathetic activity; HF power (0.15–0.4 Hz)
        reflects parasympathetic (vagal) tone; their ratio is a marker of cardiac autonomic balance.
        """
        signal = np.asarray(signal, dtype=float)
        n = len(signal)

        freqs = fftfreq(n, d=1.0 / sampling_rate)
        power = np.abs(fft(signal)) ** 2

        # Keep only positive frequencies
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        power = power[pos_mask]

        def band_power(low, high):
            mask = (freqs >= low) & (freqs < high)
            return float(np.sum(power[mask])) if np.any(mask) else 0.0

        lf = band_power(0.04, 0.15)
        hf = band_power(0.15, 0.40)
        lf_hf_ratio = lf / hf if hf > 0 else 0.0

        dominant_freq = float(freqs[np.argmax(power)])

        # Spectral entropy: normalized Shannon entropy of the power spectrum
        total_power = np.sum(power)
        if total_power > 0:
            prob = power / total_power
            prob = prob[prob > 0]
            spectral_entropy = float(-np.sum(prob * np.log2(prob)))
        else:
            spectral_entropy = 0.0

        return {
            "lf_power": round(lf, 4),
            "hf_power": round(hf, 4),
            "lf_hf_ratio": round(lf_hf_ratio, 4),
            "dominant_frequency": round(dominant_freq, 4),
            "spectral_entropy": round(spectral_entropy, 4),
        }

    def extract_all(
        self,
        signal: np.ndarray,
        bpm: float,
        systolic_bp: float,
        diastolic_bp: float,
    ) -> np.ndarray:
        """
        Combines time-domain features, frequency-domain features, and vital sign readings
        into a single flat feature vector ready for input to a machine learning model.
        """
        time_feats = self.extract_time_domain(signal)
        freq_feats = self.extract_frequency_domain(signal)

        vitals = [bpm, systolic_bp, diastolic_bp]

        feature_vector = (
            list(time_feats.values())
            + list(freq_feats.values())
            + vitals
        )

        arr = np.array(feature_vector, dtype=float)
        return arr
