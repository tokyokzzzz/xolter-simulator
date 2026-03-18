import numpy as np
from datetime import datetime, timezone
from scipy.signal import find_peaks

from simulator.modes import ALL_MODES
from simulator.noise import shot_noise, thermal_drift, bending_loss


class SignalGenerator:

    def __init__(self, mode_name="NORMAL", sampling_rate=250):
        self.mode = ALL_MODES[mode_name]
        self.sampling_rate = sampling_rate

    def generate_beat(self):
        """Returns one Gaussian heartbeat pulse as a numpy array."""
        # Width in samples: rise_time in seconds * sampling_rate
        sigma = self.mode.rise_time * self.sampling_rate
        # Pulse spans 6 sigma so the tails reach near zero
        width = int(sigma * 6)
        if width < 1:
            width = 1
        t = np.arange(width)
        center = width / 2.0
        pulse = self.mode.amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2)
        return pulse

    def generate_seconds(self, seconds=10):
        """Returns a continuous PPG signal of the given duration with all noise applied."""
        total_samples = seconds * self.sampling_rate
        signal = np.zeros(total_samples)

        beat = self.generate_beat()
        beat_len = len(beat)

        # Place beats across the timeline
        cursor = 0
        while cursor < total_samples:
            # Jitter the interval around the nominal BPM
            jittered_bpm = self.mode.bpm + np.random.uniform(
                -self.mode.bpm_variability, self.mode.bpm_variability
            )
            jittered_bpm = max(jittered_bpm, 1.0)  # guard against zero/negative
            interval = int((60.0 / jittered_bpm) * self.sampling_rate)

            end = min(cursor + beat_len, total_samples)
            slice_len = end - cursor
            signal[cursor:end] += beat[:slice_len]

            cursor += interval

        # Apply noise layers
        drift = thermal_drift(total_samples, self.sampling_rate)
        signal = signal + drift
        signal = shot_noise(signal, self.mode.noise_multiplier)
        signal = bending_loss(signal)

        return signal

    def get_live_reading(self):
        """Returns a dict with current vitals and a 2.5-second signal snippet."""
        # Generate 10 seconds for reliable peak detection
        signal_10s = self.generate_seconds(seconds=10)

        # Remove slow baseline drift with a 4-second moving-average (eliminates
        # the 0.05 Hz thermal drift sine that would otherwise cause false peaks)
        window = self.sampling_rate * 4
        padded = np.pad(signal_10s, window // 2, mode='edge')
        moving_avg = np.convolve(padded, np.ones(window) / window, mode='valid')[:len(signal_10s)]
        detrended = signal_10s - moving_avg

        # Peak detection on detrended signal
        height = np.mean(detrended) + 0.3 * np.std(detrended)
        min_distance = int(self.sampling_rate * 0.35)  # 0.35s => max ~171 BPM
        peaks, _ = find_peaks(detrended, height=height, distance=min_distance)

        # BPM from peak count over 10-second window
        bpm = (len(peaks) / 10.0) * 60.0

        # Validate: if detected BPM is outside mode's expected range
        # (e.g. overlapping beats in TACHYCARDIA or noise peaks in BRADYCARDIA)
        # fall back to the mode's nominal BPM with variability jitter
        expected_lo = max(1.0, self.mode.bpm - 3 * self.mode.bpm_variability)
        expected_hi = self.mode.bpm + 3 * self.mode.bpm_variability
        if not (expected_lo <= bpm <= expected_hi):
            bpm = self.mode.bpm + np.random.uniform(
                -self.mode.bpm_variability, self.mode.bpm_variability
            )

        # Fixed BP formula: base 120 systolic, scaled by amplitude deviation from 1.0
        systolic = round(120.0 + ((self.mode.amplitude - 1.0) * 40.0), 1)
        diastolic = round(systolic * 0.65, 1)

        # Return only the last 2.5 seconds (625 samples) as the display snippet
        snippet = signal_10s[-625:]

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bpm": round(bpm, 1),
            "systolic_bp": systolic,
            "diastolic_bp": diastolic,
            "signal_snippet": snippet.tolist(),
            "mode": self.mode.name,
        }
