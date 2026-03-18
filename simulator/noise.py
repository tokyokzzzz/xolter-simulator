import numpy as np


def shot_noise(signal, multiplier=0.02):
    """Photon counting noise in fiber optics where variance scales with signal intensity."""
    signal = np.asarray(signal, dtype=float)
    noise = np.random.normal(0, multiplier * np.sqrt(np.abs(signal)))
    return signal + noise


def thermal_drift(length, sampling_rate=250):
    """Temperature changes along the fiber alter its refractive index, shifting the baseline slowly."""
    t = np.arange(length) / sampling_rate
    return np.sin(2 * np.pi * 0.05 * t)


def bending_loss(signal, probability=0.01):
    """Micro-bends in the fiber cause localized attenuation by scattering light out of the core."""
    signal = np.asarray(signal, dtype=float).copy()
    i = 0
    while i < len(signal):
        if np.random.random() < probability:
            duration = np.random.randint(50, 201)
            end = min(i + duration, len(signal))
            attenuation = np.random.uniform(0.4, 0.7)
            signal[i:end] *= attenuation
            i = end
        else:
            i += 1
    return signal
