from dataclasses import dataclass


@dataclass
class PatientMode:
    name: str
    bpm: float
    bpm_variability: float
    amplitude: float
    rise_time: float
    noise_multiplier: float
    description: str


BRADYCARDIA = PatientMode(
    name="BRADYCARDIA",
    bpm=30,
    bpm_variability=2,
    amplitude=1.0,
    rise_time=0.1,
    noise_multiplier=0.02,
    description="Abnormally slow heart rate below 60 BPM, reducing cardiac output and causing fatigue or syncope.",
)

TACHYCARDIA = PatientMode(
    name="TACHYCARDIA",
    bpm=150,
    bpm_variability=5,
    amplitude=0.7,
    rise_time=0.08,
    noise_multiplier=0.03,
    description="Abnormally fast heart rate above 100 BPM, which can impair ventricular filling and reduce stroke volume.",
)

HYPOTENSION = PatientMode(
    name="HYPOTENSION",
    bpm=75,
    bpm_variability=3,
    amplitude=0.2,
    rise_time=0.1,
    noise_multiplier=0.06,
    description="Dangerously low blood pressure reducing organ perfusion, indicated by a weakened pulse amplitude.",
)

HYPERTENSION = PatientMode(
    name="HYPERTENSION",
    bpm=75,
    bpm_variability=2,
    amplitude=2.0,
    rise_time=0.05,
    noise_multiplier=0.02,
    description="Chronically elevated blood pressure that strains arterial walls and increases risk of stroke and heart failure.",
)

MI = PatientMode(
    name="MI",
    bpm=80,
    bpm_variability=20,
    amplitude=0.8,
    rise_time=0.1,
    noise_multiplier=0.08,
    description="Myocardial infarction where a blocked coronary artery causes ischemic death of heart muscle, producing erratic rhythm.",
)

NORMAL = PatientMode(
    name="NORMAL",
    bpm=72,
    bpm_variability=8,
    amplitude=1.0,
    rise_time=0.1,
    noise_multiplier=0.01,
    description="Healthy resting heart with normal sinus rhythm and typical heart rate variability.",
)

ALL_MODES = {
    "BRADYCARDIA": BRADYCARDIA,
    "TACHYCARDIA": TACHYCARDIA,
    "HYPOTENSION": HYPOTENSION,
    "HYPERTENSION": HYPERTENSION,
    "MI": MI,
    "NORMAL": NORMAL,
}
