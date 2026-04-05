import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from pathlib import Path
import streamlit as st

# Path configuration
MODEL_PATH = Path(__file__).resolve().parent / "models" / "medical_heart_anomaly_model.h5"

# Mock scaler parameters (based on typical MIT-BIH RR-interval normalization)
SCALER_MEAN = 0.8
SCALER_STD = 0.2

@st.cache_resource
def load_heart_model():
    """Load the pre-trained heart anomaly detection model."""
    if MODEL_PATH.exists():
        return load_model(str(MODEL_PATH))
    else:
        return None

def normalize_rr_data(rr_sequence):
    """Normalize RR sequence for model inference."""
    return (rr_sequence - SCALER_MEAN) / SCALER_STD

def predict_heart_condition(model, rr_sequence, confidence_threshold=0.8):
    """Run inference on a window of RR intervals."""
    rr_sequence = np.asarray(rr_sequence, dtype=np.float32)
    bpm = calculate_bpm(rr_sequence)
    rr_std = float(np.std(rr_sequence)) if len(rr_sequence) > 0 else 0.0

    # Heuristic fallback/calibration. This prevents "always abnormal" behavior when
    # model confidence is weak or when the model is unavailable.
    heuristic_class = "NORMAL"
    if rr_std >= 0.18 or bpm < 45 or bpm > 130:
        heuristic_class = "EMERGENCY"
    elif rr_std >= 0.08 or bpm < 55 or bpm > 110:
        heuristic_class = "WARNING"

    if model is None:
        fallback_probs = {
            "Normal": 0.8 if heuristic_class == "NORMAL" else 0.1,
            "Warning": 0.8 if heuristic_class == "WARNING" else 0.1,
            "Emergency": 0.8 if heuristic_class == "EMERGENCY" else 0.1,
        }
        return {
            "class": heuristic_class,
            "probabilities": fallback_probs,
            "trigger_sos": bool(heuristic_class == "EMERGENCY"),
            "confidence": 0.0,
        }

    rr_norm = normalize_rr_data(rr_sequence)
    X = rr_norm.reshape(1, len(rr_sequence), 1)

    probabilities = model.predict(X, verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])

    class_names = {0: "NORMAL", 1: "WARNING", 2: "EMERGENCY"}
    model_class = class_names[predicted_class]

    # Confidence-aware gating:
    # - very low confidence -> trust heuristic
    # - moderate confidence -> avoid escalating to emergency on weak evidence
    if confidence < 0.60:
        final_class = heuristic_class
    elif confidence < confidence_threshold and model_class == "EMERGENCY":
        final_class = "WARNING" if heuristic_class != "NORMAL" else "NORMAL"
    else:
        final_class = model_class

    # Format probabilities for Streamlit (dictionary for easy plotting)
    prob_dict = {
        "Normal": float(probabilities[0]),
        "Warning": float(probabilities[1]),
        "Emergency": float(probabilities[2]),
    }

    return {
        "class": final_class,
        "probabilities": prob_dict,
        "trigger_sos": bool(
            (final_class == "EMERGENCY" and confidence >= confidence_threshold)
            or (heuristic_class == "EMERGENCY" and confidence < 0.60)
        ),
        "confidence": confidence,
    }

def generate_demo_heart_data(window_size=20):
    """Generate mock RR intervals for demo purposes."""
    # Regime probabilities tuned so demo is mostly normal.
    regime_roll = np.random.rand()

    if regime_roll < 0.96:
        # NORMAL: ~67-82 BPM, very low variability to avoid false warnings.
        data = np.random.normal(loc=0.82, scale=0.02, size=window_size)
    elif regime_roll < 0.995:
        # WARNING: mild tachy/brady or modest irregularity
        base = np.random.choice([0.56, 1.08])  # around ~107 BPM or ~56 BPM
        data = np.random.normal(loc=base, scale=0.07, size=window_size)
    else:
        # EMERGENCY: highly irregular rhythm, including outliers
        data = np.random.normal(loc=0.95, scale=0.22, size=window_size)
        spike_count = max(1, window_size // 8)
        spike_idx = np.random.choice(window_size, size=spike_count, replace=False)
        data[spike_idx] += np.random.choice([-0.35, 0.35], size=spike_count)

    # Physiological clamp: 0.35s to 1.5s RR interval (~40 to 171 BPM)
    return np.clip(data, 0.35, 1.5)

def calculate_bpm(rr_intervals):
    """Calculate average BPM from RR intervals (in seconds)."""
    if len(rr_intervals) == 0:
        return 0
    mean_rr = float(np.mean(rr_intervals))
    if mean_rr <= 0:
        return 0
    bpm = 60.0 / mean_rr
    # Clamp to realistic display range for demo UX.
    return float(np.clip(bpm, 40, 160))

def generate_ecg_point(heart_rate_bpm, time_step):
    """
    Generate a single synthetic ECG signal point using a simplified P-QRS-T model.
    heart_rate_bpm: Current BPM
    time_step: Current time index in the heart cycle (0.0 to 1.0)
    """
    # Define timing of ECG components (fractions of heart cycle)
    p_peak = 0.15; q_peak = 0.25; r_peak = 0.35; s_peak = 0.45; t_peak = 0.65
    
    # Amplitudes
    p_amp = 0.15; q_amp = -0.25; r_amp = 1.0; s_amp = -0.3; t_amp = 0.35
    
    # Pulse widths (Gaussian-like)
    p_width = 0.05; q_width = 0.02; r_width = 0.02; s_width = 0.02; t_width = 0.1
    
    # Summing the components
    p_wave = p_amp * np.exp(-((time_step - p_peak)**2) / (2 * p_width**2))
    q_wave = q_amp * np.exp(-((time_step - q_peak)**2) / (2 * q_width**2))
    r_wave = r_amp * np.exp(-((time_step - r_peak)**2) / (2 * r_width**2))
    s_wave = s_amp * np.exp(-((time_step - s_peak)**2) / (2 * s_width**2))
    t_wave = t_amp * np.exp(-((time_step - t_peak)**2) / (2 * t_width**2))
    
    # Add some noise for realism
    noise = np.random.normal(0, 0.02)
    
    return p_wave + q_wave + r_wave + s_wave + t_wave + noise
