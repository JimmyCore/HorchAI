"""Audio loading and basic preprocessing for HorchAI.

Keep this module dependency-light and side-effect free: functions take
arrays/paths in, return arrays/values out, so they are easy to test and to
call both from notebooks and from scripts.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import librosa


def load_audio(path: str, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load an audio file.

    Args:
        path: path to a .wav/.flac/.ogg/... file.
        sr: target sample rate. If None, keep the file's native rate.
        mono: downmix to mono if the file has multiple channels.

    Returns:
        (samples, sample_rate) with samples as float32 in [-1, 1].
    """
    y, native_sr = sf.read(path, always_2d=False, dtype="float32")
    if y.ndim > 1 and mono:
        y = y.mean(axis=1).astype("float32")

    if sr is not None and sr != native_sr:
        y = librosa.resample(y, orig_sr=native_sr, target_sr=sr)
        native_sr = sr

    return y, native_sr


def get_duration(y: np.ndarray, sr: int) -> float:
    """Duration of a signal in seconds."""
    return len(y) / float(sr)


def normalize(y: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalize a signal to +/- target_peak. No-ops on silence."""
    peak = np.max(np.abs(y)) if len(y) else 0.0
    if peak < 1e-9:
        return y
    return (y / peak * target_peak).astype("float32")


def trim_silence(y: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """Trim leading/trailing near-silence using librosa's energy-based trim."""
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed


def preprocess(y: np.ndarray, sr: int, target_sr: int | None = None, top_db: float | None = None) -> tuple[np.ndarray, int]:
    """Convenience wrapper: optional resample + normalize + optional trim."""
    if target_sr is not None and target_sr != sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    y = normalize(y)
    if top_db is not None:
        y = trim_silence(y, top_db=top_db)
    return y, sr
