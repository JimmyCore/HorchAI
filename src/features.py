"""Feature extraction: mel spectrograms for keystroke segments."""

from __future__ import annotations

import numpy as np
import librosa


def mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = 64,
    n_fft: int = 512,
    hop_length: int = 64,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Log-scaled mel spectrogram.

    Defaults are tuned for short (~200ms) keystroke segments rather than
    speech: a small hop/FFT size keeps time resolution high enough to
    resolve the press/release click structure.

    Returns:
        array of shape (n_mels, n_frames), in dB (log-power).
    """
    if fmax is None:
        fmax = sr / 2

    n_fft = min(n_fft, len(y)) if len(y) > 0 else n_fft
    n_fft = max(n_fft, 32)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype("float32")


def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
    """Min-max normalize a spectrogram (in dB) to [0, 1] for model input."""
    lo, hi = spec.min(), spec.max()
    if hi - lo < 1e-9:
        return np.zeros_like(spec)
    return ((spec - lo) / (hi - lo)).astype("float32")


def batch_mel_spectrograms(
    segments: np.ndarray,
    sr: int,
    n_mels: int = 64,
    n_fft: int = 512,
    hop_length: int = 64,
) -> np.ndarray:
    """Compute normalized mel spectrograms for a batch of equal-length segments.

    Args:
        segments: array of shape (n_segments, n_samples).

    Returns:
        array of shape (n_segments, n_mels, n_frames), normalized to [0, 1].
    """
    specs = [
        normalize_spectrogram(mel_spectrogram(seg, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length))
        for seg in segments
    ]
    return np.stack(specs)
