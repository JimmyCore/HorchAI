"""Keystroke onset detection and segmentation.

Approach (Phase 1, intentionally simple): compute a short-time energy
envelope, find local peaks above an adaptive threshold with a minimum
spacing between them, and treat each peak as one keystroke event. This is
a baseline detector, not the final word — noisier recordings will need
better onset detection (e.g. spectral flux) in a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass
class Keystroke:
    """One detected keystroke event."""

    sample_index: int
    time_sec: float
    energy: float


def energy_envelope(y: np.ndarray, sr: int, frame_ms: float = 5.0, hop_ms: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    """Short-time RMS energy envelope.

    Returns:
        (envelope, times_sec) where envelope[i] is the RMS energy of the
        frame centered at times_sec[i].
    """
    frame_len = max(1, int(sr * frame_ms / 1000))
    hop_len = max(1, int(sr * hop_ms / 1000))

    if len(y) < frame_len:
        return np.array([]), np.array([])

    n_frames = 1 + (len(y) - frame_len) // hop_len
    envelope = np.empty(n_frames, dtype="float32")
    for i in range(n_frames):
        start = i * hop_len
        frame = y[start : start + frame_len]
        envelope[i] = np.sqrt(np.mean(frame.astype("float64") ** 2))

    times = (np.arange(n_frames) * hop_len + frame_len / 2) / sr
    return envelope, times


def detect_keystrokes(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 5.0,
    hop_ms: float = 2.5,
    min_gap_ms: float = 80.0,
    threshold_factor: float = 3.0,
) -> list[Keystroke]:
    """Detect keystroke onsets from audio energy.

    The threshold is adaptive: `threshold_factor` standard deviations above
    the envelope's median, which is robust to a quiet-but-nonzero noise
    floor. Peaks closer than `min_gap_ms` are merged (only the tallest is
    kept), since a single keystroke should not produce two events.

    Args:
        y: mono audio signal.
        sr: sample rate.
        frame_ms: analysis frame length for the energy envelope.
        hop_ms: hop size between frames.
        min_gap_ms: minimum time between two distinct keystrokes.
        threshold_factor: sensitivity of the adaptive threshold; lower
            values detect more (and noisier) events.

    Returns:
        list of Keystroke, sorted by time.
    """
    envelope, times = energy_envelope(y, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    if len(envelope) == 0:
        return []

    noise_floor = np.median(envelope)
    threshold = noise_floor + threshold_factor * np.std(envelope)

    hop_len = max(1, int(sr * hop_ms / 1000))
    min_gap_frames = max(1, int((min_gap_ms / 1000) * sr / hop_len))

    peak_indices, _ = find_peaks(envelope, height=threshold, distance=min_gap_frames)

    keystrokes = [
        Keystroke(
            sample_index=int(times[i] * sr),
            time_sec=float(times[i]),
            energy=float(envelope[i]),
        )
        for i in peak_indices
    ]
    return keystrokes


def segment_keystroke(
    y: np.ndarray,
    sr: int,
    center_sample: int,
    window_ms: float = 200.0,
    pre_ms: float = 40.0,
) -> np.ndarray:
    """Cut a fixed-length window around a detected keystroke.

    The window starts `pre_ms` before the detected onset (to capture the
    attack) and extends to cover `window_ms` total. Out-of-range windows are
    zero-padded so every segment has the same length regardless of where
    the keystroke falls in the recording.

    Args:
        y: mono audio signal.
        sr: sample rate.
        center_sample: sample index of the detected onset.
        window_ms: total segment duration in milliseconds.
        pre_ms: how much of the window sits before the onset.

    Returns:
        1D array of length round(sr * window_ms / 1000).
    """
    window_len = int(round(sr * window_ms / 1000))
    pre_len = int(round(sr * pre_ms / 1000))

    start = center_sample - pre_len
    end = start + window_len

    segment = np.zeros(window_len, dtype="float32")
    src_start = max(0, start)
    src_end = min(len(y), end)
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)
    if src_end > src_start:
        segment[dst_start:dst_end] = y[src_start:src_end]

    return segment


def segment_keystrokes(
    y: np.ndarray,
    sr: int,
    keystrokes: list[Keystroke],
    window_ms: float = 200.0,
    pre_ms: float = 40.0,
) -> np.ndarray:
    """Segment every detected keystroke into a fixed-length window.

    Returns:
        array of shape (n_keystrokes, window_samples).
    """
    return np.stack(
        [segment_keystroke(y, sr, k.sample_index, window_ms=window_ms, pre_ms=pre_ms) for k in keystrokes]
    ) if keystrokes else np.zeros((0, int(round(sr * window_ms / 1000))), dtype="float32")
