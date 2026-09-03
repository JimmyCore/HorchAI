"""Build a labeled keystroke dataset from raw recordings.

Expected raw data layout (see data/README.md and docs/recording_protocol.md):

    data/raw/<key_label>/<any_name>.wav

Each .wav is one recording session for a single key: press that key
repeatedly (with pauses) so `detect_keystrokes` can find each press as a
separate event. The parent folder name is used as the label, so keep raw
recordings out of git (see .gitignore) and only commit derived metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import audio as audio_mod
from . import detection as detection_mod

DEFAULT_KEYS = ["a", "s", "d", "f", "j", "k", "l"]


@dataclass
class DatasetBuildConfig:
    target_sr: int = 16000
    window_ms: float = 200.0
    pre_ms: float = 40.0
    frame_ms: float = 5.0
    hop_ms: float = 2.5
    min_gap_ms: float = 80.0
    threshold_factor: float = 3.0
    top_db_trim: float | None = None  # trimming is applied per-recording, not per-segment


@dataclass
class RawKeystroke:
    segment: np.ndarray
    label: str
    source_file: str
    time_sec: float
    sample_index: int
    energy: float


def build_dataset_from_raw(
    raw_dir: str,
    keys: list[str] | None = None,
    config: DatasetBuildConfig | None = None,
) -> list[RawKeystroke]:
    """Scan `raw_dir/<key>/*.wav`, detect keystrokes, and segment them.

    Returns a flat list of RawKeystroke, one per detected press, in the
    order files were processed.
    """
    keys = keys or DEFAULT_KEYS
    config = config or DatasetBuildConfig()
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"raw_dir '{raw_dir}' does not exist. Expected data/raw/<key>/*.wav — "
            "see docs/recording_protocol.md."
        )

    results: list[RawKeystroke] = []
    for key in keys:
        key_dir = raw_path / key
        if not key_dir.exists():
            print(f"[dataset] warning: no folder for key '{key}' at {key_dir}, skipping")
            continue

        wav_files = sorted(key_dir.glob("*.wav"))
        if not wav_files:
            print(f"[dataset] warning: no .wav files for key '{key}' in {key_dir}")
            continue

        for wav_file in wav_files:
            y, sr = audio_mod.load_audio(str(wav_file), sr=config.target_sr)
            y = audio_mod.normalize(y)

            keystrokes = detection_mod.detect_keystrokes(
                y,
                sr,
                frame_ms=config.frame_ms,
                hop_ms=config.hop_ms,
                min_gap_ms=config.min_gap_ms,
                threshold_factor=config.threshold_factor,
            )
            if not keystrokes:
                print(f"[dataset] warning: no keystrokes detected in {wav_file}")
                continue

            for k in keystrokes:
                segment = detection_mod.segment_keystroke(
                    y, sr, k.sample_index, window_ms=config.window_ms, pre_ms=config.pre_ms
                )
                results.append(
                    RawKeystroke(
                        segment=segment,
                        label=key,
                        source_file=str(wav_file.relative_to(raw_path)),
                        time_sec=k.time_sec,
                        sample_index=k.sample_index,
                        energy=k.energy,
                    )
                )

    return results


def to_metadata_df(items: list[RawKeystroke]) -> pd.DataFrame:
    """Metadata table (no raw audio) suitable for committing to git."""
    return pd.DataFrame(
        {
            "label": [it.label for it in items],
            "source_file": [it.source_file for it in items],
            "time_sec": [it.time_sec for it in items],
            "sample_index": [it.sample_index for it in items],
            "energy": [it.energy for it in items],
        }
    )


@dataclass
class Splits:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def stratified_split(
    labels: list[str],
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Splits:
    """Reproducible stratified train/val/test split by index.

    Splitting is stratified by label so every key is represented in every
    split in roughly the same proportion, regardless of how many samples
    were recorded per key.
    """
    n = len(labels)
    indices = np.arange(n)

    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels
    )
    train_val_labels = [labels[i] for i in train_val_idx]
    relative_val_size = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val_size, random_state=seed, stratify=train_val_labels
    )

    return Splits(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


class LabelEncoder:
    """Minimal, deterministic label <-> integer encoder (sorted label order)."""

    def __init__(self, keys: list[str]):
        self.classes_ = sorted(set(keys))
        self._to_idx = {label: i for i, label in enumerate(self.classes_)}

    def encode(self, labels: list[str]) -> np.ndarray:
        return np.array([self._to_idx[label] for label in labels], dtype="int64")

    def decode(self, indices: np.ndarray) -> list[str]:
        return [self.classes_[i] for i in indices]
