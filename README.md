# HorchAI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JimmyCore/HorchAI/blob/main/notebooks/horchai_poc.ipynb)

**HorchAI** is a security-research proof of concept investigating **acoustic
side-channel attacks on keyboards**: whether the *sound* of a keystroke
alone is enough to identify which key was pressed. It reproduces and
studies the phenomenon described in prior academic work such as
[arXiv:2504.11622](https://arxiv.org/abs/2504.11622).

## Research & security context

This project exists to understand a real information-leakage channel (a
laptop mic picking up keystrokes during a call, for example) well enough to
reason about mitigations — not to build a surveillance tool. It is used
**only with the author's own devices and recordings, or recordings with
explicit authorization**. No real third-party passwords, credentials, or
non-consensual input are ever targeted or reconstructed. Full policy:
[`docs/ethics.md`](docs/ethics.md).

## Pipeline

```
Audioaufnahme
  → Audio Preprocessing
  → Keystroke Detection
  → Segmentierung einzelner Anschläge
  → Mel-Spektrogramme
  → Klassifikation
  → Noise-Robustness-Tests
  → CNN/Transformer-Vergleich
  → Evaluation
```

A later, clearly separated research branch (Phase 6) studies — on synthetic
text only — whether a language model can correct classification errors
(e.g. `"the passwprd is banana"` → `"the password is banana"`).

## Current status: v0.1 (Phases 1–3)

| Phase | What | Status |
|---|---|---|
| 1 | Energy-based keystroke detection on a single recording | ✅ implemented, validated on synthetic audio ([exp_001](experiments/exp_001_peak_detection.md)) |
| 2 | Controlled 7-key (`A S D F J K L`) dataset: segmentation, labeling, reproducible splits | ✅ implemented (`src/dataset.py`) |
| 3 | CNN baseline: accuracy, precision/recall, confusion matrix | ✅ implemented (`src/model.py`, `src/evaluation.py`), validated end-to-end on synthetic data |
| 4 | Vision Transformer, fair CNN-vs-ViT comparison | not started |
| 5 | Noise robustness (noise level → accuracy) | not started |
| 6 | LM-based correction of classification errors (synthetic text only) | not started |

**Important:** the full pipeline (detection → segmentation → mel
spectrogram → CNN → accuracy/confusion matrix) has been validated
end-to-end, but only on **synthetic test signals**, not on real keyboard
recordings — none exist in this repository yet (by design; see
[`data/README.md`](data/README.md)). The first real-world run is the next
step, not a completed result. See [`experiments/`](experiments/) for exact
methodology and numbers of every run so far.

## Requirements

- A Google account (for Colab) — no local setup needed.
- Your own keyboard + a microphone (phone or laptop) to record with.
- Optional, for local/offline use: Python 3.10+, `pip install -r requirements.txt`.

## Usage (Google Colab)

This project is built to be run from a **phone**, primarily through Colab:

1. Open [`notebooks/horchai_poc.ipynb`](notebooks/horchai_poc.ipynb) via the
   **Open in Colab** badge above.
2. **Runtime → Run all.**
3. When prompted, upload an audio recording (Phase 1) or point the
   notebook at your `data/raw/` folder in Google Drive (Phase 2/3).
4. Read the results (waveform, spectrogram, detected keystrokes, accuracy,
   confusion matrix) at the bottom.

No manual terminal steps required.

## Data collection

Controlled recordings for the 7-key dataset follow a fixed protocol so
results are reproducible and comparable across sessions — see
[`docs/recording_protocol.md`](docs/recording_protocol.md) for exactly how
to record (key set, environment, file layout, metadata to note down).

Raw audio, segmented keystrokes, and model weights are **never committed**
to this repository (see `.gitignore`); only code and small non-sensitive
metadata (labels, timestamps, split assignments, experiment results) are
version-controlled. Details: [`data/README.md`](data/README.md).

## Repository structure

```
HorchAI/
├── README.md
├── requirements.txt
├── notebooks/
│   └── horchai_poc.ipynb      # Phase 1-3 Colab notebook
├── src/
│   ├── audio.py                # loading, resampling, normalization
│   ├── detection.py            # energy-based keystroke onset detection + segmentation
│   ├── features.py              # mel spectrogram extraction
│   ├── dataset.py              # dataset building, label encoding, reproducible splits
│   ├── model.py                # SimpleCNN baseline
│   └── evaluation.py           # accuracy, precision/recall, confusion matrix
├── experiments/                # one exp_XXX_<name>.md per experiment run
├── data/
│   └── README.md               # data layout & policy (raw data is git-ignored)
└── docs/
    ├── ethics.md                # research/security scope and rules
    └── recording_protocol.md   # exact data-collection procedure
```

## Known limitations

- No real keyboard recordings have been collected/evaluated yet — current
  validation is on synthetic signals only (see [exp_001](experiments/exp_001_peak_detection.md)).
- The Phase 1 detector is a simple adaptive energy threshold; it has not
  been tuned against real click transients or background noise, and will
  likely need `threshold_factor` adjustments per recording setup.
- The dataset size at this stage (a handful of recordings per key) is far
  too small to say anything about generalization across keyboards, typists,
  or environments — Phase 3 numbers, once run on real data, describe this
  specific controlled setup only.
- No noise-robustness, cross-keyboard, or cross-session evaluation exists
  yet (Phase 5).

## Experiment results

See [`experiments/`](experiments/) for the full, dated log of every run
(goal, dataset version, parameters, model, git commit, results,
interpretation, next step). Summary so far:

- **[exp_001_peak_detection](experiments/exp_001_peak_detection.md)** —
  Phase 1 detector recovers 5/5 synthetic click events with 0 false
  positives/negatives and a consistent +5ms detection lag. Validates the
  detection mechanics; says nothing yet about real keystroke acoustics.

## Next steps

1. Record a first real Phase 2 dataset (7 keys, following
   [`docs/recording_protocol.md`](docs/recording_protocol.md)) and run the
   full v0.1 notebook against it — this is the first result that will
   actually say something about real keystroke classification accuracy.
2. Log that run as `exp_002_cnn_clean` in `experiments/`, including
   whether the Phase 1 detector needed threshold tuning for real audio.
3. Only once that baseline is reliable: extend the key set, then move to
   Phase 4 (Transformer comparison) and Phase 5 (noise robustness).
