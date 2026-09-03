# exp_001_peak_detection

- **Datum**: 2026-09-03
- **Ziel**: Sanity-check the Phase 1 energy-based keystroke detector
  (`src/detection.py:detect_keystrokes`) before pointing it at real
  recordings — does it recover a known number of events at known times
  from a signal with a known noise floor?
- **Dataset-Version**: Synthetic only — no real keystroke recordings exist
  yet. Generated a 3s, 16kHz mono signal: Gaussian noise floor (σ=0.01)
  with 5 short (10ms) Hann-windowed noise bursts (σ=0.5) injected at
  t = 0.3, 0.9, 1.5, 2.1, 2.7s as stand-ins for keystroke clicks.
- **Parameter**: `detect_keystrokes` defaults — `frame_ms=5.0, hop_ms=2.5,
  min_gap_ms=80.0, threshold_factor=3.0`. Signal peak-normalized before
  detection (`audio.normalize`).
- **Modell**: N/A (signal-processing heuristic, not a learned model).
- **Git Commit**: recorded at the commit that introduces this experiment
  (see PR history for `experiments/exp_001_peak_detection.md`).

## Resultate

All 5 injected events were detected, at:

| injected t (s) | detected t (s) | Δ (ms) |
|---|---|---|
| 0.300 | 0.305 | +5 |
| 0.900 | 0.905 | +5 |
| 1.500 | 1.505 | +5 |
| 2.100 | 2.105 | +5 |
| 2.700 | 2.705 | +5 |

0 false positives, 0 false negatives. `segment_keystrokes` then produced 5
fixed-length (200ms) windows of shape `(5, 3200)` at 16kHz, and
`features.batch_mel_spectrograms` turned those into 5 spectrograms of
shape `(64, 51)`, each normalized to `[0, 1]`.

## Interpretation

The +5ms offset is expected and not an error: the detector's frame hop is
2.5ms and the energy envelope needs a few frames to rise above threshold
after the burst starts, so a small, consistent positive lag is the correct
behavior of an energy-threshold detector (it fires slightly after the true
onset, never before). The detector correctly separates events spaced
600ms apart with no merging or splitting, and produces the right output
shapes downstream. This validates the mechanics of the Phase 1 pipeline
(detect → segment → mel-spectrogram) end-to-end, but says nothing yet
about real keystroke acoustics — synthetic noise bursts are a much
easier detection target than an actual key click, which has structure
(press transient + release transient) and sits in a real acoustic
environment.

## Nächster sinnvoller Schritt

Run `notebooks/horchai_poc.ipynb` Phase 1 cells against one real
recording (a few presses of a single key) and check by ear/eye whether
the detected peak times line up with the audible clicks in the waveform
plot. If `threshold_factor=3.0` over- or under-detects on real audio,
tune it there before moving to Phase 2 (building the full 7-key dataset).
