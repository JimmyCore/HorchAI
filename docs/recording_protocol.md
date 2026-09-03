# Recording Protocol (Phase 2 — 7-key dataset)

This documents exactly how to record the controlled dataset used by
`notebooks/horchai_poc.ipynb` for Phase 2/3. Following the same protocol
each time keeps recordings comparable across sessions and makes results
reproducible.

## Key set

Start with 7 keys on the home row:

```
A S D F J K L
```

These are chosen because they sit on the home row (consistent hand
position, minimal finger travel) and span both hands, giving a reasonably
diverse but small starting classification problem.

## Setup

- **Device**: your own keyboard and your own recording device (phone or
  laptop mic). Note the exact model of both in the recording's metadata —
  results do not transfer across keyboard models.
- **Environment**: a quiet room. Keep the setup (mic position, distance to
  keyboard, background noise) the same across all recordings in one
  session — the whole point of Phase 2 is a *controlled* dataset.
- **Microphone distance**: pick a fixed distance (e.g. 20cm from the
  keyboard) and keep it constant. Write it down.
- **Format**: record as `.wav`, mono if possible, at 44.1kHz or 48kHz. The
  pipeline resamples to 16kHz internally, so the source rate isn't
  critical as long as it's uncompressed.

## Procedure

For **each key**, record 2-3 separate takes. In each take:

1. Start recording.
2. Wait ~1s of silence (helps the detector see the true noise floor).
3. Press the key **10-15 times**, with a natural pause (~0.5-1s) between
   presses so each press is a clearly separated event.
4. Stop recording.

Save each take as:

```
data/raw/<key>/<take_id>.wav
```

e.g. `data/raw/a/take1.wav`, `data/raw/a/take2.wav`, ...

This gives ~20-45 labeled presses per key from a few short recordings,
which `src/dataset.py` (`build_dataset_from_raw`) will automatically
detect, segment, and label from the folder name.

## What NOT to record

- Do not record anyone else typing without their explicit, informed
  consent.
- Do not record real passwords or credentials, even your own — use the
  key set above (single-key presses), not full sentences, for Phase 2/3.

## Metadata to note down

For each recording session, note (e.g. in the take's filename or a short
text file alongside it):

- Date
- Keyboard model
- Microphone / device model
- Distance from keyboard to mic
- Any background noise present

This is what later lets you compare "clean" vs "noisy" experiments
honestly in Phase 5.
