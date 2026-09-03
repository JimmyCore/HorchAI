# Research & Security Context

HorchAI is a security research project investigating **acoustic side-channel
attacks on keyboards** — the idea, studied in prior academic work such as
[arXiv:2504.11622](https://arxiv.org/abs/2504.11622), that the sound of a
keystroke leaks enough information to identify which key was pressed.

## Why this matters

Laptop and mechanical keyboards are used near always-on microphones (video
calls, voice assistants, phones on the desk). If keystroke sounds can be
reliably classified, that is a real information-leakage channel — one that
users and organizations should understand in order to defend against it
(e.g. via typing-sound masking, microphone permission hygiene, or acoustic
dampening). This project reproduces and studies that channel to understand
how practical it is and where it breaks down (noise, distance, keyboard
model, typing style).

## Rules for this project

- **Only use your own devices and recordings, or recordings you have
  explicit authorization to use.** Never record someone else's typing
  without their informed consent.
- **No reconstruction of real, non-consensual input.** Do not attempt to
  recover actual passwords, credentials, or private messages typed by
  someone else. All password/credential examples used in later experiments
  (Phase 6, the language-model correction study) are synthetic strings
  written specifically for testing, not real secrets.
- **Data minimization.** Raw audio recordings are never committed to this
  repository (see `.gitignore` and `data/README.md`). Only derived,
  non-sensitive metadata (timestamps, labels, aggregate metrics) is
  version-controlled.
- **Defensive framing.** The purpose of building an accurate classifier is
  to measure the real-world risk and inform mitigations — not to build a
  tool for surveilling third parties.

If you are unsure whether a use of this code is in scope, don't do it —
open an issue and discuss it first.
