# Experiment Tracking

Every experiment that produces a result worth remembering gets an ID and a
short writeup here, e.g.:

```
exp_001_peak_detection
exp_002_cnn_clean
exp_003_cnn_noise_low
exp_004_vit_clean
```

Each `exp_XXX_<name>.md` documents, at minimum:

- **Ziel** — what question this experiment answers
- **Dataset-Version** — which recordings / split (path or description)
- **Parameter** — the config used (detection thresholds, model hyperparams, ...)
- **Modell** — what was evaluated
- **Git Commit** — commit hash the experiment was run against
- **Resultate** — the actual numbers (accuracy, confusion matrix, etc.)
- **Interpretation** — what the numbers mean
- **Nächster sinnvoller Schritt** — the smallest next experiment this motivates

Use `template.md` as a starting point. Large result artifacts (plots,
model checkpoints) go under `experiments/<id>/outputs/`, which is
git-ignored — only the markdown writeup and small summary numbers are
committed.
