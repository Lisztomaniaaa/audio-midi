# Goal 2: Bad audio detection & robustness

Two distinct problems: detecting that input audio is low-quality, and
actually transcribing it better anyway.

## Detection (short-term, tractable)

Add an audio-quality check before/alongside transcription and surface it in
the API response (e.g. `"audio_quality": "low"` + a reason), so Papiano can
warn the user or suggest a better recording instead of silently returning a
bad transcription. Candidate signals:

- Estimated SNR / noise floor
- Clipping (samples pinned at ±1.0)
- Effective bandwidth (low-bitrate MP3s cut high frequencies)
- Sample rate / mono-downmix artifacts

This is a heuristic-engineering task, not a training task — could ship
independently of the model-robustness work below.

## Robustness (needs training)

Confirmed via research: this checkpoint (Kong et al., MAESTRO-only) has
measured degradation on noisy input — roughly 5% relative F1 drop at 12dB
SNR, 10% at 9dB SNR (see arXiv:2410.14122, which benchmarks noise-injection
augmentation on this exact model). Two levers:

1. **Cheap, already-available**: lower `onset_threshold` (default 0.3) and
   `frame_threshold` (default 0.1) — trades precision for recall, catches
   more/weaker onsets in noisy audio at the cost of more false positives.
   Not yet wired up as a configurable param in `modal_app/app.py` — that's
   the first, low-effort step.
2. **Real fix, needs training**: noise-augmented fine-tuning — inject
   synthetic noise/reverb/compression artifacts into MAESTRO during
   training (the approach the above paper validates). This is the actual
   ML work; the threshold tweak is a stopgap.

## Open questions

- What threshold value is the right default trade-off? Needs a labeled
  "bad audio" eval set to tune against, not guessing.
- Should degraded-audio detection gate the response (warn + still return
  best-effort) or block it (refuse to transcribe below some quality floor)?
