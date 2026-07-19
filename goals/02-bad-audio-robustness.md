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

1. **Done**: lowered `onset_threshold` (0.3 → 0.15) and `frame_threshold`
   (0.1 → 0.05) in `modal_app/app.py` — trades precision for recall, catches
   more/weaker onsets in noisy audio at the cost of more false positives.
   Deployed; values are a first guess, not yet tuned against a labeled eval
   set (see open questions).
2. **Real fix, needs training**: noise-augmented fine-tuning — inject
   synthetic noise/reverb/compression artifacts during training (the
   approach arXiv:2410.14122 validates). This is the actual ML work; the
   threshold tweak is a stopgap. **Do not use MAESTRO audio directly as the
   base for this** — it's CC-BY-NC-SA 4.0, not safe for us to train on (see
   the correction in `01-genre-coverage.md`). Needs a cleanly-licensed base
   corpus first (e.g. synthetic audio rendered from public-domain/licensed
   MIDI), same constraint as goal 1.

## Open questions

- What threshold value is the right default trade-off? Needs a labeled
  "bad audio" eval set to tune against, not guessing.
- Should degraded-audio detection gate the response (warn + still return
  best-effort) or block it (refuse to transcribe below some quality floor)?
