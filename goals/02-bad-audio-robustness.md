# Goal 2: Bad audio detection & robustness

Two distinct problems: detecting that input audio is low-quality, and
actually transcribing it better anyway.

## Detection — done

Shipped in `modal_app/app.py` (`_assess_audio_quality`): returns
`audio_quality` in the API response with `level` (`good`/`low`), `issues`
(`clipping`, `low_snr`, `narrow_bandwidth`), plus raw `snr_db`,
`bandwidth_hz`, `clipping_ratio`. Doesn't block the response — best-effort
transcription still runs, this is a signal for the caller.

Known limitation, found during testing: the SNR heuristic (90th vs 10th
percentile frame RMS) assumes quiet passages exist to sample a noise floor
from. A continuously loud, non-decaying signal (e.g. sustain pedal held
throughout, or a synthetic sustained tone) has no quiet frames and can read
as false-positive `low_snr` even when actually clean. Real piano recordings
almost always have natural decay/silence between phrases, so this should be
rare in practice, but it's not rigorously validated against real "bad
audio" samples yet — thresholds are first guesses.

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
