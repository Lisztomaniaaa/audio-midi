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

## First real ground-truth result (Chopin Étude Op. 25 No. 1, "Aeolian Harp")

Scored against an aligned MP3+MIDI pair the user provided (chosen
deliberately for its heavy, continuous sustain pedal — a worst-case stress
test). Using `scripts/eval_transcription.py` against the currently deployed
model:

- **Onset+pitch correct** (is this the right note, roughly the right time,
  ignoring duration): F1 = **0.034**. Low — this piece's fast, wide-register
  arpeggios are genuinely hard for the current model, not just a duration
  problem. Manual inspection of a 1-second window confirmed real pitch
  errors, not just an eval bug (though we did find and fix one — see below).
- **Onset+pitch+offset correct** (also duration): F1 = 0.0076 at the
  deployed `DECAY_THRESHOLD_RATIO=0.20`.

**Found and fixed a real bug in the eval script**: `mir_eval`'s
`Onset_Precision/Recall/F-measure` keys match on time only, ignoring pitch
entirely — a first pass reported these as "onset accuracy" and got 0.63,
which is meaningless for judging transcription quality (a dense polyphonic
piece scores deceptively high there just from onset density). The
pitch-aware numbers above (`*_no_offset` keys) are what actually matters,
and are far lower. Fixed in the script; don't trust the plain `Onset_*` keys
for anything user-facing.

**Explored, not shipped**: swept `DECAY_THRESHOLD_RATIO` from 0.20 up to
0.99 against this one file — 0.85 roughly triples the onset+pitch+offset F1
(0.0076 → 0.022) and brings median note duration much closer to the
reference (0.18s → ~0.05s vs reference's ~0.10s). **Deliberately not
deployed as the new default**: this is one deliberately-adversarial piece;
retuning a global production parameter to it risks regressing normal,
less-pedaled audio that the current 0.20 default may already suit
reasonably. Needs a small eval set spanning easy *and* hard cases before
committing to a new default — not a decision to make from n=1.

**⚠️ Correction — the 0.034 above was measuring a data mismatch, not (mainly)
model quality.** `mido.MidiFile(...).length` on the reference MIDI is 125.8s;
the MP3 is 172.0s — a 37% duration difference, meaning the MIDI and MP3 are
different performances/tempos of the same piece, not an aligned pair. To
check, we rendered audio directly from the reference MIDI itself (FluidSynth
+ a GM soundfont — exact alignment by construction, see
`scripts/render_midi_audio.py`) and re-scored against that:

|                          | vs. mismatched MP3 | vs. MIDI-rendered audio (aligned) |
|--------------------------|---------------------|-------------------------------------|
| Onset+pitch F1           | 0.034               | **0.339**                            |
| Onset+pitch+offset F1    | 0.0076              | **0.240**                            |

~10x better once the comparison is actually fair. The model is meaningfully
better than the first pass suggested — this piece is still hard (0.34 isn't
great), but it's not the near-total failure the mismatched comparison
implied. **Lesson: verify audio/MIDI duration match before trusting any
score from a pair someone provides** — a plausible-looking pair can silently
be two different performances.

**Bigger picture, revised**: real transcription quality on this hard piece
is ~0.34 onset+pitch F1, not 0.034 — worse than the field's reported
MAESTRO benchmarks (~0.97) as expected for a fast/wide-register/heavily-
pedaled piece, but not evidence of a severe architectural ceiling on its
own. The Aria-AMT (seq2seq) vs current-model (CRNN) architecture gap
discussed elsewhere may still matter for this kind of material, but this
particular data point no longer supports as strong a claim as first
thought — needs a properly aligned eval set (multiple pieces) before
drawing that conclusion with confidence.

## Open questions

- Need more aligned audio+MIDI pairs (easy AND hard cases) before touching
  any global threshold default again — one adversarial sample isn't enough
  to safely retune production.
- Should degraded-audio detection gate the response (warn + still return
  best-effort) or block it (refuse to transcribe below some quality floor)?
