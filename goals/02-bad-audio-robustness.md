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

## Second ground-truth result (jazz piece, MIDI rendered to audio ourselves)

Tested against a user-provided MIDI ("Ballad of the Jazz Cafe", trimmed to
60s), rendered to audio via `scripts/render_midi_audio.py` — exact alignment
by construction, so this result isn't confounded by the mismatch problem
above. Two concrete, user-reported issues investigated and fixed:

**"Ghost" false-positive notes.** The earlier `onset_threshold=0.15` (goal:
better bad-audio recall) was tested here on *clean* synthesized audio: 18
false-positive notes out of 331 detected. Swept the threshold back up:

| onset_threshold | false positives | recall |
|---|---|---|
| 0.15 (was deployed) | 18 | 0.969 |
| 0.20 | 11 | 0.963 |
| 0.25 | 10 | 0.960 |
| **0.30 (library default)** | **6** | **0.960** |

0.30 cuts false positives 3x for under 1% recall cost — reverted both
`ONSET_THRESHOLD` and `FRAME_THRESHOLD` to the library's own defaults; the
earlier lowering was never validated against real audio and turned out to
be a bad trade on typical (non-degraded) input.

**Unnaturally long notes.** Manually confirmed several of the worst-overshoot
notes had *no sustain pedal active* at all — that pitch's CQT energy was
bleed from a different overlapping note/harmonic, not this note's own decay,
and `_refine_note_offsets`'s decay search is bounded by the model's own
(already too-long) predicted offset, so it had no way to catch this. Added
`MAX_NOTE_DURATION_NO_PEDAL_S` (1.0s) and a looser `MAX_NOTE_DURATION_PEDAL_S`
(2.0s) as hard ceilings, using the pedal on/off track we already compute.
Explicitly **not** tuned to match this file's reference durations exactly —
per direct user instruction, the target is a plausible/humanlike duration,
not literal ground truth (a real MIDI can itself encode a duration no human
pianist would produce).

Result on this file:

|                          | before | after |
|--------------------------|--------|-------|
| Onset+pitch F1           | 0.957  | **0.970** (precision 0.946→0.981) |
| Onset+pitch+offset F1    | 0.291  | **0.341** |
| Median duration error    | +62ms  | **+48ms** |
| p95 duration error       | +494ms | **+444ms** |

Residual large-error tail is now specifically in pedal-*active* cases (worst
still ~1.8s vs a 0.7s reference) — deliberately left alone rather than
tightening `MAX_NOTE_DURATION_PEDAL_S` further, since that risks clipping
genuinely long pedaled notes and we only have this one file's evidence.

**Regression check on the Chopin étude** (rendered from its own MIDI too, so
now a second cleanly-aligned data point): scores essentially unchanged
before/after these fixes (onset+pitch F1 0.339→0.338, onset+pitch+offset
0.240→0.238). Expected and reassuring — this piece's notes are already short
and fast (dense arpeggios), so the duration caps rarely trigger there; the
fixes target a different failure mode (held/long notes) without regressing
this one. The étude's low absolute score (~0.34) holding steady also
reconfirms that piece's difficulty is a genuine onset/pitch-detection
limitation (fast, wide-register arpeggios), not a duration/threshold
artifact — consistent with the Aria-AMT (seq2seq) vs current-model (CRNN)
architecture gap discussed elsewhere.

## Classical benchmark (10 pieces, `eval_data/classical_mutopia/`)

Built a proper repeatable benchmark: 10 Chopin pieces (Études Op. 10 Nos.
1/2/5/9/12, Op. 25 Nos. 1/2, Waltz Op. 64 No. 1, Ballades Nos. 1 and 4),
sourced from Mutopia (CC-licensed, public-domain compositions — see that
folder's README for the caveat that these are mechanically-exact engravings,
not expressive performances, so don't over-index on these being harder/
easier than real recordings). Run via `scripts/run_classical_benchmark.py`.

| Metric (average across 10 pieces) | Score |
|---|---|
| Onset+pitch F1 | **0.958** |
| Onset+pitch+offset F1 | 0.796 |
| Onset+pitch+offset+velocity F1 | **0.603** |

Onset/pitch detection is consistently strong (0.877-0.996 across all 10 —
classical repertoire, unlike the earlier hard étude test, is comfortably in
this model's wheelhouse when the reference is mechanically precise).
Duration is decent (0.69-0.93). **Velocity is the standout weak point** and
inconsistent — two pieces (Op. 10 No. 9: 0.277, Op. 25 No. 1: 0.412) score
far below the rest (0.78-0.93) despite similar onset/duration accuracy. Not
yet investigated why those two specifically — candidate hypothesis (untested):
wide dynamic range or heavier pedal use obscuring the true attack velocity,
but this needs actual investigation before acting on it.

## Open questions

- Need more aligned audio+MIDI pairs (easy AND hard cases) before touching
  any global threshold default again — two adversarial samples isn't enough
  to safely retune production, and both fixes so far are validated on jazz/
  classical only.
- Is `MAX_NOTE_DURATION_PEDAL_S=2.0` actually the right ceiling, or should it
  vary by tempo/register? Needs more pedal-heavy examples to know.
- **New**: why does velocity accuracy swing so hard between pieces (0.28 to
  0.93) that otherwise score similarly on onset/duration? Worth root-causing
  before touching anything — could be a real, fixable estimation issue.
- Should degraded-audio detection gate the response (warn + still return
  best-effort) or block it (refuse to transcribe below some quality floor)?
