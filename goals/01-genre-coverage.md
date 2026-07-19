# Goal 1: Genre coverage

Target genres: jazz, classical, bossa nova, ragtime, etude, anime, ballade, pop.

## Why the baseline is weak here

The current checkpoint is trained exclusively on MAESTRO — ~200 hours of
solo piano, overwhelmingly classical competition repertoire (etudes,
sonatas, romantic-era pieces). It has effectively never seen:

- Jazz voicings, swung rhythm, walking bass, comping
- Bossa nova syncopation
- Ragtime stride patterns (though rhythmically closer to classical-era
  piano than jazz is)
- Anime/game piano arrangements (fast, dense, often pop-harmony-derived)
- Pop piano (lead sheet-style voicings, simpler textures)

"Ballade" and "etude" are already well inside MAESTRO's distribution
(they're classical forms/competition staples), so those two are closer to
"verify it's actually good" than "needs new training data."

## ⚠️ Correction: MAESTRO is NOT clean for us to train/fine-tune on directly

Earlier draft of this doc said MAESTRO could anchor classical/etude/ballade
fine-tuning. That was wrong — verified directly against the official
Magenta page (magenta.tensorflow.org/datasets/maestro): MAESTRO is
**CC-BY-NC-SA 4.0**, not CC-BY. The currently-deployed checkpoint (Kong et
al./ByteDance) was trained on MAESTRO too, but *they* — as the checkpoint's
creator — chose to release *their resulting weights* under CC-BY 4.0; that's
their license grant on their own output, not a statement that MAESTRO
itself is safe for anyone to train on. If we fine-tune further using
MAESTRO audio directly ourselves, we're the ones using NC-licensed data
commercially — same exposure as the original Aria-AMT problem. Don't use
MAESTRO audio directly for any new training/fine-tuning here.

## Approach

Fine-tune the existing checkpoint per-genre or on a curated multi-genre
mix, rather than training from scratch. Needs aligned audio+MIDI pairs per
genre — and **every source below needs its own license verified against its
official hosting page before use**, the same diligence MAESTRO just failed:

- **Classical/etude/ballade**: GiantMIDI-Piano is a candidate — license not
  yet verified, check before use. Public-domain-era classical recordings
  (pre-1928 or otherwise confirmed PD) + a virtual piano renderer (e.g.
  Pianoteq, or a well-licensed soundfont) is a fallback that sidesteps
  recording-rights questions entirely, since you're synthesizing audio from
  score/MIDI rather than using someone else's recording.
- **Pop**: POP909 has MIDI (arranger sheet-style) — license not yet
  verified, check before use; would still need audio (rendered or paired)
  either way.
- **Jazz**: transcription datasets are thin and often licensing-encumbered
  (real jazz recordings are heavily copyrighted); synthesized/rendered MIDI
  performances from cleanly-licensed jazz MIDI transcriptions is likely
  the safer path over sourcing real recordings.
- **Ragtime**: public-domain era (most ragtime is pre-1928) — piano roll
  archives and confirmed public-domain recordings are a realistic source,
  but confirm PD status per recording, not just per composition (a modern
  recording of a PD piece is not itself PD).
- **Bossa nova / anime**: no obvious existing aligned dataset; likely the
  hardest to source cleanly — probably synthetic-audio-from-MIDI is the
  only realistic clean path here too.

## Open questions before starting

- Per-genre or unified fine-tune? (Per-genre risks overfitting/catastrophic
  forgetting; unified needs balanced sampling across very unequal genre
  data sizes.)
- Licensing check per source is now the *first* step, not a formality —
  verify directly against each dataset's own official page, the way MAESTRO
  should have been checked from the start.
- What counts as "good enough" per genre — needs an eval set per genre to
  measure against, not just vibes.
