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

## Approach

Fine-tune the existing checkpoint per-genre or on a curated multi-genre
mix, rather than training from scratch. Needs aligned audio+MIDI pairs per
genre:

- **Classical/etude/ballade**: MAESTRO already covers this well; GiantMIDI-
  Piano is a large supplementary classical corpus if more depth is needed.
- **Pop**: POP909 has MIDI (arranger sheet-style) but needs audio rendered
  or paired with real recordings — check licensing per song.
- **Jazz**: transcription datasets are thin and often licensing-encumbered
  (real jazz recordings are heavily copyrighted); may need to lean on
  synthesized/rendered MIDI performances or a smaller curated set.
- **Ragtime**: public-domain era (most ragtime is pre-1928, safely public
  domain) — piano roll archives and clean public-domain recordings are a
  realistic source.
- **Bossa nova / anime**: no obvious existing aligned dataset; likely the
  hardest to source cleanly.

## Open questions before starting

- Per-genre or unified fine-tune? (Per-genre risks overfitting/catastrophic
  forgetting; unified needs balanced sampling across very unequal genre
  data sizes.)
- Licensing check per source, same class of problem as the Aria-AMT
  checkpoint issue — needs to be resolved *before* training, not after.
- What counts as "good enough" per genre — needs an eval set per genre to
  measure against, not just vibes.
