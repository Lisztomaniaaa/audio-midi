# Transcription Quality Roadmap

Three open initiatives to push transcription quality toward "as close to
perfect as possible." Each is a real ML/engineering effort (data, training,
evaluation), not a quick config change — see the individual files for scope,
approach options, and open questions.

1. [`01-genre-coverage.md`](01-genre-coverage.md) — jazz, classical, bossa
   nova, ragtime, etude, anime, ballade, pop
2. [`02-bad-audio-robustness.md`](02-bad-audio-robustness.md) — detecting
   and handling low-quality source audio
3. [`03-multi-instrument-arrangement.md`](03-multi-instrument-arrangement.md)
   — smart handling when the recording isn't solo piano

Current baseline: `piano_transcription_inference` (Kong et al./ByteDance),
trained only on MAESTRO (studio solo piano). That's the starting point all
three goals push against.
