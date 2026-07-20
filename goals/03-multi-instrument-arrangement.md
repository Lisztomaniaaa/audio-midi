# Goal 3: Smart handling of non-piano instruments

**Scope, decided:** isolate the piano part from a mixed recording (piano +
vocals/drums/other instruments), then transcribe just that — not generating
a piano arrangement for recordings that have no piano at all (that's a much
harder generative-arrangement problem and explicitly out of scope for now).

## Status: shipped (opt-in), quality not yet validated on real audio

Implemented in `modal_app/app.py` as `PianoSeparator`, a Modal class in its
own image (Spleeter needs TensorFlow, which conflicts with the main
PyTorch-based image, so it runs as a separate container). The `/transcribe`
endpoint takes an optional `separate_piano` flag — when set, the piano stem
is isolated first, then handed to the existing transcription pipeline
unchanged.

**Licensing, verified before use** (learned from the Aria-AMT and MAESTRO
mistakes — checked code and checkpoint separately, from primary sources):

- Demucs (Meta) was **rejected**: checkpoints trained partly on MUSDB18/HQ,
  which is largely CC-BY-NC-SA with no explicit relicensing statement from
  Meta — same unresolved risk pattern as MAESTRO. Its maintainer also says
  piano separation quality is poor ("not working great").
- **Spleeter (Deezer) was chosen**: code MIT, and — critically — the
  authors' own JOSS paper states the released weights are MIT too, trained
  on Deezer's proprietary internal data (MUSDB18 used only for benchmarking,
  not training). No NC taint, and it's a direct statement from the actual
  rights holder, the same pattern that made the ByteDance transcription
  checkpoint safe to use. Its 5-stem model has a dedicated `piano` output
  (vocals/drums/bass/piano/other) — Demucs's standard stems don't.

**Not yet validated**: Spleeter's own paper doesn't publish an SDR number
for the piano stem specifically — separation quality on real mixed
recordings (bleed, artifacts) hasn't been measured, only smoke-tested for
"does the pipeline run without crashing." Test on real multi-instrument
piano recordings before relying on this in production.

## Open questions

- Detection: right now `separate_piano` is caller-specified (Papiano decides
  when to send it), not auto-detected. An automatic "does this need
  separation" check would avoid the added latency/artifact risk on already-
  clean solo piano audio, but needs its own instrument-detection step.
- Quality: does separation-then-transcribe actually improve results over
  just transcribing the mixed audio directly, or do Spleeter's artifacts
  cancel out the gain? Needs a real before/after comparison on mixed
  recordings, not assumed.
- Cost/latency: this runs a second model (CPU-based TensorFlow inference)
  before the GPU transcription step — worth measuring actual added latency
  under real load.
