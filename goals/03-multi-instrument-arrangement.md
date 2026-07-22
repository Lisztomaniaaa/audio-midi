# Goal 3: Smart handling of non-piano instruments

**Scope, decided:** isolate the piano part from a mixed recording (piano +
vocals/drums/other instruments), then transcribe just that — not generating
a piano arrangement for recordings that have no piano at all (that's a much
harder generative-arrangement problem and explicitly out of scope for now).

## Status: shipped (opt-in), but measured to make results WORSE — don't use yet

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

**Tested — negative result.** Built a synthetic multi-instrument test:
took the jazz-ballad MIDI (piano-only reference), added a programmatic
walking bass line + simple drum pattern on separate MIDI channels, rendered
the full mix with FluidSynth, and compared transcribing it directly against
transcribing after `separate_piano`:

|                        | direct (mixed audio) | after separate_piano |
|------------------------|----------------------|-----------------------|
| Notes estimated (ref=323) | 389                | **505**               |
| Onset+pitch F1         | 0.803                 | **0.662**              |

Separation made it *worse*, not better — more spurious notes, not fewer.
Bass and drums have spectral/rhythmic character different enough from piano
that the transcriber already handles the raw mix reasonably (0.80 F1);
Spleeter's separation artifacts (bleed, spectral holes, phasing) apparently
confuse the transcriber more than the original bass/drums did. **Don't
recommend using `separate_piano` until this is root-caused** — leaving it
in the API as opt-in (so nothing currently using it breaks) but the finding
here is: for now, transcribing the mixed audio directly is the better
default, which is the opposite of what this goal set out to build.

Possible next steps, not yet tried: check what the isolated piano stem
actually sounds like/looks like spectrally (is Spleeter's separation itself
bad, or is the transcriber unusually sensitive to its specific artifacts);
try feeding the separated stem through the same audio-quality preprocessing
other paths use; consider whether a different separation tool/approach
would fare better, now that we have a concrete regression test to check
any candidate against before adopting it.

## Open questions

- Root cause the regression above before any further work here — is it
  Spleeter's separation quality itself, or something in how the separated
  stem interacts with the transcriber's own preprocessing?
- Detection: right now `separate_piano` is caller-specified (Papiano decides
  when to send it), not auto-detected. Moot until separation is shown to
  actually help.
- Cost/latency: this runs a second model (CPU-based TensorFlow inference)
  before the GPU transcription step — worth measuring actual added latency
  under real load, though moot if the feature isn't recommended for use.
