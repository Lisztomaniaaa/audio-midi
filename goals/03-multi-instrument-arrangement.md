# Goal 3: Smart handling of non-piano instruments

Scope needs to be pinned down first — "smart arrangement" could mean two
quite different things:

**(A) Isolate the piano part from a mixed recording, then transcribe just
that.** E.g. a song with piano + vocals + drums — pull out the piano stem,
run the existing pipeline on it. This is the more tractable interpretation:
source separation (Demucs, MIT-licensed, Meta; or Spleeter, MIT-licensed,
Deezer) already does instrument-stem separation reasonably well, could
likely be fine-tuned/adapted specifically for piano isolation. Pipeline
becomes: separate → transcribe the piano stem with the existing model →
same output as today.

**(B) Generate a piano arrangement/reduction of a full band or orchestral
recording that has no piano part at all** — condense a full mix into an
idiomatic 2-hand piano reduction (like a "piano cover" arranger would).
This is a much harder, more open-ended research problem — it's generative
arrangement, not transcription, and needs its own model/approach entirely
(closer to what a human arranger does than what an AMT model does).

## Open question (blocks scoping the rest of this doc)

Which of these is actually wanted — clean up input audio that has other
instruments alongside piano (A), or turn a non-piano recording into a piano
arrangement (B)? They're different products. Confirm before estimating
effort or picking an approach.

## If (A): rough approach

1. Source separation stage (Demucs or similar) ahead of the existing
   transcription pipeline, isolating the piano stem.
2. Existing `modal_app/app.py` pipeline runs unchanged on the isolated
   stem.
3. Evaluate isolation quality's effect on transcription accuracy — stem
   separation isn't perfect and introduces its own artifacts (bleed,
   phasing) that could reintroduce the "bad audio" problem from goal 2.

## If (B): rough approach

Needs its own research spike — no existing off-the-shelf model does
audio-to-piano-reduction end-to-end well. Likely a multi-stage pipeline
(full transcription of all instruments → harmonic/melodic reduction logic
→ piano-idiomatic voicing) rather than a single model.
