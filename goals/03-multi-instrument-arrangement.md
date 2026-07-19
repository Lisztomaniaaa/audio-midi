# Goal 3: Smart handling of non-piano instruments

**Scope, decided:** isolate the piano part from a mixed recording (piano +
vocals/drums/other instruments), then transcribe just that — not generating
a piano arrangement for recordings that have no piano at all (that's a much
harder generative-arrangement problem and explicitly out of scope for now).

## Approach

1. Source separation stage ahead of the existing transcription pipeline,
   isolating the piano stem. Candidates: Demucs (MIT-licensed, Meta — best
   general-purpose separation quality) or Spleeter (MIT-licensed, Deezer —
   lighter weight). Neither ships a piano-specific stem out of the box
   (usual stems are vocals/drums/bass/other), so "other" would need to
   double as the piano stem, or the separator needs fine-tuning/adapting
   for a dedicated piano stem.
2. Existing `modal_app/app.py` pipeline runs unchanged on the isolated
   stem — no changes needed downstream of separation.
3. Evaluate isolation quality's effect on transcription accuracy — stem
   separation isn't perfect and introduces its own artifacts (bleed,
   phasing) that could reintroduce the "bad audio" problem from goal 2.
   Worth checking whether the same onset/frame threshold tuning from goal 2
   helps here too.

## Open questions

- Detection: does every request need a "does this contain non-piano
  instruments" check first (skip separation when it's already solo piano,
  to avoid wasted compute + separation artifacts on clean input)?
- Licensing: confirm Demucs/Spleeter checkpoint licenses commercially
  before depending on either (same class of check as the Aria-AMT issue —
  don't assume, verify).
