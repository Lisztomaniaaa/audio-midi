# Classical benchmark (Mutopia Project)

Chopin/Liszt-era piano MIDI, sourced from the [Mutopia Project](https://www.mutopiaproject.org)
— confirmed CC-licensed, safe for commercial use (see
[mutopiaproject.org/legal.html](https://www.mutopiaproject.org/legal.html)).
Compositions are all public domain (Chopin, d. 1849); the MIDI encodings
themselves are Mutopia's own CC-licensed engravings.

**Caveat**: these are LilyPond-generated "audio preview" files — mechanically
exact to the notated score, no human performance timing/rubato. Scores on
this set measure how well the transcriber handles precise, regular playing;
they're not a substitute for testing against real expressive performances
(see the earlier étude/jazz-ballad tests in `goals/02-bad-audio-robustness.md`
for that, which use different sourcing).

Not every requested opus is available — Mutopia's catalog is volunteer-
engraved and incomplete. This set is what's actually present:

| File | Piece |
|---|---|
| chopin_op10_no1.mid | Étude Op. 10 No. 1 |
| chopin_op10_no2.mid | Étude Op. 10 No. 2 |
| chopin_op10_no5.mid | Étude Op. 10 No. 5 ("Black Keys") |
| chopin_op10_no9.mid | Étude Op. 10 No. 9 |
| chopin_op10_no12.mid | Étude Op. 10 No. 12 ("Revolutionary") |
| chopin_op25_no1.mid | Étude Op. 25 No. 1 ("Aeolian Harp") |
| chopin_op25_no2.mid | Étude Op. 25 No. 2 |
| chopin_waltz_op64_no1.mid | Waltz Op. 64 No. 1 ("Minute Waltz") |
| chopin_ballade_no1.mid | Ballade No. 1 in G minor, Op. 23 |
| chopin_ballade_no4.mid | Ballade No. 4 in F minor, Op. 52 |

Run the full benchmark: `python scripts/run_classical_benchmark.py`
