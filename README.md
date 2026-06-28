# Papiano Transcribe

Piano audio → MIDI transcription service for Papiano. Modal-hosted, T4 GPU,
scale-to-zero.

Checkpoint cached on a Modal Volume, seeded once via `scripts/setup_checkpoint_volume.py`.

## Layout

```
modal_app/app.py                     Serving app (GPU class + HTTP endpoint)
scripts/setup_checkpoint_volume.py   Seeds the checkpoint volume
scripts/test_endpoint.py             CLI test client
requirements.txt                     Reference deps for local dev
```

## Setup

```bash
modal setup
modal run scripts/setup_checkpoint_volume.py
modal secret create papiano-api-key API_KEY="<shared-secret>"
modal deploy modal_app/app.py
```

Base URL is printed on deploy: `https://<workspace>--papiano-transcribe-web.modal.run`.
CORS is open (`*`).

## API

```
POST /transcribe
Header: X-API-Key: <shared secret>
{
  "audio_base64": "<base64 audio bytes>",
  "quantize": true,
  "tempo_hint": 72,
  "time_signature": "3/4"
}

200 OK
{
  "notes": [{ "pitch": 60, "onset": 1.23, "offset": 1.81, "velocity": 87 }],
  "pedals": [{ "onset": 0.50, "offset": 2.10 }],
  "tempo": 92.0,
  "time_signature": "4/4",
  "key": "Db major",
  "midi_base64": "...",
  "musicxml": "<?xml ...>"
}
```

`pitch`: MIDI note number. `onset`/`offset`: seconds, raw performance timing
synced to the audio. `tempo`: detected BPM. `key`: detected key signature.
`midi_base64`: standard MIDI file with the same notes + sustain pedal (CC64),
carrying the detected tempo and time signature. `musicxml`: an engraved
2-staff piano score (hands split at middle C, key + time signature, rhythm in
measures) — import into notation/arranger software (MuseScore, Sibelius,
Finale). `null` if engraving failed.

Beat + downbeat tracking uses Beat This! (neural); tempo, time signature, and
bar alignment are derived from it.

`tempo_hint` (optional, BPM) and `time_signature` (optional, e.g. `"3/4"`)
override the automatic detection. Auto beat tracking is unreliable on
expressive/rubato solo piano (it half/double-errors or finds no stable
pulse), so for that material a hint produces far cleaner bars — the same
reason klang.io asks the user to pick a tempo range. `tempo_hint` rescales
the detected beats to the nearest matching density (keeping rubato), or lays
down a uniform grid if no beats were found.

The MIDI carries a per-beat **tempo map** (not one flat tempo) so playback
follows the performance's rubato instead of sounding metronomic. Note
durations are also run through a hand "humanizer": within each hand a note is
released around the next note's onset, so a run/arpeggio doesn't hold its
first note through the whole figure — the sustain you hear is the pedal
(CC64), modelled separately.

Hands (for the humanizer and the two-staff score) are assigned by hand-span
limit + continuity — hands move smoothly and one hand spans at most ~a ninth
— rather than a fixed middle-C line, so a bass line that rises above middle C
stays in the left hand and wide chords split at their gap.

`quantize` (optional, default `true`): snap MIDI onsets/durations to a
1/16-note grid relative to the detected beats so it reads cleanly in notation
software. Set `false` to keep raw performance timing in the MIDI. The `notes`
array is always raw timing regardless. Time signature is assumed 4/4; tempo is
detected per request.

One shared API key, set as a Modal Secret. The Papiano backend holds the
same key and calls this endpoint server-to-server; per-user access, auth,
and limits are Papiano's responsibility, not this service's.

## Testing

```bash
python scripts/test_endpoint.py --url <base url>/transcribe --audio song.wav --out output.mid --key <api-key>
```

## Retraining

Write the new checkpoint into the `papiano-transcribe-checkpoints` volume and
redeploy.
