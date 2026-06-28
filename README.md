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
{ "audio_base64": "<base64 audio bytes>", "quantize": true }

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
bar alignment are derived from it, with librosa as a fallback.

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
