# Papiano Transcribe

Piano audio → MIDI transcription service for Papiano. Modal-hosted, T4 GPU,
scale-to-zero.

Model: [`piano_transcription_inference`](https://github.com/qiuqiangkong/piano_transcription_inference)
(Kong et al., arXiv:2010.01815, MIT). Checkpoint sourced from
[Zenodo](https://zenodo.org/record/4034264), cached on a Modal Volume.

## Layout

```
modal_app/app.py                     Serving app (GPU class + HTTP endpoint)
scripts/setup_checkpoint_volume.py   Seeds the checkpoint volume
scripts/test_endpoint.py             CLI test client
web/index.html                       Browser test client
requirements.txt                     Reference deps for local dev
```

## Setup

```bash
modal setup
modal run scripts/setup_checkpoint_volume.py
modal deploy modal_app/app.py
```

Base URL is printed on deploy: `https://<workspace>--papiano-transcribe-web.modal.run`.
API is `POST <base url>/transcribe`. CORS is open (`*`).

## API

```
POST /transcribe
{ "audio_base64": "<base64 audio bytes>" }

200 OK
{
  "notes": [{ "pitch": 60, "onset": 1.23, "offset": 1.81, "velocity": 87 }],
  "pedals": [{ "onset": 0.50, "offset": 2.10 }],
  "midi_base64": "..."
}
```

`pitch`: MIDI note number. `onset`/`offset`: seconds. `midi_base64`: standard
MIDI file with the same notes + sustain pedal (CC64).

## Testing

```bash
python scripts/test_endpoint.py --url <base url>/transcribe --audio song.wav --out output.mid
```

Or open `web/index.html` in a browser.

## Retraining

Write the new checkpoint into the `papiano-transcribe-checkpoints` volume and
redeploy.

## License

Model architecture, training, and checkpoint: Q. Kong, B. Li, X. Song, Y.
Wan, Y. Wang, "High-Resolution Piano Transcription with Pedals by Regressing
Onset and Offset Times," arXiv:2010.01815. MIT License.
