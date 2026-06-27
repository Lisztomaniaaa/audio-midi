# audio-midi

Piano audio → MIDI transcription backend, served on [Modal](https://modal.com).

Model: high-resolution piano transcription
(PyTorch port of ByteDance's model,
[arXiv:2010.01815](https://arxiv.org/abs/2010.01815), MIT licensed). The
checkpoint is downloaded once from the original author's
[Zenodo release](https://zenodo.org/record/4034264), stored on a **Modal
Volume**, and loaded straight from disk at serving time — no Hugging Face
involved anywhere in this pipeline. Postprocessing (regression outputs →
discrete onset/offset/velocity/pedal events and MIDI file generation) reuses
the original author's MIT-licensed
[`piano_transcription_inference`](https://github.com/qiuqiangkong/piano_transcription_inference)
package.

## Repo layout

```
modal_app/app.py                     Modal app: loads the model from a Volume, exposes an HTTP endpoint
scripts/setup_checkpoint_volume.py   One-off: downloads the checkpoint from Zenodo and seeds the Modal Volume
scripts/test_endpoint.py             Calls a deployed endpoint with a local audio file
requirements.txt                     Reference deps (for local dev; Modal apps pin their own images)
```

## 1. Seed the checkpoint volume

```bash
pip install modal
modal setup            # one-time auth
modal run scripts/setup_checkpoint_volume.py
```

This downloads the `.pth` checkpoint once from Zenodo and stores it on a
Modal Volume named `piano-transcription-checkpoints`. The serving app reads
from this volume directly — no external network access at request time.

Re-run this script whenever you want to push a new/retrained checkpoint
(swap the download step for your own training output).

## 2. Deploy

```bash
modal deploy modal_app/app.py
```

Modal prints the HTTPS base URL, e.g.:

```
https://YOUR_WORKSPACE--piano-transcription-web.modal.run
```

POST to `<base url>/transcribe`. CORS is wide open, so it can be called
directly from browser JS (see `web/index.html`).

The model loads onto a T4 GPU on first request (cold start) and the
container scales to zero after ~2 minutes idle (`scaledown_window=120` in
`app.py`), so there's no cost while unused.

## 3. Test

**Browser:** open `web/index.html` directly in a browser, paste your
`<base url>/transcribe` URL, pick an audio file, and click "Transkripsi".

**CLI:**

```bash
pip install requests
python scripts/test_endpoint.py \
    --url https://YOUR_WORKSPACE--piano-transcription-web.modal.run/transcribe \
    --audio path/to/song.wav \
    --out output.mid
```

Or with curl:

```bash
AUDIO_B64=$(base64 -w0 song.wav)
curl -X POST "$URL/transcribe" -H "Content-Type: application/json" \
    -d "{\"audio_base64\": \"$AUDIO_B64\"}" -o response.json
```

### Response shape

```json
{
  "notes": [
    {"pitch": 60, "onset": 1.23, "offset": 1.81, "velocity": 87}
  ],
  "pedals": [
    {"onset": 0.50, "offset": 2.10}
  ],
  "midi_base64": "..."
}
```

`pitch` is a MIDI note number (21–108, piano range). `onset`/`offset` are in
seconds. `midi_base64` is a ready-to-save standard MIDI file containing the
same note and sustain-pedal (CC64) events.

## Retraining later

Run training on Modal with an on-demand GPU function (separate from this
inference app, or added alongside it), then write the resulting checkpoint
straight into the `piano-transcription-checkpoints` volume (e.g. from the
training function itself, or by adapting `setup_checkpoint_volume.py` to
copy your new `.pth` file instead of downloading from Zenodo). Redeploy (or
just let existing containers scale down) and the next cold start picks up
the new weights.

## Using from another project (e.g. a piano visualizer)

Call the deployed HTTPS endpoint directly with a base64-encoded audio file as
shown above — no Python/Modal SDK needed. This makes it usable as a drop-in
server-side transcription option from any client (web app, mobile, etc.).

## License / attribution

This repo's code is provided as-is. The model architecture, training, and
checkpoint originate from:

> Q. Kong, B. Li, X. Song, Y. Wan, Y. Wang, "High-Resolution Piano
> Transcription with Pedals by Regressing Onset and Offset Times," 2020.
> arXiv:2010.01815. MIT License.
> Checkpoint release: https://zenodo.org/record/4034264
