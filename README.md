# audio-midi

Piano audio → MIDI transcription backend, served on [Modal](https://modal.com).

Model: a personal Hugging Face duplicate of
[`Genius-Society/piano_trans`](https://huggingface.co/Genius-Society/piano_trans)
(PyTorch port of ByteDance's high-resolution piano transcription model,
[arXiv:2010.01815](https://arxiv.org/abs/2010.01815), MIT licensed), loaded
via `timm.create_model("hf_hub:<your-repo>", pretrained=True)`. Postprocessing
(regression outputs → discrete onset/offset/velocity/pedal events and MIDI
file generation) reuses the original author's MIT-licensed
[`piano_transcription_inference`](https://github.com/qiuqiangkong/piano_transcription_inference)
package.

## Repo layout

```
modal_app/app.py        Modal app: loads the model, exposes an HTTP endpoint
scripts/duplicate_model.py  One-off: forks Genius-Society/piano_trans to your HF account
scripts/test_endpoint.py    Calls a deployed endpoint with a local audio file
requirements.txt        Reference deps (for local dev; Modal app pins its own image)
```

## 1. Duplicate the model to your HF account

```bash
pip install huggingface_hub
huggingface-cli login   # or export HF_TOKEN=...
python scripts/duplicate_model.py --target-repo YOUR_USERNAME/piano_trans
```

This snapshots `Genius-Society/piano_trans`, uploads it to
`YOUR_USERNAME/piano_trans`, and appends a provenance/attribution section to
the README crediting the original model and ByteDance paper. The MIT
`LICENSE` file is copied as-is. This new repo is the base for any future
retraining — push new checkpoints here.

## 2. Configure the Modal app

Edit `HF_MODEL_REPO` in `modal_app/app.py` (or set it via env var at deploy
time) to point at your duplicated repo, e.g. `YOUR_USERNAME/piano_trans`.

If your HF repo is private, create a Modal secret with your HF token:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxx
```

and reference it in `app.py`'s `@app.cls(...)` decorator with
`secrets=[modal.Secret.from_name("huggingface-secret")]`.

## 3. Deploy

```bash
pip install modal
modal setup            # one-time auth
modal deploy modal_app/app.py
```

Modal prints the HTTPS endpoint URL for `transcribe`, e.g.:

```
https://YOUR_WORKSPACE--piano-transcription-transcribe.modal.run
```

The model loads onto a T4 GPU on first request (cold start) and the
container scales to zero after ~2 minutes idle (`scaledown_window=120` in
`app.py`), so there's no cost while unused.

## 4. Test with a local audio file

```bash
pip install requests
python scripts/test_endpoint.py \
    --url https://YOUR_WORKSPACE--piano-transcription-transcribe.modal.run \
    --audio path/to/song.wav \
    --out output.mid
```

Or with curl:

```bash
AUDIO_B64=$(base64 -w0 song.wav)
curl -X POST "$URL" -H "Content-Type: application/json" \
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
inference app, or added alongside it), and push resulting checkpoints back to
your `YOUR_USERNAME/piano_trans` HF repo with `huggingface_hub.HfApi.upload_file`
or `upload_folder` — the same mechanism `scripts/duplicate_model.py` uses for
the initial fork. The inference app in `modal_app/app.py` will pick up new
weights automatically on next cold start (it always loads `pretrained=True`
from the HF repo).

## Using from another project (e.g. a piano visualizer)

Call the deployed HTTPS endpoint directly with a base64-encoded audio file as
shown above — no Python/Modal SDK needed. This makes it usable as a drop-in
server-side transcription option from any client (web app, mobile, etc.),
without touching that project's existing client-side transcription path.

## License / attribution

This repo's code is provided as-is. The model architecture and
postprocessing algorithm originate from:

> Q. Kong, B. Li, X. Song, Y. Wan, Y. Wang, "High-Resolution Piano
> Transcription with Pedals by Regressing Onset and Offset Times," 2020.
> arXiv:2010.01815. MIT License.

and the Hugging Face port:

> [Genius-Society/piano_trans](https://huggingface.co/Genius-Society/piano_trans), MIT License.
