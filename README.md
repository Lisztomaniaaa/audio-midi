# audio-midi

Piano audio → MIDI transcription backend, served on [Modal](https://modal.com).

Model: high-resolution piano transcription
(PyTorch port of ByteDance's model,
[arXiv:2010.01815](https://arxiv.org/abs/2010.01815), MIT licensed). The
checkpoint is stored on a **Modal Volume** and loaded straight from disk at
serving time — the deployed app never talks to Hugging Face. Postprocessing
(regression outputs → discrete onset/offset/velocity/pedal events and MIDI
file generation) reuses the original author's MIT-licensed
[`piano_transcription_inference`](https://github.com/qiuqiangkong/piano_transcription_inference)
package.

## Repo layout

```
modal_app/app.py                     Modal app: loads the model from a Volume, exposes an HTTP endpoint
scripts/setup_checkpoint_volume.py   One-off: downloads the checkpoint and seeds the Modal Volume
scripts/duplicate_model.py           Optional: forks Genius-Society/piano_trans to your own HF account (for retraining/provenance)
scripts/test_endpoint.py             Calls a deployed endpoint with a local audio file
requirements.txt                     Reference deps (for local dev; Modal apps pin their own images)
```

## 1. (Optional) Duplicate the model to your own HF account

Only needed if you plan to retrain/fine-tune and want your own copy as the
base checkpoint. Skip straight to step 2 if you just want to serve the
original model.

```bash
pip install huggingface_hub
hf auth login   # or export HF_TOKEN=...
python scripts/duplicate_model.py --target-repo YOUR_USERNAME/piano_trans
```

This snapshots `Genius-Society/piano_trans`, uploads it to
`YOUR_USERNAME/piano_trans`, and appends a provenance/attribution section to
the README crediting the original model and ByteDance paper. The MIT
`LICENSE` file is copied as-is.

## 2. Seed the checkpoint volume

```bash
pip install modal
modal setup            # one-time auth
modal run scripts/setup_checkpoint_volume.py --hf-repo Genius-Society/piano_trans
# or, if you duplicated it in step 1:
modal run scripts/setup_checkpoint_volume.py --hf-repo YOUR_USERNAME/piano_trans
```

This downloads the `.pth` checkpoint once and stores it on a Modal Volume
named `piano-transcription-checkpoints`. The serving app reads from this
volume directly — no Hugging Face access at request time, and no repo bloat
(MuseScore AppImage, example MP3s, etc.) ever touches the serving image.

Re-run this script whenever you push a new/retrained checkpoint.

## 3. Deploy

```bash
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
inference app, or added alongside it). Once you have a new checkpoint:

1. (Optional) Push it to your HF repo for backup/provenance via
   `huggingface_hub.HfApi.upload_file`.
2. Run `modal run scripts/setup_checkpoint_volume.py --hf-repo
   YOUR_USERNAME/piano_trans` again to refresh the volume — or write the new
   `.pth` file straight into the volume from your training function.
3. Redeploy (or just let existing containers scale down; the next cold
   start picks up the new file on the volume).

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
