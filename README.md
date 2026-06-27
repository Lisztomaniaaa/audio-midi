# Papiano Transcribe

Piano audio → MIDI transcription service for Papiano. Modal-hosted, T4 GPU,
scale-to-zero.

Checkpoint cached on a Modal Volume, seeded from Zenodo.

## Layout

```
modal_app/app.py                     Serving app (GPU class + HTTP endpoint)
scripts/setup_checkpoint_volume.py   Seeds the checkpoint volume
scripts/test_endpoint.py             CLI test client
web/index.html                       Browser test client
web/admin.html                       Admin: generate/revoke keys, approve requests
web/request-access.html              Public form to request an API key
requirements.txt                     Reference deps for local dev
```

## Setup

```bash
modal setup
modal run scripts/setup_checkpoint_volume.py
modal secret create papiano-admin-password ADMIN_PASSWORD="<your-password>"
modal deploy modal_app/app.py
```

Base URL is printed on deploy: `https://<workspace>--papiano-transcribe-web.modal.run`.
CORS is open (`*`).

## API

```
POST /transcribe
Header: X-API-Key: <key>
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

API keys are stored server-side in a Modal Dict, never in this repo or in
any frontend. Issue/revoke them from `web/admin.html`.

```
POST /request-access            { name, email, reason } -> request_id, pending review
GET  /admin/requests            X-Admin-Password header -> pending requests
POST /admin/requests/{id}/approve   -> { api_key }
POST /admin/requests/{id}/reject
POST /admin/keys                { label } -> { api_key }
GET  /admin/keys                -> all keys + status
POST /admin/keys/{key}/revoke
```

`web/admin.html`: connect with base URL + admin password, generate/revoke
keys, approve/reject pending requests.

`web/request-access.html`: public form for requesting a key; submissions
sit in `/admin/requests` until approved.

## Testing

```bash
python scripts/test_endpoint.py --url <base url>/transcribe --audio song.wav --out output.mid --key <api-key>
```

Or open `web/index.html` in a browser (enter the API key in the form).

## Retraining

Write the new checkpoint into the `papiano-transcribe-checkpoints` volume and
redeploy.
