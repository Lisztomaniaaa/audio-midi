"""Papiano Transcribe — piano audio -> MIDI, served on Modal.

First-time setup: modal run scripts/setup_checkpoint_volume.py
Deploy: modal deploy modal_app/app.py
"""

import base64
import secrets
import time
import uuid

import modal

CHECKPOINT_FILENAME = "CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
CHECKPOINT_DIR = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "torchlibrosa",
        "librosa",
        "mido",
        "piano_transcription_inference",
        "numpy",
        "fastapi[standard]",
    )
)

app = modal.App("papiano-transcribe", image=image)

checkpoint_volume = modal.Volume.from_name(
    "papiano-transcribe-checkpoints", create_if_missing=True
)

admin_password_secret = modal.Secret.from_name("papiano-admin-password")

api_keys = modal.Dict.from_name("papiano-api-keys", create_if_missing=True)
key_requests = modal.Dict.from_name("papiano-key-requests", create_if_missing=True)
public_rate_limits = modal.Dict.from_name("papiano-public-rate-limits", create_if_missing=True)

SAMPLE_RATE = 16000
PUBLIC_DAILY_LIMIT = 5


@app.cls(
    gpu="T4",
    scaledown_window=120,
    volumes={CHECKPOINT_DIR: checkpoint_volume},
)
class PianoTranscriber:
    @modal.enter()
    def load(self):
        import os

        import torch
        from piano_transcription_inference import PianoTranscription

        checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(
                f"Checkpoint not found at {checkpoint_path}. Run "
                "`modal run scripts/setup_checkpoint_volume.py` once to "
                "seed the volume."
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transcriptor = PianoTranscription(
            checkpoint_path=checkpoint_path, device=device
        )
        self.transcriptor.frame_threshold = 0.3
        self.transcriptor.offset_threshod = 0.5

    @modal.method()
    def transcribe(self, audio_b64: str) -> dict:
        import tempfile

        import librosa

        audio_bytes = base64.b64decode(audio_b64)

        with tempfile.NamedTemporaryFile(suffix=".audio") as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in.flush()
            audio_waveform, _ = librosa.core.load(
                tmp_in.name, sr=SAMPLE_RATE, mono=True
            )

        with tempfile.NamedTemporaryFile(suffix=".mid") as tmp_out:
            result = self.transcriptor.transcribe(audio_waveform, tmp_out.name)
            tmp_out.seek(0)
            midi_bytes = tmp_out.read()

        notes = [
            {
                "pitch": int(ev["midi_note"]),
                "onset": float(ev["onset_time"]),
                "offset": float(ev["offset_time"]),
                "velocity": int(ev["velocity"]),
            }
            for ev in result["est_note_events"]
        ]

        pedals = [
            {"onset": float(ev["onset_time"]), "offset": float(ev["offset_time"])}
            for ev in result["est_pedal_events"]
        ]

        return {
            "notes": notes,
            "pedals": pedals,
            "midi_base64": base64.b64encode(midi_bytes).decode("utf-8"),
        }


@app.function(secrets=[admin_password_secret])
@modal.asgi_app()
def web():
    import os

    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI()
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def require_admin(x_admin_password: str):
        if x_admin_password != os.environ["ADMIN_PASSWORD"]:
            raise HTTPException(status_code=401, detail="Invalid admin password")

    def client_ip(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host

    @web_app.post("/convert-public")
    def convert_public(item: dict, request: Request):
        import datetime

        ip = client_ip(request)
        today = datetime.date.today().isoformat()
        rate_key = f"{ip}:{today}"
        count = public_rate_limits.get(rate_key, 0)
        if count >= PUBLIC_DAILY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Daily limit of {PUBLIC_DAILY_LIMIT} conversions reached. "
                    "For higher volume or programmatic access, use Papiano."
                ),
            )
        public_rate_limits[rate_key] = count + 1
        audio_b64 = item["audio_base64"]
        transcriber = PianoTranscriber()
        return transcriber.transcribe.remote(audio_b64)

    @web_app.post("/transcribe")
    def transcribe(item: dict, x_api_key: str = Header(default="")):
        record = api_keys.get(x_api_key)
        if record is None or record.get("revoked"):
            raise HTTPException(status_code=401, detail="Invalid API key")
        audio_b64 = item["audio_base64"]
        transcriber = PianoTranscriber()
        return transcriber.transcribe.remote(audio_b64)

    @web_app.post("/request-access")
    def request_access(item: dict):
        name = item.get("name", "").strip()
        email = item.get("email", "").strip()
        reason = item.get("reason", "").strip()
        if not name or not email:
            raise HTTPException(status_code=400, detail="name and email required")
        request_id = str(uuid.uuid4())
        key_requests[request_id] = {
            "name": name,
            "email": email,
            "reason": reason,
            "status": "pending",
            "created_at": time.time(),
        }
        return {"request_id": request_id}

    @web_app.get("/admin/requests")
    def list_requests(x_admin_password: str = Header(default="")):
        require_admin(x_admin_password)
        return [
            {"request_id": rid, **data}
            for rid, data in key_requests.items()
            if data.get("status") == "pending"
        ]

    @web_app.post("/admin/requests/{request_id}/approve")
    def approve_request(request_id: str, x_admin_password: str = Header(default="")):
        require_admin(x_admin_password)
        record = key_requests.get(request_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Request not found")
        new_key = secrets.token_urlsafe(32)
        api_keys[new_key] = {
            "label": record["name"],
            "email": record["email"],
            "revoked": False,
            "created_at": time.time(),
        }
        record["status"] = "approved"
        key_requests[request_id] = record
        return {"api_key": new_key}

    @web_app.post("/admin/requests/{request_id}/reject")
    def reject_request(request_id: str, x_admin_password: str = Header(default="")):
        require_admin(x_admin_password)
        record = key_requests.get(request_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Request not found")
        record["status"] = "rejected"
        key_requests[request_id] = record
        return {"status": "rejected"}

    @web_app.post("/admin/keys")
    def create_key(item: dict, x_admin_password: str = Header(default="")):
        require_admin(x_admin_password)
        label = item.get("label", "").strip() or "unlabeled"
        new_key = secrets.token_urlsafe(32)
        api_keys[new_key] = {
            "label": label,
            "email": item.get("email", ""),
            "revoked": False,
            "created_at": time.time(),
        }
        return {"api_key": new_key}

    @web_app.get("/admin/keys")
    def list_keys(x_admin_password: str = Header(default="")):
        require_admin(x_admin_password)
        return [
            {"api_key": key, **data}
            for key, data in api_keys.items()
        ]

    @web_app.post("/admin/keys/{api_key}/revoke")
    def revoke_key(api_key: str, x_admin_password: str = Header(default="")):
        require_admin(x_admin_password)
        record = api_keys.get(api_key)
        if record is None:
            raise HTTPException(status_code=404, detail="Key not found")
        record["revoked"] = True
        api_keys[api_key] = record
        return {"status": "revoked"}

    return web_app
