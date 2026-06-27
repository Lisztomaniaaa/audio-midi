"""Papiano Transcribe — piano audio -> MIDI, served on Modal.

First-time setup: modal run scripts/setup_checkpoint_volume.py
Deploy: modal deploy modal_app/app.py
"""

import base64

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

SAMPLE_RATE = 16000


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
        # Library defaults (frame_threshold=0.1, offset_threshod=0.3) make
        # notes ring on far longer than they actually sound. Raising these
        # makes the offset detector cut notes off sooner; tune if needed.
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


@app.function()
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI()
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.post("/transcribe")
    def transcribe(item: dict):
        audio_b64 = item["audio_base64"]
        transcriber = PianoTranscriber()
        return transcriber.transcribe.remote(audio_b64)

    return web_app
