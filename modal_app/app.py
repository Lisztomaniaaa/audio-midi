"""
Modal app serving piano audio -> MIDI transcription.

Model: a personal HF duplicate of Genius-Society/piano_trans (PyTorch
checkpoint of ByteDance's high-resolution piano transcription model,
MIT licensed). The HF repo only contains the raw `.pth` checkpoint (no
timm hub config), so it's downloaded directly via `hf_hub_download` and
loaded through the original author's `piano_transcription_inference`
package (MIT, same upstream method/architecture), which also handles
segmenting, the CRNN forward pass, regression postprocessing into
note/pedal events, and MIDI file writing.

Deploy:
    modal deploy modal_app/app.py

Set your HF repo id either by editing HF_MODEL_REPO below or via a Modal
secret/env var HF_MODEL_REPO. If the repo is private, also create a Modal
secret named "huggingface-secret" with an HF_TOKEN key and reference it
in the @app.cls(...) decorator below.
"""

import base64
import os

import modal

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "Lisztomaniaaa/piano_trans")
HF_CHECKPOINT_FILENAME = "CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "huggingface_hub",
        "torchlibrosa",
        "librosa",
        "mido",
        "piano_transcription_inference",
        "numpy",
        "fastapi[standard]",
    )
)

app = modal.App("piano-transcription", image=image)

SAMPLE_RATE = 16000


@app.cls(gpu="T4", scaledown_window=120)
class PianoTranscriber:
    @modal.enter()
    def load(self):
        import torch
        from huggingface_hub import hf_hub_download
        from piano_transcription_inference import PianoTranscription

        checkpoint_path = hf_hub_download(
            repo_id=HF_MODEL_REPO, filename=HF_CHECKPOINT_FILENAME
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transcriptor = PianoTranscription(
            checkpoint_path=checkpoint_path, device=device
        )

    @modal.method()
    def transcribe(self, audio_b64: str) -> dict:
        """
        audio_b64: base64-encoded audio file bytes (any format ffmpeg/librosa
        can decode: wav, mp3, flac, etc).

        Returns a dict with:
          - notes: list of {pitch, onset, offset, velocity}
          - pedals: list of {onset, offset}
          - midi_base64: base64-encoded MIDI file bytes
        """
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
@modal.fastapi_endpoint(method="POST")
def transcribe(item: dict):
    """
    HTTP endpoint. POST JSON: {"audio_base64": "<base64 audio bytes>"}
    Returns JSON: {"notes": [...], "pedals": [...], "midi_base64": "..."}
    """
    audio_b64 = item["audio_base64"]
    transcriber = PianoTranscriber()
    return transcriber.transcribe.remote(audio_b64)
