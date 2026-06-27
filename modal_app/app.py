"""
Modal app serving piano audio -> MIDI transcription.

Model: a personal HF duplicate of Genius-Society/piano_trans (PyTorch port of
ByteDance's high-resolution piano transcription model, MIT licensed). Loaded
via `timm.create_model("hf_hub:<HF_MODEL_REPO>", pretrained=True)`.

Postprocessing (regression -> discrete note/pedal events) reuses the
`piano_transcription_inference` package (MIT, same original author) rather
than reimplementing the onset/offset regression decoding.

Deploy:
    modal deploy modal_app/app.py

Set your HF repo id either by editing HF_MODEL_REPO below or via a Modal
secret/env var HF_MODEL_REPO. If the repo is private, also create a Modal
secret named "huggingface-secret" with an HF_TOKEN key.
"""

import base64
import os

import modal

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "Lisztomaniaaa/piano_trans")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "timm",
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


def _load_model():
    """Load the timm-wrapped checkpoint and adapt it to the
    piano_transcription_inference postprocessing pipeline."""
    import timm
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(f"hf_hub:{HF_MODEL_REPO}", pretrained=True)
    model = model.to(device).eval()
    return model, device


def _enframe(audio, segment_samples):
    """(1, audio_samples) -> (N, segment_samples), 50% hop, mirroring
    PianoTranscription.enframe."""
    import numpy as np

    assert audio.shape[1] % segment_samples == 0
    batch = []
    pointer = 0
    while pointer + segment_samples <= audio.shape[1]:
        batch.append(audio[:, pointer : pointer + segment_samples])
        pointer += segment_samples // 2
    return np.concatenate(batch, axis=0)


def _deframe(x):
    """(N, segment_frames, classes_num) -> (audio_frames, classes_num),
    mirroring PianoTranscription.deframe."""
    import numpy as np

    if x.shape[0] == 1:
        return x[0]

    x = x[:, 0:-1, :]
    n = x.shape[0]
    segment_frames = x.shape[1]
    parts = [x[0, 0 : int(segment_frames * 0.75)]]
    for i in range(1, n - 1):
        parts.append(x[i, int(segment_frames * 0.25) : int(segment_frames * 0.75)])
    parts.append(x[-1, int(segment_frames * 0.25) :])
    return np.concatenate(parts, axis=0)


def _run_inference(model, device, audio_waveform):
    """Run the CRNN forward pass over 10s/50%-hop segments and deframe,
    mirroring piano_transcription_inference.PianoTranscription.transcribe."""
    import numpy as np
    from piano_transcription_inference.pytorch_utils import forward
    from piano_transcription_inference.utilities import RegressionPostProcessor

    segment_samples = SAMPLE_RATE * 10
    audio = audio_waveform[None, :]  # (1, audio_samples)
    audio_len = audio.shape[1]
    pad_len = (
        int(np.ceil(audio_len / segment_samples)) * segment_samples - audio_len
    )
    audio = np.concatenate((audio, np.zeros((1, pad_len), dtype=audio.dtype)), axis=1)

    segments = _enframe(audio, segment_samples)  # (N, segment_samples)
    output_dict = forward(model, segments, batch_size=4)

    for key in output_dict.keys():
        output_dict[key] = _deframe(output_dict[key])[0:audio_len]

    post_processor = RegressionPostProcessor(
        frames_per_second=100,
        classes_num=88,
        onset_threshold=0.3,
        offset_threshold=0.3,
        frame_threshold=0.1,
        pedal_offset_threshold=0.2,
    )

    est_note_events, est_pedal_events = post_processor.output_dict_to_midi_events(
        output_dict
    )
    return est_note_events, est_pedal_events


@app.cls(gpu="T4", scaledown_window=120)
class PianoTranscriber:
    @modal.enter()
    def load(self):
        self.model, self.device = _load_model()

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
        from piano_transcription_inference.utilities import write_events_to_midi

        audio_bytes = base64.b64decode(audio_b64)

        with tempfile.NamedTemporaryFile(suffix=".audio") as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in.flush()
            audio_waveform, _ = librosa.core.load(
                tmp_in.name, sr=SAMPLE_RATE, mono=True
            )

        est_note_events, est_pedal_events = _run_inference(
            self.model, self.device, audio_waveform
        )

        notes = [
            {
                "pitch": int(ev["midi_note"]),
                "onset": float(ev["onset_time"]),
                "offset": float(ev["offset_time"]),
                "velocity": int(ev["velocity"]),
            }
            for ev in est_note_events
        ]

        pedals = [
            {"onset": float(ev["onset_time"]), "offset": float(ev["offset_time"])}
            for ev in est_pedal_events
        ]

        with tempfile.NamedTemporaryFile(suffix=".mid") as tmp_out:
            write_events_to_midi(
                start_time=0,
                note_events=est_note_events,
                pedal_events=est_pedal_events,
                midi_path=tmp_out.name,
            )
            tmp_out.seek(0)
            midi_bytes = tmp_out.read()

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
