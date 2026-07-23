"""A/B a fine-tuned checkpoint against the base checkpoint on Modal.

Remote side only runs the raw model (piano_transcription_inference's own
post-processor) for a given checkpoint name from the checkpoint volume and
returns note events; scoring against reference MIDI happens locally in
scripts (mir_eval), so both checkpoints go through an identical pipeline.

Usage (from repo root):
  modal run training/eval_checkpoint.py --audio-path x.wav --checkpoint note_pedal_ft_v1.pth
Normally invoked via scripts/ab_eval_checkpoints.py rather than directly.
"""

import modal

CHECKPOINT_DIR = "/checkpoints"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.0",
    "numpy",
    "librosa",
    "torchlibrosa",
    "piano_transcription_inference",
)

app = modal.App("papiano-eval-checkpoint", image=image)
checkpoint_volume = modal.Volume.from_name("papiano-transcribe-checkpoints")


@app.function(gpu="T4", timeout=1800, volumes={CHECKPOINT_DIR: checkpoint_volume})
def transcribe_raw(audio_b64: str, checkpoint_name: str) -> list:
    import base64
    import os
    import tempfile

    import librosa
    import torch
    from piano_transcription_inference import PianoTranscription

    transcriptor = PianoTranscription(
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_path=os.path.join(CHECKPOINT_DIR, checkpoint_name),
    )

    audio_bytes = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".audio") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        y, _ = librosa.load(tmp.name, sr=16000, mono=True)

    result = transcriptor.transcribe(y, None)
    return [
        {
            "pitch": int(ev["midi_note"]),
            "onset": float(ev["onset_time"]),
            "offset": float(ev["offset_time"]),
            "velocity": int(ev["velocity"]),
        }
        for ev in result["est_note_events"]
    ]


@app.function(gpu="T4", timeout=1800, volumes={CHECKPOINT_DIR: checkpoint_volume})
def transcribe_raw_pedal(audio_b64: str, checkpoint_name: str) -> list:
    """Same as transcribe_raw but returns the raw per-frame pedal confidence
    instead of note events, for testing pedal-extraction logic offline."""
    import base64
    import os
    import tempfile

    import librosa
    import torch
    from piano_transcription_inference import PianoTranscription

    transcriptor = PianoTranscription(
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_path=os.path.join(CHECKPOINT_DIR, checkpoint_name),
    )

    audio_bytes = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".audio") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        y, _ = librosa.load(tmp.name, sr=16000, mono=True)

    result = transcriptor.transcribe(y, None)
    return result["output_dict"]["pedal_frame_output"][:, 0].tolist()
