"""One-time: download the checkpoint from Zenodo into the Modal Volume.

Usage: modal run scripts/setup_checkpoint_volume.py
"""

import modal

CHECKPOINT_FILENAME = "CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
CHECKPOINT_DIR = "/checkpoints"
ZENODO_URL = (
    "https://zenodo.org/record/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)

image = modal.Image.debian_slim(python_version="3.11").pip_install("requests")

app = modal.App("papiano-transcribe-setup", image=image)

checkpoint_volume = modal.Volume.from_name(
    "papiano-transcribe-checkpoints", create_if_missing=True
)


@app.function(volumes={CHECKPOINT_DIR: checkpoint_volume}, timeout=600)
def seed_checkpoint() -> str:
    import requests

    dest_path = f"{CHECKPOINT_DIR}/{CHECKPOINT_FILENAME}"
    print(f"Downloading checkpoint from {ZENODO_URL} ...")

    with requests.get(ZENODO_URL, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)

    checkpoint_volume.commit()
    print(f"Checkpoint stored at {dest_path} on volume 'papiano-transcribe-checkpoints'.")
    return dest_path


@app.local_entrypoint()
def main():
    result = seed_checkpoint.remote()
    print(f"Done: {result}")
