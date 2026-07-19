"""One-time: download the Aria-AMT checkpoint into the Modal Volume.

Usage: modal run scripts/setup_checkpoint_volume.py
"""

import modal

CHECKPOINT_FILENAME = "piano-medium-double-1.0.safetensors"
CHECKPOINT_DIR = "/checkpoints"
CHECKPOINT_URL = (
    "https://huggingface.co/datasets/loubb/aria-midi/resolve/main/"
    "piano-medium-double-1.0.safetensors?download=true"
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
    print(f"Downloading checkpoint from {CHECKPOINT_URL} ...")

    with requests.get(CHECKPOINT_URL, stream=True, timeout=300) as response:
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
