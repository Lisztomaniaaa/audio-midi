"""
One-time setup: download the piano transcription checkpoint and store it on
a Modal Volume, so the serving app (modal_app/app.py) never needs to talk to
Hugging Face at request time.

Usage:
    modal run scripts/setup_checkpoint_volume.py --hf-repo YOUR_USERNAME/piano_trans

This runs on Modal (not locally), downloads the .pth file from the given HF
repo straight into the volume, and exits. Re-run it whenever you push a new
checkpoint to retrain/update the model.
"""

import modal

CHECKPOINT_FILENAME = "CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
CHECKPOINT_DIR = "/checkpoints"

image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub")

app = modal.App("piano-transcription-setup", image=image)

checkpoint_volume = modal.Volume.from_name(
    "piano-transcription-checkpoints", create_if_missing=True
)


@app.function(volumes={CHECKPOINT_DIR: checkpoint_volume})
def seed_checkpoint(hf_repo: str) -> str:
    import shutil

    from huggingface_hub import hf_hub_download

    print(f"Downloading {CHECKPOINT_FILENAME} from {hf_repo} ...")
    downloaded_path = hf_hub_download(repo_id=hf_repo, filename=CHECKPOINT_FILENAME)

    dest_path = f"{CHECKPOINT_DIR}/{CHECKPOINT_FILENAME}"
    shutil.copyfile(downloaded_path, dest_path)
    checkpoint_volume.commit()

    print(f"Checkpoint stored at {dest_path} on volume 'piano-transcription-checkpoints'.")
    return dest_path


@app.local_entrypoint()
def main(hf_repo: str):
    result = seed_checkpoint.remote(hf_repo)
    print(f"Done: {result}")
