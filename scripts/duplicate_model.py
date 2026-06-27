"""
Duplicate Genius-Society/piano_trans into your own Hugging Face account.

This creates a copy of the model repo under your namespace so it can serve as
the base for future retraining/checkpoints, while preserving the original
MIT license file and adding attribution to the original author.

Usage:
    huggingface-cli login   # or set HF_TOKEN env var
    python scripts/duplicate_model.py --target-repo YOUR_USERNAME/piano_trans

Requires: huggingface_hub
"""

import argparse

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

SOURCE_REPO = "Genius-Society/piano_trans"

ATTRIBUTION_NOTICE = f"""
## Provenance

This repository is a duplicate of [`{SOURCE_REPO}`](https://huggingface.co/{SOURCE_REPO}),
forked to serve as the base checkpoint for further fine-tuning/retraining.

- Original model: [{SOURCE_REPO}](https://huggingface.co/{SOURCE_REPO})
- Original architecture/method: Kong et al., "High-resolution Piano Transcription
  with Pedals by Regressing Onset and Offset Times" (arXiv:2010.01815), ByteDance.
- License: MIT (preserved from the original repo; see LICENSE).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-repo",
        required=True,
        help="Destination repo id, e.g. your-username/piano_trans",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the destination repo as private (default: public)",
    )
    args = parser.parse_args()

    api = HfApi()

    print(f"Downloading snapshot of {SOURCE_REPO} ...")
    local_dir = snapshot_download(repo_id=SOURCE_REPO)

    print(f"Creating destination repo {args.target_repo} ...")
    api.create_repo(repo_id=args.target_repo, private=args.private, exist_ok=True)

    print(f"Uploading files to {args.target_repo} ...")
    api.upload_folder(
        repo_id=args.target_repo,
        folder_path=local_dir,
        commit_message=f"Duplicate from {SOURCE_REPO}",
    )

    print("Appending provenance/attribution notice to README ...")
    try:
        readme = hf_hub_download(repo_id=args.target_repo, filename="README.md")
        with open(readme, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        content = f"# {args.target_repo.split('/')[-1]}\n"

    if "## Provenance" not in content:
        content = content.rstrip() + "\n" + ATTRIBUTION_NOTICE + "\n"
        api.upload_file(
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.target_repo,
            commit_message="Add provenance/attribution notice",
        )

    print(f"Done. Model duplicated to: https://huggingface.co/{args.target_repo}")


if __name__ == "__main__":
    main()
