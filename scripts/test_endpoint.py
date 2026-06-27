"""
Quick test client for the deployed Papiano Transcribe endpoint.

Usage:
    python scripts/test_endpoint.py --url https://your-workspace--papiano-transcribe-web.modal.run/transcribe \
        --audio path/to/song.wav --out output.mid
"""

import argparse
import base64
import json

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Modal endpoint URL")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--out", default="output.mid", help="Path to write output MIDI")
    parser.add_argument("--key", required=True, help="API key")
    args = parser.parse_args()

    with open(args.audio, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    response = requests.post(
        args.url,
        json={"audio_base64": audio_b64},
        headers={"X-API-Key": args.key},
        timeout=300,
    )
    response.raise_for_status()
    result = response.json()

    print(f"Detected {len(result['notes'])} notes, {len(result['pedals'])} pedal events")
    print("First few notes:", json.dumps(result["notes"][:5], indent=2))

    midi_bytes = base64.b64decode(result["midi_base64"])
    with open(args.out, "wb") as f:
        f.write(midi_bytes)
    print(f"Wrote MIDI to {args.out}")


if __name__ == "__main__":
    main()
