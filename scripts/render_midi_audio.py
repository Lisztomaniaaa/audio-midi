"""
Render a MIDI file to audio via FluidSynth, so a bare MIDI (no matching
recording) can still be used as an aligned audio+MIDI pair for
scripts/eval_transcription.py. Alignment is exact by construction, since the
audio is synthesized directly from the MIDI's own note events.

Usage:
    python scripts/render_midi_audio.py --midi song.mid --out song.wav

Requires the `fluidsynth` binary and a General MIDI soundfont (.sf2) —
install via: apt-get install --no-install-recommends fluidsynth
(a default soundfont is usually pulled in as a dependency, e.g.
timgm6mb-soundfont on Debian/Ubuntu).
"""

import argparse
import shutil
import subprocess
import sys

DEFAULT_SOUNDFONTS = [
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/sounds/sf2/TimGM6mb.sf2",
]


def find_soundfont() -> str:
    import os

    for path in DEFAULT_SOUNDFONTS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No default General MIDI soundfont found. Install one (e.g. "
        "`apt-get install --no-install-recommends fluid-soundfont-gm`) or "
        "pass --soundfont explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midi", required=True, help="Path to input MIDI file")
    parser.add_argument("--out", default="rendered.wav", help="Path to write output WAV")
    parser.add_argument("--soundfont", default=None, help="Path to a .sf2 soundfont (auto-detected if omitted)")
    parser.add_argument("--sample-rate", type=int, default=44100)
    args = parser.parse_args()

    if not shutil.which("fluidsynth"):
        sys.exit("fluidsynth binary not found — install it first (see module docstring).")

    soundfont = args.soundfont or find_soundfont()

    subprocess.run(
        [
            "fluidsynth", "-ni", soundfont, args.midi,
            "-F", args.out, "-r", str(args.sample_rate),
        ],
        check=True,
    )
    print(f"Rendered {args.midi} -> {args.out} (soundfont: {soundfont})")


if __name__ == "__main__":
    main()
