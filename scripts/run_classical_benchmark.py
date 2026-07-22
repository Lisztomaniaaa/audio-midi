"""
Run the classical benchmark (eval_data/classical_mutopia/) end-to-end: render
each MIDI to audio, transcribe via the deployed model, score against the
MIDI itself as ground truth, and print a summary table.

Usage:
    python scripts/run_classical_benchmark.py

Requires: pip install mir_eval mido; fluidsynth + a GM soundfont installed
(see scripts/render_midi_audio.py); `modal token set` configured for this
workspace.
"""

import base64
import glob
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from eval_transcription import _midi_to_notes, _notes_to_mir_eval  # noqa: E402
from render_midi_audio import find_soundfont  # noqa: E402

import mir_eval.transcription  # noqa: E402
import mir_eval.transcription_velocity  # noqa: E402
import modal  # noqa: E402

BENCH_DIR = os.path.join(os.path.dirname(__file__), "..", "eval_data", "classical_mutopia")


def render(midi_path: str, wav_path: str, soundfont: str) -> None:
    subprocess.run(
        ["fluidsynth", "-ni", soundfont, midi_path, "-F", wav_path, "-r", "44100"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    soundfont = find_soundfont()
    transcriber = modal.Cls.from_name("papiano-transcribe", "PianoTranscriber")()

    midi_paths = sorted(glob.glob(os.path.join(BENCH_DIR, "*.mid")))
    rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for midi_path in midi_paths:
            name = os.path.splitext(os.path.basename(midi_path))[0]
            wav_path = os.path.join(tmpdir, f"{name}.wav")
            print(f"Rendering + transcribing {name}...", file=sys.stderr)
            render(midi_path, wav_path, soundfont)

            with open(wav_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            result = transcriber.transcribe.remote(audio_b64, True, None, None)

            ref = _midi_to_notes(midi_path)
            est = [(n["onset"], n["offset"], n["pitch"], n["velocity"]) for n in result["notes"]]
            ref_i, ref_p, ref_v = _notes_to_mir_eval(ref)
            est_i, est_p, est_v = _notes_to_mir_eval(est)

            scores = mir_eval.transcription.evaluate(ref_i, ref_p, est_i, est_p)
            vel_scores = mir_eval.transcription_velocity.evaluate(
                ref_i, ref_p, ref_v, est_i, est_p, est_v
            )
            rows.append(
                (
                    name,
                    len(ref),
                    len(est),
                    scores["F-measure_no_offset"],
                    scores["F-measure"],
                    vel_scores["F-measure"],
                )
            )

    header = f"{'piece':<28}{'ref':>6}{'est':>6}{'onset+pitch':>13}{'+offset':>10}{'+velocity':>11}"
    print()
    print(header)
    print("-" * len(header))
    for name, n_ref, n_est, f1_op, f1_full, f1_vel in rows:
        print(f"{name:<28}{n_ref:>6}{n_est:>6}{f1_op:>13.3f}{f1_full:>10.3f}{f1_vel:>11.3f}")

    avg = lambda i: sum(r[i] for r in rows) / len(rows)
    print("-" * len(header))
    print(f"{'AVERAGE':<28}{'':>6}{'':>6}{avg(3):>13.3f}{avg(4):>10.3f}{avg(5):>11.3f}")


if __name__ == "__main__":
    main()
