"""Score base vs fine-tuned checkpoint on the held-out benchmark pieces.

The held-out pieces (op10_no9, waltz_op64_no1, ballade_no4) were never seen
during fine-tuning; each is rendered with both installed soundfonts and both
checkpoints transcribe the identical audio via training/eval_checkpoint.py.

Usage: python scripts/ab_eval_checkpoints.py [--ft-checkpoint note_pedal_ft_v1.pth]
"""

import argparse
import base64
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from eval_transcription import _midi_to_notes, _notes_to_mir_eval  # noqa: E402

import mir_eval.transcription  # noqa: E402
import mir_eval.transcription_velocity  # noqa: E402
import modal  # noqa: E402

BASE = "note_F1=0.9677_pedal_F1=0.9186.pth"
BENCH = os.path.join(os.path.dirname(__file__), "..", "eval_data", "classical_mutopia")
HELD_OUT = ["chopin_op10_no9", "chopin_waltz_op64_no1", "chopin_ballade_no4"]
SOUNDFONTS = {
    "FluidR3": "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "TimGM6mb": "/usr/share/sounds/sf2/TimGM6mb.sf2",
}


def score(ref_path, est_notes):
    ref = _midi_to_notes(ref_path)
    est = [(n["onset"], n["offset"], n["pitch"], n["velocity"]) for n in est_notes]
    ref_i, ref_p, ref_v = _notes_to_mir_eval(ref)
    est_i, est_p, est_v = _notes_to_mir_eval(est)
    s = mir_eval.transcription.evaluate(ref_i, ref_p, est_i, est_p)
    v = mir_eval.transcription_velocity.evaluate(ref_i, ref_p, ref_v, est_i, est_p, est_v)
    return s["F-measure_no_offset"], s["F-measure"], v["F-measure"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ft-checkpoint", default="note_pedal_ft_v1.pth")
    args = parser.parse_args()

    transcribe_raw = modal.Function.from_name("papiano-eval-checkpoint", "transcribe_raw")

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for piece in HELD_OUT:
            midi_path = os.path.join(BENCH, f"{piece}.mid")
            for sf_name, sf_path in SOUNDFONTS.items():
                wav_path = os.path.join(tmpdir, f"{piece}_{sf_name}.wav")
                subprocess.run(
                    ["fluidsynth", "-ni", sf_path, midi_path, "-F", wav_path, "-r", "16000"],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                with open(wav_path, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode()
                for ckpt_label, ckpt in [("base", BASE), ("ft", args.ft_checkpoint)]:
                    est = transcribe_raw.remote(audio_b64, ckpt)
                    f1s = score(midi_path, est)
                    rows.append((piece, sf_name, ckpt_label, len(est), *f1s))
                    print(f"{piece:<24}{sf_name:<10}{ckpt_label:<6}"
                          f"est={len(est):<6} onset+pitch={f1s[0]:.3f} "
                          f"+offset={f1s[1]:.3f} +velocity={f1s[2]:.3f}", flush=True)

    print("\nAverages:")
    for label in ("base", "ft"):
        sub = [r for r in rows if r[2] == label]
        for i, name in ((4, "onset+pitch"), (5, "+offset"), (6, "+velocity")):
            print(f"  {label} {name}: {sum(r[i] for r in sub)/len(sub):.3f}")


if __name__ == "__main__":
    main()
