"""
Score the deployed transcriber against a ground-truth MIDI for the same
performance (aligned audio + MIDI pair), using mir_eval's standard
transcription metrics (the same metric family used in AMT papers/benchmarks).

Usage:
    python scripts/eval_transcription.py --audio song.mp3 --reference-midi song.mid

Requires: pip install mir_eval mido
Calls the Modal app directly (no API key needed) — must be run from an
environment with `modal token set` already configured for this workspace.
"""

import argparse
import base64

import mido
import numpy as np
import mir_eval.transcription
import mir_eval.transcription_velocity
import modal


def _midi_to_notes(path: str) -> list[tuple[float, float, int, int]]:
    """Parse a MIDI file into (onset_s, offset_s, pitch, velocity) tuples,
    merging all tracks and honoring tempo changes (mido gives per-track,
    per-tick deltas; this walks them in absolute-tick order)."""
    mid = mido.MidiFile(path)
    ticks_per_beat = mid.ticks_per_beat

    events = []
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            events.append((tick, msg))
    events.sort(key=lambda e: e[0])

    notes = []
    active: dict[int, list[tuple[float, int]]] = {}
    tempo = 500000  # default 120 BPM
    prev_tick = 0
    time_s = 0.0
    for tick, msg in events:
        time_s += mido.tick2second(tick - prev_tick, ticks_per_beat, tempo)
        prev_tick = tick
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            active.setdefault(msg.note, []).append((time_s, msg.velocity))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            pending = active.get(msg.note)
            if pending:
                onset, vel = pending.pop(0)
                notes.append((onset, time_s, msg.note, vel))
    return notes


def _notes_to_mir_eval(notes: list[tuple[float, float, int, int]]):
    intervals = np.array([[n[0], n[1]] for n in notes], dtype=float)
    pitches_hz = np.array([mir_eval.util.midi_to_hz(n[2]) for n in notes], dtype=float)
    velocities = np.array([n[3] for n in notes], dtype=float)
    return intervals, pitches_hz, velocities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="Path to input audio (mp3/wav/etc)")
    parser.add_argument("--reference-midi", required=True, help="Ground-truth MIDI for the same performance")
    parser.add_argument("--separate-piano", action="store_true", help="Run piano-stem separation first")
    args = parser.parse_args()

    with open(args.audio, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    if args.separate_piano:
        separator = modal.Cls.from_name("papiano-transcribe", "PianoSeparator")()
        audio_b64 = separator.separate_piano.remote(audio_b64)

    transcriber = modal.Cls.from_name("papiano-transcribe", "PianoTranscriber")()
    result = transcriber.transcribe.remote(audio_b64, True, None, None)

    est_notes = [(n["onset"], n["offset"], n["pitch"], n["velocity"]) for n in result["notes"]]
    ref_notes = _midi_to_notes(args.reference_midi)

    ref_intervals, ref_pitches, ref_velocities = _notes_to_mir_eval(ref_notes)
    est_intervals, est_pitches, est_velocities = _notes_to_mir_eval(est_notes)

    print(f"Reference: {len(ref_notes)} notes | Estimated: {len(est_notes)} notes")
    print(f"Audio quality flags: {result['audio_quality']}")
    print()

    scores = mir_eval.transcription.evaluate(ref_intervals, ref_pitches, est_intervals, est_pitches)

    # NB: mir_eval's "Onset_*" keys (onset_precision_recall_f1) match on time
    # ONLY, ignoring pitch entirely - not a meaningful transcription-accuracy
    # number (a dense polyphonic passage can score deceptively high there
    # just from onset density). "*_no_offset" is the one that requires both
    # onset AND pitch to match, which is what "is this note right" means.
    print("Onset+pitch, ignoring duration (is this the right note, roughly the right time):")
    for k in ("Precision_no_offset", "Recall_no_offset", "F-measure_no_offset"):
        print(f"  {k}: {scores[k]:.3f}")

    print("\nOnset+pitch+offset (also: is the note's duration roughly right):")
    for k in ("Precision", "Recall", "F-measure"):
        print(f"  {k}: {scores[k]:.4f}")

    velocity_scores = mir_eval.transcription_velocity.evaluate(
        ref_intervals, ref_pitches, ref_velocities, est_intervals, est_pitches, est_velocities
    )
    print("\nOnset+offset+velocity (also: is the loudness roughly right):")
    for k in ("Precision", "Recall", "F-measure"):
        print(f"  {k}: {velocity_scores[k]:.3f}")


if __name__ == "__main__":
    main()
