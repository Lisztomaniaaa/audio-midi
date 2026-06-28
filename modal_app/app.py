"""Papiano Transcribe — piano audio -> MIDI, served on Modal.

First-time setup: modal run scripts/setup_checkpoint_volume.py
Deploy: modal deploy modal_app/app.py
"""

import base64

import modal

CHECKPOINT_FILENAME = "piano-medium-double-1.0.safetensors"
CHECKPOINT_DIR = "/checkpoints"
MODEL_NAME = "medium-double"

# Pinned commit of github.com/EleutherAI/aria-amt — this code calls a few of
# its internal (non-public) inference helpers directly, so an unpinned
# install could silently break on a future upstream refactor.
ARIA_AMT_COMMIT = "a1ab73fc901d1759ec3bc173c146b3c6a3040261"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==2.5.0",
        "torchaudio==2.5.0",
        "numpy",
        "librosa",
        "soundfile",
        "mido",
        "music21",
        "beat-this",
        f"git+https://github.com/EleutherAI/aria-amt.git@{ARIA_AMT_COMMIT}",
        "fastapi[standard]",
    )
    # Bake the Beat This! checkpoint into the image so warm containers never
    # have to download it at request time.
    .run_commands(
        "python -c \"from beat_this.inference import File2Beats; "
        "File2Beats(checkpoint_path='final0', device='cpu')\""
    )
)

app = modal.App("papiano-transcribe", image=image)

checkpoint_volume = modal.Volume.from_name(
    "papiano-transcribe-checkpoints", create_if_missing=True
)

api_key_secret = modal.Secret.from_name("papiano-api-key")

SAMPLE_RATE = 16000
# Aria-AMT's own note-off prediction is the noisiest part of the model (this
# is the harder half of "onset+offset F1" in its benchmarks) and it tends to
# err toward predicting notes longer than they actually sound, especially in
# pedal-heavy passages where other notes' resonance bleeds into the audio.
# This refines each predicted offset against the actual per-pitch decay in
# the source audio, so it only ever shortens a note, never lengthens one.
MIN_NOTE_DURATION_S = 0.05
DECAY_THRESHOLD_RATIO = 0.15

# Hand "humanizer": within one hand the fingers can't keep holding a note once
# they move to the next one in a run/arpeggio — the sustain you hear is the
# PEDAL (captured separately as CC64), not the fingers. So each note in a run
# is released around the next note's onset. Notes struck together (within the
# chord window) are treated as one press.
LEGATO_OVERLAP_S = 0.04
CHORD_WINDOW_S = 0.05
MAX_HAND_SPAN = 14  # semitones one hand can comfortably span (~a ninth)
VOICE_MAX_JUMP = 16  # max pitch jump (semitones) to keep notes in one voice

# Glissando: a long, fast, one-directional, mostly-stepwise run.
GLISS_MIN_NOTES = 6
GLISS_MAX_IOI = 0.12  # seconds between consecutive notes (fast)
GLISS_MAX_STEP = 2.5  # avg semitones/step (stepwise/chromatic, not leaps)
GLISS_MIN_SPAN = 7  # total semitones covered (at least a fifth)

# MIDI rhythm grid. The model emits absolute millisecond timings with no sense
# of tempo, so the raw MIDI opens at a meaningless 120 BPM and nothing lines up
# to bars/beats in notation software. We detect the real tempo + beat positions
# and snap onsets/durations to a 1/16-note grid so the output reads as music.
TICKS_PER_BEAT = 480
GRID_SUBDIVISIONS = 4  # 4 per quarter-note beat == 1/16-note grid
DEFAULT_BPM = 120.0


def _refine_note_offsets(y, notes: list) -> list:
    import librosa
    import numpy as np

    hop_length = 256
    # bins_per_octave=12 means one CQT bin per semitone, lining up directly
    # with MIDI pitch numbers (bin 0 == MIDI note 21 == A0).
    cqt = np.abs(
        librosa.cqt(
            y,
            sr=SAMPLE_RATE,
            hop_length=hop_length,
            fmin=librosa.midi_to_hz(21),
            n_bins=88,
            bins_per_octave=12,
        )
    )
    frame_times = librosa.frames_to_time(
        np.arange(cqt.shape[1]), sr=SAMPLE_RATE, hop_length=hop_length
    )

    refined = []
    for note in notes:
        bin_idx = note["pitch"] - 21
        onset_idx = int(np.searchsorted(frame_times, note["onset"]))
        offset_idx = int(np.searchsorted(frame_times, note["offset"]))
        offset_idx = min(offset_idx, cqt.shape[1] - 1)

        if not (0 <= bin_idx < 88) or offset_idx <= onset_idx:
            refined.append(note)
            continue

        segment = cqt[bin_idx, onset_idx : offset_idx + 1]
        peak = segment.max()
        if peak <= 0:
            refined.append(note)
            continue

        threshold = peak * DECAY_THRESHOLD_RATIO
        peak_idx = int(segment.argmax())
        decay_idx = next(
            (i for i in range(peak_idx, len(segment)) if segment[i] < threshold),
            None,
        )
        if decay_idx is None:
            refined.append(note)
            continue

        new_offset = float(frame_times[onset_idx + decay_idx])
        new_offset = max(new_offset, note["onset"] + MIN_NOTE_DURATION_S)
        new_offset = min(new_offset, note["offset"])
        refined.append({**note, "offset": round(new_offset, 3)})

    return refined


def _assign_hands(notes):
    """Assign each note to right/left hand by hand-span limit + continuity
    (hands move smoothly), not a fixed middle-C line. Notes struck together
    that span more than one hand can reach are split at their largest gap.
    Returns a list of bools aligned to `notes` (True = right hand).

    `notes` items need "pitch" and "onset"; pitch may be fractional (a chord's
    mean) when called on already-grouped notation elements.
    """
    n = len(notes)
    if n == 0:
        return []

    order = sorted(range(n), key=lambda i: (notes[i]["onset"], notes[i]["pitch"]))
    is_rh = [True] * n
    rh_ref, lh_ref = 72.0, 48.0  # priors: right hand ~C5, left hand ~C3

    j = 0
    while j < len(order):
        t0 = notes[order[j]]["onset"]
        c = j
        while c < len(order) and notes[order[c]]["onset"] <= t0 + CHORD_WINDOW_S:
            c += 1
        cluster = sorted(order[j:c], key=lambda i: notes[i]["pitch"])

        lo, hi = notes[cluster[0]]["pitch"], notes[cluster[-1]]["pitch"]
        if hi - lo > MAX_HAND_SPAN:
            gap_m = max(
                range(len(cluster) - 1),
                key=lambda m: notes[cluster[m + 1]]["pitch"] - notes[cluster[m]]["pitch"],
            )
            lh_part, rh_part = cluster[: gap_m + 1], cluster[gap_m + 1 :]
        else:
            mean_p = sum(notes[i]["pitch"] for i in cluster) / len(cluster)
            if abs(mean_p - rh_ref) <= abs(mean_p - lh_ref):
                rh_part, lh_part = cluster, []
            else:
                rh_part, lh_part = [], cluster

        for i in rh_part:
            is_rh[i] = True
        for i in lh_part:
            is_rh[i] = False
        if rh_part:
            rh_ref = sum(notes[i]["pitch"] for i in rh_part) / len(rh_part)
        if lh_part:
            lh_ref = sum(notes[i]["pitch"] for i in lh_part) / len(lh_part)
        j = c

    return is_rh


def _separate_voices(notes):
    """Split notes into monophonic voices (within each hand) by greedy
    pitch-continuity streaming: the top melodic line stays one voice, inner
    parts another. This is what lets a held melody note ring while a faster
    inner voice in the same hand moves underneath. Returns a voice id per
    note (aligned to `notes`)."""
    n = len(notes)
    if n == 0:
        return []

    hands = _assign_hands(notes)
    voice_of = [0] * n
    next_vid = 0
    for hand in (True, False):
        idxs = sorted(
            (i for i in range(n) if hands[i] == hand),
            key=lambda i: notes[i]["onset"],
        )
        voices = []  # each: {"last_pitch": int, "vid": int}
        j = 0
        while j < len(idxs):
            t0 = notes[idxs[j]]["onset"]
            c = j
            while c < len(idxs) and notes[idxs[c]]["onset"] <= t0 + CHORD_WINDOW_S:
                c += 1
            cluster = sorted(idxs[j:c], key=lambda i: -notes[i]["pitch"])  # high to low
            used = set()
            for i in cluster:
                pitch = notes[i]["pitch"]
                best = None
                for vi, v in enumerate(voices):
                    if vi in used:
                        continue
                    d = abs(v["last_pitch"] - pitch)
                    if best is None or d < best[1]:
                        best = (vi, d)
                if best is not None and best[1] <= VOICE_MAX_JUMP:
                    vi = best[0]
                else:
                    voices.append({"last_pitch": pitch, "vid": next_vid})
                    vi = len(voices) - 1
                    next_vid += 1
                used.add(vi)
                voices[vi]["last_pitch"] = pitch
                voice_of[i] = voices[vi]["vid"]
            j = c
    return voice_of


def _detect_glissandos(notes):
    """Find glissando runs: long, fast, one-directional, mostly-stepwise
    sequences within a hand (the model emits them as many tiny notes). Returns
    a list of {onset, offset, start_pitch, end_pitch, direction, notes}."""
    if len(notes) < GLISS_MIN_NOTES:
        return []

    hands = _assign_hands(notes)
    out = []
    for hand in (True, False):
        idxs = sorted(
            (i for i in range(len(notes)) if hands[i] == hand),
            key=lambda i: notes[i]["onset"],
        )
        i = 0
        while i < len(idxs) - 1:
            run = [idxs[i]]
            direction = 0
            j = i + 1
            while j < len(idxs):
                prev, cur = notes[run[-1]], notes[idxs[j]]
                step = cur["pitch"] - prev["pitch"]
                d = (step > 0) - (step < 0)
                if (
                    cur["onset"] - prev["onset"] <= GLISS_MAX_IOI
                    and 0 < abs(step) <= 4
                    and (direction == 0 or d == direction)
                ):
                    direction = direction or d
                    run.append(idxs[j])
                    j += 1
                else:
                    break

            span = abs(notes[run[-1]]["pitch"] - notes[run[0]]["pitch"])
            if (
                len(run) >= GLISS_MIN_NOTES
                and span >= GLISS_MIN_SPAN
                and span / (len(run) - 1) <= GLISS_MAX_STEP
            ):
                out.append({
                    "onset": round(notes[run[0]]["onset"], 3),
                    "offset": round(notes[run[-1]]["offset"], 3),
                    "start_pitch": int(notes[run[0]]["pitch"]),
                    "end_pitch": int(notes[run[-1]]["pitch"]),
                    "direction": "up" if direction > 0 else "down",
                    "notes": len(run),
                })
                i = j
            else:
                i += 1

    out.sort(key=lambda g: g["onset"])
    return out


def _humanize_durations(notes, voices):
    """Clip note durations to piano hand physics, per voice.

    Within each monophonic voice a note is released around the next note's
    onset in that same voice, so a run/arpeggio doesn't hold its first note —
    but a sustained melody or inner voice is NOT cut short by faster notes
    elsewhere in the same hand (that's why we clip per voice, not per hand).
    The pedal (CC64) still carries the actual sustain.
    """
    if len(notes) < 2:
        return notes

    from collections import defaultdict

    by_voice = defaultdict(list)
    for i, v in enumerate(voices):
        by_voice[v].append(i)

    new_offset = {}
    for idxs in by_voice.values():
        idxs.sort(key=lambda i: notes[i]["onset"])
        for j, i in enumerate(idxs):
            onset = notes[i]["onset"]
            nxt = next(
                (notes[k]["onset"] for k in idxs[j + 1 :]
                 if notes[k]["onset"] > onset + CHORD_WINDOW_S),
                None,
            )
            if nxt is None:
                continue
            clipped = min(notes[i]["offset"], nxt + LEGATO_OVERLAP_S)
            clipped = max(clipped, onset + MIN_NOTE_DURATION_S)
            new_offset[i] = round(clipped, 3)

    return [
        {**n, "offset": new_offset[i]} if i in new_offset else n
        for i, n in enumerate(notes)
    ]


def _beat_position_fn(beat_times):
    """Maps an absolute time (s) to a fractional beat index, following the
    detected beat times so the grid tracks tempo drift (rubato). The caller
    guarantees at least two beats."""
    import numpy as np

    median_interval = float(np.median(np.diff(beat_times)))

    def pos(t):
        if t <= beat_times[0]:
            return (t - beat_times[0]) / median_interval
        if t >= beat_times[-1]:
            return (beat_times.size - 1) + (t - beat_times[-1]) / median_interval
        i = max(0, min(int(np.searchsorted(beat_times, t) - 1), beat_times.size - 2))
        span = beat_times[i + 1] - beat_times[i]
        return i + (t - beat_times[i]) / span if span > 0 else float(i)

    return pos


def _readable_sharps(sharps: int) -> int:
    """Prefer the enharmonic key signature with fewer accidentals
    (e.g. C# major / 7 sharps -> Db major / 5 flats)."""
    alt = sharps - 12 if sharps > 0 else sharps + 12
    if -7 <= alt <= 7 and abs(alt) < abs(sharps):
        return alt
    return sharps


def _respell(part, use_flats: bool) -> None:
    """Re-spell accidentals toward the key's direction so a flat key shows
    Db, not C#."""
    for n in part.recurse().notes:
        for p in n.pitches:
            if p.accidental is None:
                continue
            if use_flats and p.accidental.alter > 0:
                p.getHigherEnharmonic(inPlace=True)
            elif not use_flats and p.accidental.alter < 0:
                p.getLowerEnharmonic(inPlace=True)


_CHORD_TEMPLATES = [
    ("", (0, 4, 7)), ("m", (0, 3, 7)), ("7", (0, 4, 7, 10)),
    ("maj7", (0, 4, 7, 11)), ("m7", (0, 3, 7, 10)), ("dim", (0, 3, 6)),
    ("aug", (0, 4, 8)),
]
_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_NAMES = ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]


def _analyze_chords(score_in, beats_per_bar, use_flats):
    """Per-bar chord progression via chroma + template matching (one symbol
    per bar, repeats collapsed). Aggregating a whole bar ignores passing
    tones, so it reads as a real progression instead of one 'chord' per
    vertical slice. Returns a list of {"bar": int, "symbol": str}."""
    import numpy as np
    from collections import defaultdict

    names = _FLAT_NAMES if use_flats else _SHARP_NAMES
    bars = defaultdict(lambda: np.zeros(12))
    for el in score_in.flatten().notes:
        bar = int(float(el.offset) // beats_per_bar)
        dur = max(0.25, float(el.quarterLength))
        for p in el.pitches:
            bars[bar][p.midi % 12] += dur

    progression = []
    last = None
    for bar in sorted(bars):
        chroma = bars[bar]
        if chroma.sum() <= 0:
            continue
        best = None
        for root in range(12):
            for suffix, intervals in _CHORD_TEMPLATES:
                inside = sum(chroma[(root + i) % 12] for i in intervals)
                score = inside - 0.55 * (chroma.sum() - inside)
                if best is None or score > best[0]:
                    best = (score, root, suffix)
        symbol = names[best[1]] + best[2]
        if symbol != last:
            progression.append({"bar": bar, "symbol": symbol})
            last = symbol
    return progression


def _build_musicxml(midi_bytes, beats_per_bar):
    """Engrave the quantized MIDI into a 2-staff piano score (MusicXML).

    Detects the key, splits hands at middle C, re-spells accidentals to match
    the key, and lays the rhythm out in measures. Returns (xml, key_name).
    """
    import copy
    import tempfile

    import music21

    with tempfile.NamedTemporaryFile(suffix=".mid") as tmp_mid:
        tmp_mid.write(midi_bytes)
        tmp_mid.flush()
        score_in = music21.converter.parse(tmp_mid.name)

    try:
        analyzed = score_in.analyze("key")
        sharps = _readable_sharps(analyzed.sharps)
        mode = analyzed.mode
    except Exception:
        sharps, mode = 0, "major"
    use_flats = sharps < 0
    key_name = music21.key.KeySignature(sharps).asKey(mode).name

    ts_str = f"{beats_per_bar}/4"
    parts = {}
    for hand, clef in (("RH", music21.clef.TrebleClef()), ("LH", music21.clef.BassClef())):
        part = music21.stream.Part()
        part.partName = f"Piano ({hand})"
        part.insert(0, clef)
        part.insert(0, music21.meter.TimeSignature(ts_str))
        part.insert(0, music21.key.KeySignature(sharps))
        parts[hand] = part

    elements = list(score_in.flatten().notes)
    items = [
        {"pitch": sum(p.midi for p in el.pitches) / len(el.pitches),
         "onset": float(el.offset)}
        for el in elements
    ]
    hands = _assign_hands(items)
    for el, is_rh in zip(elements, hands):
        parts["RH" if is_rh else "LH"].insert(el.offset, copy.deepcopy(el))

    for part in parts.values():
        _respell(part, use_flats)
        part.makeMeasures(inPlace=True)
        part.makeRests(fillGaps=True, inPlace=True)

    score_out = music21.stream.Score()
    score_out.insert(0, parts["RH"])
    score_out.insert(0, parts["LH"])
    score_out.insert(
        0,
        music21.layout.StaffGroup(
            [parts["RH"], parts["LH"]], symbol="brace", barTogether=True
        ),
    )

    chords = _analyze_chords(score_in, beats_per_bar, use_flats)

    xml = music21.musicxml.m21ToXml.GeneralObjectExporter(score_out).parse()
    return xml.decode("utf-8"), key_name, chords


def _parse_time_signature(ts):
    """'3/4' -> 3 (beats per bar). Returns None if unusable."""
    if not ts:
        return None
    try:
        num = int(str(ts).split("/")[0])
    except (ValueError, IndexError):
        return None
    return num if 2 <= num <= 7 else None


def _apply_tempo_hint(beats, downbeats, tempo_hint, bpb_hint, audio_dur):
    """Use a user-supplied tempo to fix the beat grid.

    Auto beat tracking is unreliable on expressive/rubato solo piano (it
    octave-errors to half/double tempo, or finds no stable pulse). A tempo
    hint resolves that: it either rescales the detected beats by the nearest
    power-of-two so their density matches the hint (keeping the real, rubato
    positions), or — if no beats were found — lays down a uniform grid at the
    hinted tempo. This mirrors how klang.io asks the user for a tempo range.
    """
    import numpy as np

    if not tempo_hint or tempo_hint <= 0:
        return beats, downbeats

    if beats.size < 2:
        interval = 60.0 / tempo_hint
        beats = np.arange(0.0, audio_dur + interval, interval)
        downbeats = beats[:: (bpb_hint or 4)]
        return beats, downbeats

    detected = 60.0 / float(np.median(np.diff(beats)))
    factors = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    f = float(factors[np.argmin(np.abs(detected * factors - tempo_hint))])
    if f > 1.0:
        n = int(round(f))
        dense = []
        for a, b in zip(beats[:-1], beats[1:]):
            dense.extend(a + (b - a) * k / n for k in range(n))
        dense.append(float(beats[-1]))
        beats = np.asarray(dense, dtype=float)  # original beats kept => downbeats stay valid
    elif f < 1.0:
        beats = beats[:: int(round(1.0 / f))]
    return beats, downbeats


def _infer_meter(beats, downbeats):
    """Beats per bar from the spacing of detected downbeats (4 -> 4/4)."""
    import numpy as np

    if downbeats.size >= 2 and beats.size >= 2:
        counts = [
            int(np.sum((beats >= a - 1e-6) & (beats < b - 1e-6)))
            for a, b in zip(downbeats[:-1], downbeats[1:])
        ]
        counts = [c for c in counts if c > 0]
        if counts:
            bpb = int(round(np.median(counts)))
            if 2 <= bpb <= 7:
                return bpb
    return 4


def _downbeats_reliable(beats, downbeats):
    """True only if the tracked downbeats imply a consistent bar length —
    otherwise we don't trust them for meter (e.g. rubato solo piano, where
    nearly every beat got flagged as a downbeat)."""
    import numpy as np
    from collections import Counter

    if downbeats.size < 3 or beats.size < 6:
        return False
    counts = [
        int(np.sum((beats >= a - 1e-6) & (beats < b - 1e-6)))
        for a, b in zip(downbeats[:-1], downbeats[1:])
    ]
    counts = [c for c in counts if c > 0]
    if len(counts) < 2:
        return False
    mode, freq = Counter(counts).most_common(1)[0]
    return freq / len(counts) >= 0.6 and 2 <= mode <= 4


def _infer_meter_from_notes(notes, beats):
    """Infer beats-per-bar + downbeat phase from where the music puts its
    metric weight: bass notes mark downbeats (waltz "oom-pah-pah", ragtime
    stride). Autocorrelating a per-beat accent signal reveals the period
    (3 -> waltz, 2/4 -> ragtime/march). Used when downbeat tracking is
    unreliable. Returns (beats_per_bar, phase)."""
    import numpy as np

    if beats.size < 4 or len(notes) < 8:
        return 4, 0

    pos = _beat_position_fn(beats)
    n_slots = int(round(pos(max(n["onset"] for n in notes)))) + 1
    if n_slots < 6:
        return 4, 0

    # Per-beat weight: each onset counts, lower (bass) pitches count much more
    # because they carry the downbeat in stride/waltz accompaniment.
    weight = np.zeros(n_slots)
    for n in notes:
        b = int(round(pos(n["onset"])))
        if 0 <= b < n_slots:
            weight[b] += 1.0 + max(0.0, (60 - n["pitch"]) / 6.0)

    w = weight - weight.mean()
    ac = {p: float(np.sum(w[:-p] * w[p:])) for p in (2, 3, 4) if n_slots > p}
    if not ac:
        return 4, 0
    bpb = max(ac, key=ac.get)
    # 2/4 and 4/4 are metrically nested; prefer the larger common meter on ties.
    if bpb == 2 and ac.get(4, float("-inf")) >= 0.8 * ac[2]:
        bpb = 4
    phase = max(range(bpb), key=lambda ph: float(weight[ph::bpb].sum()))
    return bpb, phase


def _build_midi(notes, pedals, bpm, beats, downbeats, bpb, quantize):
    import io

    import mido
    import numpy as np

    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # (tick, group, message); group orders events sharing a tick: time_sig (-2)
    # then tempo (-1) then note/pedal offs (0) before new ons (1, 2).
    events = [(0, -2, mido.MetaMessage("time_signature", numerator=bpb, denominator=4, time=0))]

    if quantize and beats.size >= 2:
        pos = _beat_position_fn(beats)

        # Align bar 1 to the first downbeat so measures line up, while keeping
        # every tick >= 0 (pre-downbeat pickup notes fill a leading partial bar).
        origin = pos(float(downbeats[0])) if downbeats.size else 0.0
        all_onsets = [n["onset"] for n in notes] + [p["onset"] for p in pedals]
        min_pos = min((pos(t) for t in all_onsets), default=0.0)
        deficit = origin - min_pos
        pad_bars = int(np.ceil(deficit / bpb)) if deficit > 0 else 0
        shift = origin - pad_bars * bpb

        def to_tick(t, snap):
            bp = pos(t) - shift
            if snap:
                bp = round(bp * GRID_SUBDIVISIONS) / GRID_SUBDIVISIONS
            return max(0, round(bp * TICKS_PER_BEAT))

        # Tempo map: one set_tempo per beat from its real duration, so playback
        # follows the performance's rubato instead of one flat (monotone) tempo.
        prev_bpm = None
        for i, interval in enumerate(np.diff(beats)):
            if interval <= 0:
                continue
            local_bpm = max(20.0, min(400.0, 60.0 / float(interval)))
            if prev_bpm is not None and abs(local_bpm - prev_bpm) < 1.0:
                continue
            prev_bpm = local_bpm
            btick = max(0, round((pos(float(beats[i])) - shift) * TICKS_PER_BEAT))
            events.append((btick, -1, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(local_bpm))))
    else:
        beats_per_sec = bpm / 60.0

        def to_tick(t, snap):
            return max(0, round(t * beats_per_sec * TICKS_PER_BEAT))

    if not any(g == -1 and tk == 0 for tk, g, _ in events):
        safe_bpm = max(20.0, min(400.0, bpm))
        events.append((0, -1, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(safe_bpm))))

    for n in notes:
        on = to_tick(n["onset"], True)
        off = max(to_tick(n["offset"], True), on + 1)
        vel = max(1, min(127, int(n["velocity"])))
        events.append((on, 1, mido.Message("note_on", note=n["pitch"], velocity=vel)))
        events.append((off, 0, mido.Message("note_off", note=n["pitch"], velocity=0)))
    for p in pedals:
        on = to_tick(p["onset"], False)
        off = max(to_tick(p["offset"], False), on + 1)
        events.append((on, 2, mido.Message("control_change", control=64, value=127)))
        events.append((off, 0, mido.Message("control_change", control=64, value=0)))

    events.sort(key=lambda e: (e[0], e[1]))
    prev_tick = 0
    for tick, _group, msg in events:
        msg.time = tick - prev_tick
        prev_tick = tick
        track.append(msg)

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


@app.cls(
    gpu="T4",
    scaledown_window=120,
    timeout=600,
    volumes={CHECKPOINT_DIR: checkpoint_volume},
)
class PianoTranscriber:
    @modal.enter()
    def load(self):
        import os

        import torch
        from amt.audio import AudioTransform
        from amt.config import load_model_config
        from amt.inference.model import AmtEncoderDecoder, ModelConfig
        from amt.tokenizer import AmtTokenizer
        from amt.utils import _load_weight

        checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(
                f"Checkpoint not found at {checkpoint_path}. Run "
                "`modal run scripts/setup_checkpoint_volume.py` once to "
                "seed the volume."
            )

        self.tokenizer = AmtTokenizer()

        model_config = ModelConfig(**load_model_config(MODEL_NAME))
        model_config.set_vocab_size(self.tokenizer.vocab_size)
        model = AmtEncoderDecoder(model_config)

        model_state = _load_weight(ckpt_path=checkpoint_path)
        model_state = {
            (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
            for k, v in model_state.items()
        }
        model.load_state_dict(model_state)

        model.decoder.setup_cache(
            batch_size=1,
            max_seq_len=4096,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float,
        )
        model.cuda()
        model.eval()

        self.model = model
        self.audio_transform = AudioTransform().cuda()

        from beat_this.inference import File2Beats

        self.beat_tracker = File2Beats(
            checkpoint_path="final0",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    @modal.method()
    def transcribe(
        self,
        audio_b64: str,
        quantize: bool = True,
        tempo_hint: float = None,
        time_signature: str = None,
    ) -> dict:
        import logging
        import tempfile

        from amt.data import get_wav_segments
        from amt.inference.transcribe import (
            CHUNK_LEN_MS,
            LEN_MS,
            STRIDE_FACTOR,
            _get_silent_intervals,
            _process_silent_intervals,
            _shift_onset,
            _truncate_seq,
            process_segments,
        )

        logger = logging.getLogger(__name__)
        tokenizer = self.tokenizer
        audio_bytes = base64.b64decode(audio_b64)

        # Mirrors aria-amt's own transcribe_file(), but calls process_segments()
        # in-process instead of handing segments off to its multiprocessing
        # GPU-worker queue (overkill for one request on an already-loaded model).
        with tempfile.NamedTemporaryFile(suffix=".audio") as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in.flush()

            seq = [tokenizer.bos_tok]
            concat_seq = [tokenizer.bos_tok]
            idx = 0
            for curr_audio_segment in get_wav_segments(
                audio_path=tmp_in.name,
                stride_factor=STRIDE_FACTOR,
                pad_last=True,
            ):
                init_idx = len(seq)
                silent_intervals = _get_silent_intervals(curr_audio_segment)
                input_seq = list(seq)

                results = process_segments(
                    tasks=[((curr_audio_segment, seq), 0)],
                    model=self.model,
                    audio_transform=self.audio_transform,
                    tokenizer=tokenizer,
                    logger=logger,
                )
                seq = results[0]

                seq_adj = _process_silent_intervals(
                    seq, intervals=silent_intervals, tokenizer=tokenizer
                )
                if len(seq_adj) < len(seq) - 15:
                    seq = seq_adj

                try:
                    next_seq = _truncate_seq(
                        seq, CHUNK_LEN_MS, LEN_MS - CHUNK_LEN_MS
                    )
                except Exception:
                    try:
                        seq = _truncate_seq(
                            input_seq, CHUNK_LEN_MS - 2, CHUNK_LEN_MS
                        )
                    except Exception:
                        seq = [tokenizer.bos_tok]
                else:
                    if seq[-1] == tokenizer.eos_tok:
                        seq = seq[:-1]
                    concat_seq += _shift_onset(seq[init_idx:], idx * CHUNK_LEN_MS)
                    seq = [tokenizer.bos_tok] if len(next_seq) == 1 else next_seq

                idx += 1

        last_onset = next(
            tok[1]
            for tok in reversed(concat_seq)
            if isinstance(tok, tuple) and tok[0] == "onset"
        )
        midi_dict = tokenizer.detokenize(concat_seq, last_onset)
        midi_dict.remove_redundant_pedals()

        notes = [
            {
                "pitch": int(m["data"]["pitch"]),
                "onset": m["data"]["start"] / 1000.0,
                "offset": m["data"]["end"] / 1000.0,
                "velocity": int(m["data"]["velocity"]),
            }
            for m in midi_dict.note_msgs
        ]

        pedals = []
        pedal_on_tick = None
        for m in sorted(midi_dict.pedal_msgs, key=lambda m: m["tick"]):
            if m["data"] == 1 and pedal_on_tick is None:
                pedal_on_tick = m["tick"]
            elif m["data"] == 0 and pedal_on_tick is not None:
                pedals.append(
                    {
                        "onset": pedal_on_tick / 1000.0,
                        "offset": m["tick"] / 1000.0,
                    }
                )
                pedal_on_tick = None
        if pedal_on_tick is not None:
            pedals.append(
                {"onset": pedal_on_tick / 1000.0, "offset": last_onset / 1000.0}
            )

        # Load the source audio once for both the offset refinement and the
        # tempo/beat detection that drives the MIDI rhythm grid.
        import librosa
        import numpy as np
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".audio") as tmp_audio:
            tmp_audio.write(audio_bytes)
            tmp_audio.flush()
            y, _ = librosa.load(tmp_audio.name, sr=SAMPLE_RATE, mono=True)

        notes = _refine_note_offsets(y, notes)
        glissandos = _detect_glissandos(notes)
        voices = _separate_voices(notes)
        notes_pre = notes
        notes = _humanize_durations(notes, voices)
        clipped = sum(
            1 for a, b in zip(notes_pre, notes)
            if b["offset"] < a["offset"] - 1e-6
        )

        # Beat This! (neural, ISMIR 2024) tracks beats + downbeats and handles
        # rubato/expressive piano, where classic trackers octave-error
        # (double-tempo) on solo piano. If it yields no usable beats, the MIDI
        # is written at a default tempo without a rhythm grid.
        beats = downbeats = np.asarray([], dtype=float)
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_wav:
                sf.write(tmp_wav.name, y, SAMPLE_RATE)
                bt_beats, bt_downbeats = self.beat_tracker(tmp_wav.name)
            beats = np.asarray(bt_beats, dtype=float)
            downbeats = np.asarray(bt_downbeats, dtype=float)
        except Exception:
            logger.exception("Beat This! tracking failed")

        # Optional user hints (klang.io-style) override unreliable auto-detection.
        bpb_hint = _parse_time_signature(time_signature)
        beats, downbeats = _apply_tempo_hint(
            beats, downbeats, tempo_hint, bpb_hint, len(y) / SAMPLE_RATE
        )

        if beats.size >= 2:
            bpm = 60.0 / float(np.median(np.diff(beats)))
            if not np.isfinite(bpm) or bpm <= 0:
                bpm = tempo_hint or DEFAULT_BPM
        else:
            bpm = tempo_hint or DEFAULT_BPM

        # Meter: explicit hint > consistent tracked downbeats > music-theory
        # inference from note accents (handles waltz 3/4 and ragtime phase when
        # the tracker's downbeats can't be trusted).
        if bpb_hint:
            bpb = bpb_hint
        elif _downbeats_reliable(beats, downbeats):
            bpb = _infer_meter(beats, downbeats)
        elif beats.size >= 2:
            bpb, phase = _infer_meter_from_notes(notes, beats)
            downbeats = beats[phase::bpb]
        else:
            bpb = _infer_meter(beats, downbeats)

        midi_bytes = _build_midi(notes, pedals, bpm, beats, downbeats, bpb, quantize)

        # Engrave a 2-staff piano score (MusicXML) from the quantized MIDI —
        # this is what notation/arranger software imports as real sheet music.
        musicxml = None
        key_name = None
        chords = []
        try:
            musicxml, key_name, chords = _build_musicxml(midi_bytes, bpb)
        except Exception:
            logger.exception("MusicXML engraving failed")

        # notes/pedals stay in raw performance seconds (synced to the audio);
        # the MIDI file carries the tempo-mapped, grid-quantized rhythm.
        return {
            "notes": notes,
            "pedals": pedals,
            "tempo": round(bpm, 1),
            "time_signature": f"{bpb}/4",
            "key": key_name,
            "chords": chords,
            "glissandos": glissandos,
            "debug": {
                "notes": len(notes),
                "voices": len(set(voices)),
                "durations_clipped": clipped,
                "glissandos": len(glissandos),
            },
            "midi_base64": base64.b64encode(midi_bytes).decode("utf-8"),
            "musicxml": musicxml,
        }


@app.function(secrets=[api_key_secret])
@modal.asgi_app()
def web():
    import os

    from fastapi import FastAPI, Header, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI()
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.post("/transcribe")
    def transcribe(item: dict, x_api_key: str = Header(default="")):
        if x_api_key != os.environ["API_KEY"]:
            raise HTTPException(status_code=401, detail="Invalid API key")
        audio_b64 = item["audio_base64"]
        quantize = bool(item.get("quantize", True))
        tempo_hint = item.get("tempo_hint")
        time_signature = item.get("time_signature")
        transcriber = PianoTranscriber()
        return transcriber.transcribe.remote(
            audio_b64, quantize, tempo_hint, time_signature
        )

    return web_app
