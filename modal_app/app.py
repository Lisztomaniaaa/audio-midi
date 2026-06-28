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


STAFF_SPLIT = 60  # middle C: pitches >= split go to the right hand (treble)


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

    for el in score_in.flatten().notes:
        top_pitch = max(p.midi for p in el.pitches)
        hand = "RH" if top_pitch >= STAFF_SPLIT else "LH"
        parts[hand].insert(el.offset, copy.deepcopy(el))

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

    xml = music21.musicxml.m21ToXml.GeneralObjectExporter(score_out).parse()
    return xml.decode("utf-8"), key_name


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


def _build_midi(notes, pedals, bpm, beats, downbeats, bpb, quantize):
    import io

    import mido
    import numpy as np

    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(round(bpm)), time=0))
    track.append(
        mido.MetaMessage("time_signature", numerator=bpb, denominator=4, time=0)
    )

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
    else:
        beats_per_sec = bpm / 60.0

        def to_tick(t, snap):
            return max(0, round(t * beats_per_sec * TICKS_PER_BEAT))

    # (tick, group, message); group orders events sharing a tick so note/pedal
    # offs land before new ons and nothing is cut off a beat early.
    events = []
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
    def transcribe(self, audio_b64: str, quantize: bool = True) -> dict:
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

        if beats.size >= 2:
            bpm = 60.0 / float(np.median(np.diff(beats)))
            if not np.isfinite(bpm) or bpm <= 0:
                bpm = DEFAULT_BPM
        else:
            bpm = DEFAULT_BPM

        bpb = _infer_meter(beats, downbeats)
        midi_bytes = _build_midi(notes, pedals, bpm, beats, downbeats, bpb, quantize)

        # Engrave a 2-staff piano score (MusicXML) from the quantized MIDI —
        # this is what notation/arranger software imports as real sheet music.
        musicxml = None
        key_name = None
        try:
            musicxml, key_name = _build_musicxml(midi_bytes, bpb)
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
        transcriber = PianoTranscriber()
        return transcriber.transcribe.remote(audio_b64, quantize)

    return web_app
