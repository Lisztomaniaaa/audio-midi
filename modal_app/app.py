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
        "mido",
        f"git+https://github.com/EleutherAI/aria-amt.git@{ARIA_AMT_COMMIT}",
        "fastapi[standard]",
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


def _detect_tempo_and_beats(y):
    import librosa
    import numpy as np

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=SAMPLE_RATE)
    beat_times = librosa.frames_to_time(beat_frames, sr=SAMPLE_RATE)
    bpm = float(np.atleast_1d(tempo)[0])
    if not np.isfinite(bpm) or bpm <= 0:
        bpm = DEFAULT_BPM
    return bpm, np.asarray(beat_times, dtype=float)


def _beat_position_fn(bpm, beat_times):
    """Maps an absolute time (s) to a fractional beat index.

    Uses the actual detected beat times so the mapping follows tempo drift
    (rubato): the rhythmic grid stays clean even when the performance speeds
    up or slows down. Falls back to a fixed tempo if beat tracking failed.
    """
    import numpy as np

    if beat_times.size >= 2:
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

    beats_per_sec = bpm / 60.0
    return lambda t: t * beats_per_sec


def _build_midi(notes, pedals, bpm, beat_times, quantize):
    import io

    import mido

    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(round(bpm)), time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))

    if quantize:
        pos = _beat_position_fn(bpm, beat_times)

        def to_tick(t, snap):
            bp = pos(t)
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

        with tempfile.NamedTemporaryFile(suffix=".audio") as tmp_audio:
            tmp_audio.write(audio_bytes)
            tmp_audio.flush()
            y, _ = librosa.load(tmp_audio.name, sr=SAMPLE_RATE, mono=True)

        notes = _refine_note_offsets(y, notes)
        bpm, beat_times = _detect_tempo_and_beats(y)
        midi_bytes = _build_midi(notes, pedals, bpm, beat_times, quantize)

        # notes/pedals stay in raw performance seconds (synced to the audio);
        # the MIDI file carries the tempo-mapped, grid-quantized rhythm.
        return {
            "notes": notes,
            "pedals": pedals,
            "tempo": round(bpm, 1),
            "time_signature": "4/4",
            "midi_base64": base64.b64encode(midi_bytes).decode("utf-8"),
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
