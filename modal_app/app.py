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
        f"git+https://github.com/EleutherAI/aria-amt.git@{ARIA_AMT_COMMIT}",
        "fastapi[standard]",
    )
)

app = modal.App("papiano-transcribe", image=image)

checkpoint_volume = modal.Volume.from_name(
    "papiano-transcribe-checkpoints", create_if_missing=True
)

api_key_secret = modal.Secret.from_name("papiano-api-key")


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
    def transcribe(self, audio_b64: str) -> dict:
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
        midi_file = midi_dict.to_midi()

        with tempfile.NamedTemporaryFile(suffix=".mid") as tmp_out:
            midi_file.save(tmp_out.name)
            tmp_out.seek(0)
            midi_bytes = tmp_out.read()

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

        return {
            "notes": notes,
            "pedals": pedals,
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
        transcriber = PianoTranscriber()
        return transcriber.transcribe.remote(audio_b64)

    return web_app
