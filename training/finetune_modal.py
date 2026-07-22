"""Fine-tune the Kong et al. Note_pedal checkpoint on synthesized audio.

Why: the released checkpoint was trained only on MAESTRO (real Steinway
recordings). On synthesized/digital-piano timbres it hallucinates and drops
notes (measured on our benchmark). Fine-tuning on FluidSynth-rendered audio
from cleanly-licensed MIDI adapts it to non-"real piano" timbres without
touching NC-licensed data.

Data layout (uploaded to the papiano-finetune-data volume):
  <piece>.mid + <piece>__<soundfont>.wav  (16kHz)

Usage:
  modal volume create papiano-finetune-data
  modal volume put papiano-finetune-data <local ft_data dir> /data
  modal run training/finetune_modal.py --steps 3000

Writes note_pedal_ft_<tag>.pth into the existing checkpoint volume, alongside
the original (never overwrites it).
"""

import modal

CHECKPOINT_DIR = "/checkpoints"
DATA_DIR = "/data"
BASE_CHECKPOINT = "note_F1=0.9677_pedal_F1=0.9186.pth"

SEGMENT_SECONDS = 10.0
SAMPLE_RATE = 16000
FRAMES_PER_SECOND = 100
BEGIN_NOTE = 21
CLASSES_NUM = 88

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.0",
        "numpy",
        "librosa",
        "mido",
        "torchlibrosa",
        "piano_transcription_inference",
    )
    .add_local_python_source("bytedance_targets")
)

app = modal.App("papiano-finetune", image=image)
checkpoint_volume = modal.Volume.from_name("papiano-transcribe-checkpoints")
data_volume = modal.Volume.from_name("papiano-finetune-data", create_if_missing=True)


@app.function(
    gpu="A10G",  # T4 (16GB) OOMs at batch 8 on 10s segments; A10G's 24GB fits
    timeout=6 * 3600,
    volumes={CHECKPOINT_DIR: checkpoint_volume, DATA_DIR: data_volume},
)
def finetune(steps: int = 3000, batch_size: int = 8, lr: float = 1e-4, tag: str = "v1"):
    import glob
    import os
    import random

    import librosa
    import numpy as np
    import torch

    from bytedance_targets import TargetProcessor, read_midi_any
    from piano_transcription_inference.models import Note_pedal

    device = torch.device("cuda")

    # ---- Load all (waveform, midi-events) pairs into memory ----
    pairs = []
    for wav_path in sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)):
        piece = os.path.basename(wav_path).split("__")[0]
        mid_path = os.path.join(os.path.dirname(wav_path), f"{piece}.mid")
        if not os.path.exists(mid_path):
            continue
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        events = read_midi_any(mid_path)
        pairs.append((audio, events))
        print(f"loaded {os.path.basename(wav_path)} ({len(audio)/SAMPLE_RATE:.0f}s)")
    assert pairs, "no training data found on the volume"

    target_processor = TargetProcessor(
        SEGMENT_SECONDS, FRAMES_PER_SECOND, BEGIN_NOTE, CLASSES_NUM
    )
    segment_samples = int(SEGMENT_SECONDS * SAMPLE_RATE)

    def sample_batch_n(n):
        waveforms, targets = [], []
        while len(waveforms) < n:
            audio, events = random.choice(pairs)
            if len(audio) <= segment_samples:
                continue
            start = random.randint(0, len(audio) - segment_samples - 1)
            start_time = start / SAMPLE_RATE
            target_dict, note_events, _ = target_processor.process(
                start_time, events["midi_event_time"], events["midi_event"]
            )
            if len(note_events) == 0:  # skip silent segments
                continue
            seg = audio[start : start + segment_samples].copy()
            # Cheap augmentation: random gain + light noise, so the model
            # doesn't lock onto one rendering loudness/noise floor.
            seg *= 10 ** (random.uniform(-6, 3) / 20)
            if random.random() < 0.3:
                seg += np.random.normal(0, random.uniform(1e-4, 3e-3), seg.shape)
            waveforms.append(seg)
            targets.append(target_dict)
        batch_wav = torch.tensor(np.stack(waveforms), dtype=torch.float32, device=device)
        batch_tgt = {}
        for key in [
            "reg_onset_roll", "reg_offset_roll", "frame_roll", "velocity_roll",
            "mask_roll", "onset_roll", "reg_pedal_onset_roll",
            "reg_pedal_offset_roll", "pedal_frame_roll",
        ]:
            batch_tgt[key] = torch.tensor(
                np.stack([t[key] for t in targets]), dtype=torch.float32, device=device
            )
        return batch_wav, batch_tgt

    # ---- Model ----
    model = Note_pedal(frames_per_second=FRAMES_PER_SECOND, classes_num=CLASSES_NUM)
    checkpoint = torch.load(
        os.path.join(CHECKPOINT_DIR, BASE_CHECKPOINT), map_location="cpu"
    )
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    def bce(output, target, mask):
        eps = 1e-7
        output = torch.clamp(output, eps, 1.0 - eps)
        matrix = -target * torch.log(output) - (1.0 - target) * torch.log(1.0 - output)
        return torch.sum(matrix * mask) / torch.sum(mask)

    accum = 2  # micro-batches per optimizer step (memory: full batch OOMs)
    micro_bs = max(1, batch_size // accum)

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        note_loss_val = pedal_loss_val = 0.0
        for _ in range(accum):
            wav, tgt = sample_batch_n(micro_bs)
            out = model(wav)
            note_loss = (
                bce(out["reg_onset_output"], tgt["reg_onset_roll"], tgt["mask_roll"])
                + bce(out["reg_offset_output"], tgt["reg_offset_roll"], tgt["mask_roll"])
                + bce(out["frame_output"], tgt["frame_roll"], tgt["mask_roll"])
                + bce(out["velocity_output"], tgt["velocity_roll"] / 128.0, tgt["onset_roll"])
            )
            pedal_loss = (
                torch.nn.functional.binary_cross_entropy(
                    out["reg_pedal_onset_output"], tgt["reg_pedal_onset_roll"][:, :, None]
                )
                + torch.nn.functional.binary_cross_entropy(
                    out["reg_pedal_offset_output"], tgt["reg_pedal_offset_roll"][:, :, None]
                )
                + torch.nn.functional.binary_cross_entropy(
                    out["pedal_frame_output"], tgt["pedal_frame_roll"][:, :, None]
                )
            )
            loss = (note_loss + pedal_loss) / accum
            loss.backward()
            note_loss_val += note_loss.item() / accum
            pedal_loss_val += pedal_loss.item() / accum
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            print(f"step {step}/{steps} "
                  f"note={note_loss_val:.4f} pedal={pedal_loss_val:.4f}")
        if step % 1000 == 0 or step == steps:
            out_name = f"note_pedal_ft_{tag}.pth"
            state = {
                "model": {
                    "note_model": model.note_model.state_dict(),
                    "pedal_model": model.pedal_model.state_dict(),
                }
            }
            torch.save(state, os.path.join(CHECKPOINT_DIR, out_name))
            checkpoint_volume.commit()
            print(f"saved {out_name} at step {step}")

    return f"done: note_pedal_ft_{tag}.pth"


@app.local_entrypoint()
def main(steps: int = 3000, batch_size: int = 8, lr: float = 1e-4, tag: str = "v1"):
    print(finetune.remote(steps=steps, batch_size=batch_size, lr=lr, tag=tag))
