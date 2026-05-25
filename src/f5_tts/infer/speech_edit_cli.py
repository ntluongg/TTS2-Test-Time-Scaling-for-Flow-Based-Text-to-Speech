import argparse
import os
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility

from importlib.resources import files

import torch
import torch.nn.functional as F
import torchaudio
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    save_spectrogram,
    mel_spec_type,
    target_rms,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    device as default_device,
)
from f5_tts.model.utils import convert_char_to_pinyin


def main():
    parser = argparse.ArgumentParser(description="Speech Editing Inference CLI for F5-TTS")
    
    parser.add_argument("--audio_to_edit", type=str, required=True, help="Path to the audio file to edit")
    parser.add_argument("--target_text", type=str, required=True, help="The complete target text after editing")
    parser.add_argument("--parts_to_edit", type=str, required=True, help="Pipe-separated list of start,end times in seconds. e.g., '1.42,2.44|4.04,4.9'")
    parser.add_argument("--fix_duration", type=str, default=None, help="Pipe-separated list of fix durations in seconds for each edit part, e.g., '1.2|1.0'. If not provided, it uses the original text duration.")
    
    parser.add_argument("--model", type=str, default="F5TTS_v1_Base", help="The model name: F5TTS_v1_Base | F5TTS_Base | E2TTS_Base | etc.")
    parser.add_argument("--model_cfg", type=str, help="The path to F5-TTS model config file .yaml")
    parser.add_argument("--ckpt_file", type=str, help="The path to model checkpoint .pt or .safetensors, leave blank to use default")
    parser.add_argument("--vocab_file", type=str, default="", help="The path to vocab file .txt, leave blank to use default")
    
    parser.add_argument("--vocoder_name", type=str, default="vocos", choices=["vocos", "bigvgan"], help=f"Used vocoder name: vocos | bigvgan")
    parser.add_argument("--load_vocoder_from_local", action="store_true", help="To load vocoder from local dir")
    
    parser.add_argument("--output_dir", type=str, default="tests", help="Directory to save the edited audio")
    parser.add_argument("--output_name", type=str, default="speech_edit_out", help="Base name for the output files")
    
    parser.add_argument("--nfe_step", type=int, default=32, help=f"The number of function evaluation (denoising steps), default 32")
    parser.add_argument("--cfg_strength", type=float, default=2.0, help=f"Classifier-free guidance strength, default 2.0")
    parser.add_argument("--sway_sampling_coef", type=float, default=-1.0, help=f"Sway Sampling coefficient, default -1.0")
    parser.add_argument("--target_rms", type=float, default=0.1, help=f"Target output speech loudness normalization value, default 0.1")
    parser.add_argument("--device", type=str, default=default_device, help="Specify the device to run on")

    args = parser.parse_args()

    # Parse parts to edit and durations
    parts_to_edit = []
    for part in args.parts_to_edit.split("|"):
        start, end = map(float, part.split(","))
        parts_to_edit.append([start, end])
        
    fix_dur_list = None
    if args.fix_duration and args.fix_duration.lower() != "none":
        fix_dur_list = [float(x) for x in args.fix_duration.split("|")]
        if len(fix_dur_list) != len(parts_to_edit):
            raise ValueError("Length of fix_duration must match the length of parts_to_edit")

    # Load vocoder
    if args.vocoder_name == "vocos":
        vocoder_local_path = "../checkpoints/vocos-mel-24khz" 
    elif args.vocoder_name == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"

    vocoder = load_vocoder(
        vocoder_name=args.vocoder_name, is_local=args.load_vocoder_from_local, local_path=vocoder_local_path, device=args.device
    )

    # Load TTS model
    model_cfg = OmegaConf.load(
        args.model_cfg or str(files("f5_tts").joinpath(f"configs/{args.model}.yaml"))
    )
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"
    
    if args.model != "F5TTS_Base":
        assert args.vocoder_name == model_cfg.model.mel_spec.mel_spec_type

    # override for previous models
    model_name_for_ckpt = args.model
    if args.model == "F5TTS_Base":
        if args.vocoder_name == "vocos":
            ckpt_step = 1200000
        elif args.vocoder_name == "bigvgan":
            model_name_for_ckpt = "F5TTS_Base_bigvgan"
            ckpt_type = "pt"
    elif args.model == "E2TTS_Base":
        repo_name = "E2-TTS"
        ckpt_step = 1200000

    ckpt_file = args.ckpt_file
    if not ckpt_file:
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model_name_for_ckpt}/model_{ckpt_step}.{ckpt_type}"))
    elif ckpt_file.startswith("hf://"):
        ckpt_file = str(cached_path(ckpt_file))

    vocab_file = args.vocab_file
    if vocab_file.startswith("hf://"):
        vocab_file = str(cached_path(vocab_file))

    print(f"Using {args.model}...")
    model = load_model(
        model_cls, model_arc, ckpt_file, mel_spec_type=args.vocoder_name, vocab_file=vocab_file, device=args.device
    )

    # Get settings from model_cfg for mel spectrogram creation
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
    n_mel_channels = model_cfg.model.mel_spec.n_mel_channels
    hop_length = model_cfg.model.mel_spec.hop_length
    tokenizer_type = model_cfg.model.tokenizer

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Audio prep
    audio, sr = torchaudio.load(args.audio_to_edit)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < args.target_rms:
        audio = audio * args.target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)

    # Convert to mel spectrogram FIRST (on clean original audio)
    audio = audio.to(args.device)
    with torch.inference_mode():
        original_mel = model.mel_spec(audio)  # (batch, n_mel, n_frames)
        original_mel = original_mel.permute(0, 2, 1)  # (batch, n_frames, n_mel)

    # Build mel_cond and edit_mask at FRAME level
    offset_frame = 0
    mel_cond = torch.zeros(1, 0, n_mel_channels, device=args.device)
    edit_mask = torch.zeros(1, 0, dtype=torch.bool, device=args.device)

    for part in parts_to_edit:
        start, end = part
        part_dur_sec = end - start if fix_dur_list is None else fix_dur_list.pop(0)

        # Convert to frames
        start_frame = round(start * target_sample_rate / hop_length)
        end_frame = round(end * target_sample_rate / hop_length)
        part_dur_frames = round(part_dur_sec * target_sample_rate / hop_length)

        # Number of frames for the kept (non-edited) region
        keep_frames = start_frame - offset_frame

        # Build mel_cond
        mel_cond = torch.cat(
            (
                mel_cond,
                original_mel[:, offset_frame:start_frame, :],
                torch.zeros(1, part_dur_frames, n_mel_channels, device=args.device),
            ),
            dim=1,
        )
        edit_mask = torch.cat(
            (
                edit_mask,
                torch.ones(1, keep_frames, dtype=torch.bool, device=args.device),
                torch.zeros(1, part_dur_frames, dtype=torch.bool, device=args.device),
            ),
            dim=-1,
        )
        offset_frame = end_frame

    # Append remaining mel frames after last edit
    mel_cond = torch.cat((mel_cond, original_mel[:, offset_frame:, :]), dim=1)
    edit_mask = F.pad(edit_mask, (0, mel_cond.shape[1] - edit_mask.shape[-1]), value=True)

    # Text prep
    text_list = [args.target_text]
    if tokenizer_type == "pinyin":
        final_text_list = convert_char_to_pinyin(text_list)
    else:
        final_text_list = [text_list]
    print(f"text  : {text_list}")
    print(f"pinyin: {final_text_list}")

    # Duration
    duration = mel_cond.shape[1]

    # Inference
    seed = None
    with torch.inference_mode():
        generated, trajectory = model.sample(
            cond=mel_cond,
            text=final_text_list,
            duration=duration,
            steps=args.nfe_step,
            cfg_strength=args.cfg_strength,
            sway_sampling_coef=args.sway_sampling_coef,
            seed=seed,
            edit_mask=edit_mask,
        )
        print(f"Generated mel: {generated.shape}")

        # Final result
        generated = generated.to(torch.float32)
        gen_mel_spec = generated.permute(0, 2, 1)
        if args.vocoder_name == "vocos":
            generated_wave = vocoder.decode(gen_mel_spec).cpu()
        elif args.vocoder_name == "bigvgan":
            generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()

        if rms < args.target_rms:
            generated_wave = generated_wave * rms / args.target_rms

        save_spectrogram(gen_mel_spec[0].cpu().numpy(), os.path.join(args.output_dir, f"{args.output_name}.png"))
        torchaudio.save(os.path.join(args.output_dir, f"{args.output_name}.wav"), generated_wave, target_sample_rate)
        print(f"Saved generated audio to: {os.path.join(args.output_dir, f'{args.output_name}.wav')}")

if __name__ == "__main__":
    main()
