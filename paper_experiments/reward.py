import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
import torchaudio
from fastapi import FastAPI
from pydantic import BaseModel
import hashlib
import torch.hub

import torch.nn.functional as F
from typing import Optional, List, Dict

import jiwer

from f5_tts.infer.utils_infer import load_vocoder
from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL


from functools import wraps

import torchaudio
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda *args, **kwargs: None

import sys
from types import ModuleType
if not hasattr(torchaudio, 'sox_effects'):
    fake_sox = ModuleType('torchaudio.sox_effects')
    def fake_apply_effects_tensor(waveform, sample_rate, effects, channels_first=True):
        return waveform, sample_rate
    fake_sox.apply_effects_tensor = fake_apply_effects_tensor
    sys.modules['torchaudio.sox_effects'] = fake_sox


# ------------------------------------------------------------
# Patch 1: Redirect torch.hub.load_state_dict_from_url for wavlm
# ------------------------------------------------------------
_original_load = torch.hub.load_state_dict_from_url

@wraps(_original_load)
def _patched_load(url, map_location=None, *args, **kwargs):
    if "wavlm_large.pt" in url:
        local_path = os.path.join(ROOT_DIR, "ckpts", "wavlm_large_finetune.pth")
        if os.path.exists(local_path):
            return torch.load(local_path, map_location=map_location)
    return _original_load(url, map_location=map_location, *args, **kwargs)

torch.hub.load_state_dict_from_url = _patched_load

# ------------------------------------------------------------
# Patch 2: Fix missing apply_effects_tensor in modern torchaudio
# ------------------------------------------------------------
if not hasattr(torchaudio.functional, 'apply_effects_tensor'):
    torchaudio.functional.apply_effects_tensor = lambda w, sr, e, c: (w, sr)



class RewardPipeline:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # ---------------- Vocoder ----------------
        self.vocoder_name = os.environ.get("REWARD_VOCODER", "vocos").lower()
        if self.vocoder_name not in ["vocos", "bigvgan"]:
            raise ValueError(f"Unsupported REWARD_VOCODER={self.vocoder_name}")

        self.vocoder = load_vocoder(
            vocoder_name=self.vocoder_name,
            is_local=False,
            local_path="",
            device=device,)

        self.resampler = torchaudio.transforms.Resample(
            orig_freq=24000,
            new_freq=16000
        ).to(self.device)

        # ---------------- UTMOS ----------------
        self.utmos = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True
        )
        self.utmos = self.utmos.eval().to(self.device)

        # ---------------- ASR (lazy) ----------------
        self.asr_pipe = None

        self.text_transform = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ])

        # ---------------- SIM-O (ECAPA + WavLM) ----------------


        self.sim_model = None
        self.wavlm_ckpt_path = os.path.join(ROOT_DIR, "ckpts", "wavlm_large_finetune.pth")
        self.ref_embedding_cache = {}
        self.max_ref_cache_size = 256



    # ============================================================
    # Decode mel -> waveform 16k
    # ============================================================

    @torch.no_grad()
    def decode_mel(self, mel, ref_len, ref_rms):
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        mel = mel.to(self.device, dtype=torch.float32)
        mel = mel[:, ref_len:, :]
        mel = mel.permute(0, 2, 1).contiguous()

        waveform = self.vocoder.decode(mel)

        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        waveform = waveform.float()

        target_rms = 0.1
        if ref_rms is not None and ref_rms < target_rms:
            waveform = waveform * (ref_rms / target_rms)

        waveform_16k = self.resampler(waveform)
        return waveform_16k

    # ============================================================
    # MOS
    # ============================================================

    @torch.no_grad()
    def compute_mos(self, waveform_16k):
        waveform_16k = waveform_16k.to(self.device)
        score = self.utmos(waveform_16k, 16000)
        if score.dim() > 1:
            score = score.squeeze(-1)
        return float(score.mean().item())


    def _lazy_load_sim(self):
        if self.sim_model is not None:
            return

        state_dict = torch.load(
            self.wavlm_ckpt_path,
            map_location="cpu",
            weights_only=True
        )

        model = ECAPA_TDNN_SMALL(
            feat_dim=1024,
            feat_type="wavlm_large",
            config_path=None
        )
        
        model.load_state_dict(state_dict["model"], strict=False)
        self.sim_model = model.to(self.device).eval()

    @torch.no_grad()
    def compute_wer(self, waveform_16k, ref_text):
        self._lazy_load_whisper()

        audio_np = waveform_16k.squeeze(0).detach().cpu().numpy()
        result = self.asr_pipe(audio_np)

        pred_text = result["text"]

        ref_norm = self.text_transform(ref_text)
        pred_norm = self.text_transform(pred_text)

        print(f"REF: {ref_text}")
        print(f"PRED: {pred_text}")

        wer = jiwer.wer(ref_norm, pred_norm)

        return float(wer), pred_text

    # ============================================================
    # SIM-O (Seed-TTS compatible)
    # ============================================================

    @torch.no_grad()
    def compute_sim_o(self, gen_waveform, ref_waveform):
        self._lazy_load_sim()
        # ensure mono
        if gen_waveform.dim() == 2 and gen_waveform.size(0) > 1:
            gen_waveform = gen_waveform.mean(0, keepdim=True)

        if ref_waveform.dim() == 2 and ref_waveform.size(0) > 1:
            ref_waveform = ref_waveform.mean(0, keepdim=True)

        gen_waveform = gen_waveform.to(self.device)
        ref_waveform = ref_waveform.to(self.device)

        emb1 = self.sim_model(gen_waveform)
        emb2 = self.sim_model(ref_waveform)

        sim = F.cosine_similarity(emb1, emb2)[0]

        return float(sim.item())

    def _make_ref_cache_key(self, ref_mel_tensor, ref_rms):
        ref_bytes = ref_mel_tensor.detach().cpu().contiguous().numpy().tobytes()
        digest = hashlib.sha1(ref_bytes).hexdigest()
        return (digest, tuple(ref_mel_tensor.shape), ref_rms)

    @torch.no_grad()
    def get_cached_ref_embedding(self, ref_mel_tensor, ref_rms):
        self._lazy_load_sim()

        key = self._make_ref_cache_key(ref_mel_tensor, ref_rms)
        cached = self.ref_embedding_cache.get(key)
        if cached is not None:
            return cached

        ref_wave = self.decode_mel(ref_mel_tensor, 0, ref_rms)
        if ref_wave.dim() == 2 and ref_wave.size(0) > 1:
            ref_wave = ref_wave.mean(0, keepdim=True)

        ref_wave = ref_wave.to(self.device)
        ref_embedding = self.sim_model(ref_wave).detach()

        if len(self.ref_embedding_cache) >= self.max_ref_cache_size:
            oldest_key = next(iter(self.ref_embedding_cache))
            self.ref_embedding_cache.pop(oldest_key, None)

        self.ref_embedding_cache[key] = ref_embedding
        return ref_embedding

    @torch.no_grad()
    def compute_sim_o_from_embedding(self, gen_waveform, ref_embedding):
        self._lazy_load_sim()

        if gen_waveform.dim() == 2 and gen_waveform.size(0) > 1:
            gen_waveform = gen_waveform.mean(0, keepdim=True)

        gen_waveform = gen_waveform.to(self.device)
        emb1 = self.sim_model(gen_waveform)
        sim = F.cosine_similarity(emb1, ref_embedding.to(self.device))[0]
        return float(sim.item())


# ============================================================
# FastAPI
# ============================================================

app = FastAPI()
reward_device = os.environ.get("REWARD_DEVICE", "cuda")
print(f"[reward] loading pipeline on device={reward_device}")
reward_pipeline = RewardPipeline(device=reward_device)


class RewardRequest(BaseModel):
    reward_types: List[str]
    weights: Optional[Dict[str, float]] = None
    mel: Optional[list] = None
    ref_mel: Optional[list] = None
    ref_len: Optional[int] = None
    ref_rms: Optional[float] = None
    ref_text: Optional[str] = None


@app.post("/infer")
async def infer(req: RewardRequest):

    results = {}
    gen_wave = None

    if any(r in ["mos", "wer", "sim-o"] for r in req.reward_types):

        if req.mel is None or req.ref_len is None:
            return {"error": "Missing mel or ref_len"}

        mel_tensor = torch.tensor(req.mel).float().unsqueeze(0)

        gen_wave = reward_pipeline.decode_mel(
            mel_tensor,
            req.ref_len,
            req.ref_rms
        )

    if "mos" in req.reward_types:
        results["mos"] = reward_pipeline.compute_mos(gen_wave)

    if "wer" in req.reward_types:
        if req.ref_text is None:
            return {"error": "ref_text required for WER"}

        wer_score, pred_text = reward_pipeline.compute_wer(
            gen_wave,
            req.ref_text
        )

        results["wer"] = wer_score
        results["prediction"] = pred_text

    if "sim-o" in req.reward_types:
        if req.ref_mel is None:
            return {"error": "ref_mel required for sim-o"}

        ref_mel_tensor = torch.tensor(req.ref_mel).float().unsqueeze(0)
        ref_embedding = reward_pipeline.get_cached_ref_embedding(
            ref_mel_tensor,
            req.ref_rms
        )
        sim_score = reward_pipeline.compute_sim_o_from_embedding(gen_wave, ref_embedding)
        results["sim-o"] = sim_score

    if req.weights is not None:
        total = 0.0
        for key, weight in req.weights.items():
            if key in results:
                total += weight * results[key]
        results["aggregated_score"] = total

    return results

# uvicorn reward:app --host 0.0.0.0 --port 8000
