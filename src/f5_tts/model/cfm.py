"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import os
from random import random
from typing import Callable
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
import random
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)
import requests

class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
        reward_url: str | None = None,
        reward_timeout: float = 120.0,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

        # reward service config
        self.reward_url = reward_url or os.environ.get("MRM_REWARD_URL", "http://127.0.0.1:8000/infer")
        reward_urls_env = os.environ.get("MRM_REWARD_URLS")
        if reward_urls_env:
            self.reward_urls = [url.strip() for url in reward_urls_env.split(",") if url.strip()]
        else:
            self.reward_urls = [url.strip() for url in self.reward_url.split(",") if url.strip()]
        if not self.reward_urls:
            self.reward_urls = ["http://127.0.0.1:8000/infer"]
        self.reward_timeout = reward_timeout
        self.reward_session = requests.Session()

        # load MOS model once
        # self.mos_pipeline = LocalMOSPipeline(device="cuda")

    @property
    def device(self):
        return next(self.parameters()).device


    def reward(
        self,
        x0,
        ref_audio_len,
        cond_mask,
        cond,
        ref_rms=None,
        reward_types=("mos",),
        weights=None,
        ref_texts=None,
    ):
        x0 = torch.where(cond_mask, cond, x0)

        if x0.dim() == 2:
            x0 = x0.unsqueeze(0)

        B = x0.size(0)
        device = x0.device

        rewards = []
        payloads = []

        for i in range(B):

            gen_mel = x0[i].detach().cpu().float().tolist()

            payload = {
                "reward_types": list(reward_types),
                "mel": gen_mel,
                "ref_len": ref_audio_len,
                "ref_rms": float(ref_rms) if ref_rms is not None else None,
            }

            if "sim-o" in reward_types:
                ref_mel = cond[i, :ref_audio_len].detach().cpu().float().tolist()
                payload["ref_mel"] = ref_mel

            if "wer" in reward_types:
                if ref_texts is None:
                    raise ValueError("ref_texts required for WER")
                if isinstance(ref_texts, str):
                    payload["ref_text"] = ref_texts
                else:
                    payload["ref_text"] = ref_texts[i]

            if weights is not None:
                payload["weights"] = weights

            payloads.append(payload)

        def request_reward(args):
            idx, payload = args
            url = self.reward_urls[idx % len(self.reward_urls)]
            response = requests.post(
                url,
                json=payload,
                timeout=self.reward_timeout,
            )

            if response.status_code != 200:
                raise RuntimeError(response.text)

            result = response.json()

            if weights is not None:
                return result["aggregated_score"]

            return result[reward_types[0]]

        if len(payloads) == 1 and len(self.reward_urls) == 1:
            response = self.reward_session.post(
                self.reward_urls[0],
                json=payloads[0],
                timeout=self.reward_timeout,
            )

            if response.status_code != 200:
                raise RuntimeError(response.text)

            result = response.json()

            if weights is not None:
                rewards.append(result["aggregated_score"])
            else:
                rewards.append(result[reward_types[0]])
        else:
            max_workers = min(len(payloads), len(self.reward_urls))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                rewards = list(executor.map(request_reward, enumerate(payloads)))

        return torch.tensor(rewards, device=device, dtype=x0.dtype)


    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        # raw wave
        # print("text", text)
        joined_text = ''.join(c for sub in text for c in sub)
        ref_text = joined_text.split('. ')[-1].strip()
        print("result_text", ref_text)
        rms = None
        print("condshape",cond.shape)

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        ref_len = cond.shape[1]
        print("ref_len for reward:", ref_len)

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode
        v_store = []

        def fn(t, x):

            if cfg_strength < 1e-5:
                v = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                )
            else:
                pred_cfg = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    cfg_infer=True,
                    cache=True,
                )
                pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
                v = pred + (pred - null_pred) * cfg_strength

            # DEBUG
            # print("t:", t.item() if torch.numel(t)==1 else t)
            # print("mean|x|", x.abs().mean().item())
            # print("mean|v|", v.abs().mean().item())
            v_store.append(v.detach())

            return v

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            # torch.manual_seed(42)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0
        # use_epss=True
        # print("???", use_epss, t_start)
        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        # t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        # print("T", t)
        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()
        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        print(trajectory.shape)
        print("debug", trajectory)
        for i in range(trajectory.shape[0]):
            print(f"Step {i} - Reward:", self.reward(trajectory[i], ref_len, cond_mask, cond).item())


        # print(len(v_store), trajectory.shape[0])
        # print(t)
        # num = min(len(v_store), trajectory.shape[0])

        # for i in range(num):
        #     x_i = trajectory[i]
        #     v_i = v_store[i]
        #     t_i = t[i]

        #     x_pred = x_i + (1 - t_i) * v_i


        #     wer_value = self.reward(
        #         x0=x_pred,
        #         ref_audio_len=ref_len,
        #         cond_mask=cond_mask,
        #         cond=cond,
        #         ref_rms=rms,
        #         reward_types=("wer",),
        #         weights={"wer": 1.0},      # nếu muốn maximize
        #         ref_texts=[ref_text],       # ⚠ bắt buộc phải là list
        #     )

            # print(f"Step {i} | t={t_i.item():.4f} |  SIM-O={sim_value.item():.4f}")
            # print(
            #     f"Step {i} | t={t_i.item():.4f} "
            #     # f"| mean||v||={torch.norm(v_i, dim=-1).mean().item():.6f} "
            #     f"| tweedie_reward={r.item():.6f}"
            # )


            # log_path = "/home/ntluong/workspace/F5-TTS/plot_wer_tweedie.txt"

            # with open(log_path, "a", encoding="utf-8") as f:
            #     f.write(
            #         f"Step {i} | t={t_i.item():.4f} "
            #         f"|  wer={wer_value.item():.4f}\n"
            #     )

        # wer_value = self.reward(
        #     x0=out,
        #     ref_audio_len=ref_len,
        #     cond_mask=cond_mask,
        #     cond=cond,
        #     ref_rms=rms,
        #     reward_types=("wer",),
        #     weights={"wer": -1.0},      # nếu muốn maximize
        #     ref_texts=[ref_text],       # ⚠ bắt buộc phải là list
        # )

        # sim_value = self.reward(
        #     x0=out,
        #     ref_audio_len=ref_len,
        #     cond_mask=cond_mask,
        #     cond=cond,          # ⚠ bắt buộc vì sim-o dùng cond làm ref_mel
        #     ref_rms=rms,
        #     reward_types=("sim-o",),
        # )
        # mos_value = self.reward(
        #     x0=out,
        #     ref_audio_len=ref_len,
        #     cond_mask=cond_mask,
        #     cond=cond,
        #     ref_rms=rms,
        #     reward_types=("mos",),
        # )
        # print(f"Final WER: {wer_value.item():.4f}", f"Final SIM-O: {sim_value.item():.4f}, Final MOS: {mos_value.item():.4f}")
        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        *,
        lens: int["b"] | None = None,  # noqa: F821
        noise_scheduler: str | None = None,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred



class SearchStrategy:

    def run(
        self,
        model,
        y0,
        timesteps,
        fn,
        reward_fn,
        context,
    ):
        raise NotImplementedError


class RBFSearch(SearchStrategy):

    def __init__(self, branch_num=4):
        self.branch_num = branch_num

    def run(
        self,
        model,
        y0,
        timesteps,
        fn,
        reward_fn,
        context,
    ):
        x_s = y0
        r_star = reward_fn(x_s)

        M = len(timesteps) - 1
        Q = [self.branch_num] * M
        trajectory = [x_s]

        for i in range(M):
            s = timesteps[i]
            s_next = timesteps[i + 1]
            dt = s_next - s
            q = Q[i]

            best_local_reward = -1e9
            best_local_sample = None

            v = fn(s, x_s)

            for j in range(1, q + 1):
                drift = v

                if model.sample_method == "sde":
                    score = model.convert_velocity_to_score(v, s, x_s)
                    g = model.get_diffuse(s)
                    while g.ndim < x_s.ndim:
                        g = g.unsqueeze(-1)
                    drift = v - 0.5 * (g ** 2) * score

                x_candidate = x_s + drift * dt

                if model.sample_method == "sde":
                    g = model.get_diffuse(s)
                    while g.ndim < x_s.ndim:
                        g = g.unsqueeze(-1)
                    noise = torch.randn_like(x_s)
                    x_candidate = x_candidate + g * torch.sqrt(
                        torch.clamp(dt, min=1e-8)
                    ) * noise

                r_candidate = reward_fn(x_candidate, reward_types=("mos",))

                if r_candidate > r_star:
                    if i + 1 < M:
                        Q[i + 1] += (Q[i] - j)
                    r_star = r_candidate
                    x_s = x_candidate
                    break

                if r_candidate > best_local_reward:
                    best_local_reward = r_candidate.item()
                    best_local_sample = x_candidate

                if j == q:
                    x_s = best_local_sample

            trajectory.append(x_s)

        return [x_s], trajectory


class BestOfNSearch(SearchStrategy):

    def __init__(self, N):
        self.N = N

    def run(
        self,
        model,
        y0,
        timesteps,
        fn,
        reward_fn,
        context,
    ):
        particles = y0  # (batch * N, T, C)
        trajectory = []

        B_times_N = particles.shape[0]
        N = self.N
        assert B_times_N % N == 0, "y0 must be batch*N"

        B = B_times_N // N

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            particles = model.propagate_step(particles, t, t_next, fn)
            trajectory.append(particles)

        rewards = reward_fn(particles, reward_types=("mos",))  # (B*N,)

        # reshape về (B, N)
        rewards = rewards.view(B, N)

        print("choice rewards:", rewards)
        # chọn best particle trong mỗi batch
        best_idx = torch.argmax(rewards, dim=1)  # (B,)

        # reshape particles về (B, N, T, C)
        particles = particles.view(B, N, *particles.shape[1:])

        best = particles[
            torch.arange(B, device=particles.device),
            best_idx
        ]  # (B, T, C)

        return [best], trajectory

class SMCSearch(SearchStrategy):

    def __init__(self, N, beta=1.0, ess_threshold=0.5):
        self.N = N
        self.beta = beta
        self.ess_threshold = ess_threshold

    def run(
        self,
        model,
        y0,
        timesteps,
        fn,
        reward_fn,
        context,
    ):
        particles = y0  # (B*N, T, C)

        B_times_N = particles.shape[0]
        N = self.N
        assert B_times_N % N == 0

        B = B_times_N // N

        weights = torch.ones(B, N, device=particles.device) / N

        T, C = particles.shape[1], particles.shape[2]

        trajectory = []

        for i in range(len(timesteps) - 1):

            t = timesteps[i]
            t_next = timesteps[i + 1]

            old_reward = reward_fn(particles, reward_types=("sim-o",)).view(B, N)

            particles = model.propagate_step(
                particles, t, t_next, fn
            )

            new_reward = reward_fn(particles, reward_types=("sim-o",)).view(B, N)
            print(f"[MRM][step {i}] t={t.item():.4f}")
            print("  MOS old mean:", old_reward.mean().item())
            print("  MOS new mean:", new_reward.mean().item())
            print("  MOS best:", new_reward.max().item())

            log_ratio = (new_reward - old_reward) / self.beta
            weights = weights * torch.exp(log_ratio)
            weights = weights / weights.sum(dim=1, keepdim=True)

            ess = 1.0 / torch.sum(weights ** 2, dim=1)

            resample_mask = ess < (self.ess_threshold * N)

            if resample_mask.any():

                particles_reshaped = particles.view(B, N, T, C)

                new_particles = []

                for b in range(B):
                    if resample_mask[b]:
                        idx = torch.multinomial(
                            weights[b], N, replacement=True
                        )
                        new_particles.append(particles_reshaped[b, idx])
                        weights[b] = torch.ones(
                            N, device=particles.device
                        ) / N
                    else:
                        new_particles.append(particles_reshaped[b])

                particles = torch.stack(new_particles, dim=0)
                particles = particles.view(B * N, T, C)

            trajectory.append(particles)

        final_rewards = reward_fn(particles, reward_types=("sim-o",)).view(B, N)

        best_idx = torch.argmax(final_rewards, dim=1)

        particles = particles.view(B, N, T, C)

        best = particles[
            torch.arange(B, device=particles.device),
            best_idx
        ]

        return [best], trajectory

class MRMSearch(SearchStrategy):

    def __init__(self, N, beta=1.0, ess_threshold=0.5, t1=0.5, t2=0.8):
        if not 0.0 <= t1 < t2 <= 1.0:
            raise ValueError(f"Expected 0 <= t1 < t2 <= 1, got t1={t1}, t2={t2}")
        self.N = N
        self.beta = beta
        self.ess_threshold = ess_threshold
        self.t1 = t1
        self.t2 = t2

    def run(
        self,
        model,
        y0,
        timesteps,
        fn,
        reward_fn,
        context,
    ):
        particles = y0  # (B*N, T, C)

        B_times_N = particles.shape[0]
        N = self.N
        assert B_times_N % N == 0

        B = B_times_N // N
        T, C = particles.shape[1], particles.shape[2]

        weights = torch.ones(B, N, device=particles.device) / N

        trajectory = []

        print(
            f"[MRM] search config: N={self.N}, t1={self.t1:.4f}, "
            f"t2={self.t2:.4f}, beta={self.beta}, ess={self.ess_threshold}"
        )

        for i in range(len(timesteps) - 1):

            t = timesteps[i]
            t_next = timesteps[i + 1]

            # -------- PHASE 1: pure parallel denoise (t <= t1)
            if t_next <= self.t1:

                particles = model.propagate_step(
                    particles, t, t_next, fn
                )

            # -------- PHASE 2: SMC (t1 < t <= t2)
            elif t < self.t2:

                old_reward = reward_fn(
                    particles,
                    reward_types=("sim-o",)
                ).view(B, N)

                particles = model.propagate_step(
                    particles, t, t_next, fn
                )

                new_reward = reward_fn(
                    particles,
                    reward_types=("sim-o",)
                ).view(B, N)

                log_ratio = (new_reward - old_reward) / self.beta
                weights = weights * torch.exp(log_ratio)
                weights = weights / weights.sum(dim=1, keepdim=True)

                ess = 1.0 / torch.sum(weights ** 2, dim=1)
                resample_mask = ess < (self.ess_threshold * N)

                if resample_mask.any():

                    particles_reshaped = particles.view(B, N, T, C)

                    new_particles = []

                    for b in range(B):
                        if resample_mask[b]:
                            idx = torch.multinomial(
                                weights[b], N, replacement=True
                            )
                            new_particles.append(
                                particles_reshaped[b, idx]
                            )
                            weights[b] = torch.ones(
                                N, device=particles.device
                            ) / N
                        else:
                            new_particles.append(
                                particles_reshaped[b]
                            )

                    particles = torch.stack(new_particles, dim=0)
                    particles = particles.view(B * N, T, C)

            # -------- PHASE 3: BoN-like final refinement (t > t2)
            else:

                particles = model.propagate_step(
                    particles, t, t_next, fn
                )

            trajectory.append(particles)

        # -------- FINAL SELECTION (BoN style)
        final_rewards = reward_fn(
            particles,
            reward_types=("mos",)
        ).view(B, N)

        best_idx = torch.argmax(final_rewards, dim=1)

        particles = particles.view(B, N, T, C)

        best = particles[
            torch.arange(B, device=particles.device),
            best_idx
        ]

        return [best], trajectory

class CFM_SDE(CFM):

    def __init__(
        self,
        *args,
        sample_method="sde",
        diffusion_norm=0.03,
        diffusion_schedule="square",
        search_type="mrm",
        branch_num=4,
        search_N=8,
        mrm_t1=0.5,
        mrm_t2=0.8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.sample_method = sample_method
        self.diffusion_norm = diffusion_norm
        self.diffusion_schedule = diffusion_schedule

        if search_type == "rbf":
            self.search_strategy = RBFSearch(branch_num=branch_num)
        elif search_type == "bon":
            self.search_strategy = BestOfNSearch(N=search_N)
        elif search_type == "smc":
            self.search_strategy = SMCSearch(N=search_N)
        elif search_type == "mrm":
            self.search_strategy = MRMSearch(
                N=search_N,
                beta=1.0,
                ess_threshold=0.5,
                t1=mrm_t1,
                t2=mrm_t2,
            )
        elif search_type is None or search_type == "none":
            self.search_strategy = None
        else:
            raise ValueError

    def propagate_step(self, x, t, t_next, fn):
        dt = t_next - t
        v = fn(t, x)

        drift = v

        if self.sample_method == "sde":
            score = self.convert_velocity_to_score(v, t, x)
            g = self.get_diffuse(t)
            while g.ndim < x.ndim:
                g = g.unsqueeze(-1)
            drift = v - 0.5 * (g ** 2) * score

        x_next = x + drift * dt

        if self.sample_method == "sde":
            g = self.get_diffuse(t)
            while g.ndim < x.ndim:
                g = g.unsqueeze(-1)
            noise = torch.randn_like(x)
            x_next = x_next + g * torch.sqrt(
                torch.clamp(dt, min=1e-8)
            ) * noise

        return x_next

    def convert_velocity_to_score(self, velocity, t, sample):

        orig_dtype = sample.dtype

        velocity = velocity.float()
        sample = sample.float()

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=sample.device, dtype=torch.float32)
        else:
            t = t.float()

        while t.ndim < sample.ndim:
            t = t.unsqueeze(-1)

        var = torch.clamp(t, min=1e-3)  # IMPORTANT luongnt29 need to check later

        reverse_alpha_ratio = -(1 - t)

        score = (reverse_alpha_ratio * velocity - sample) / var

        return score.to(orig_dtype)

    def get_diffuse(self, t):

        if self.diffusion_schedule == "linear":
            return self.diffusion_norm * (1-t)

        if self.diffusion_schedule == "square":
            return self.diffusion_norm * ((1-t) ** 2)

        if self.diffusion_schedule == "constant":
            return torch.ones_like(t) * self.diffusion_norm

        raise ValueError


    @torch.no_grad()
    def sample(
        self,
        cond,
        text,
        duration,
        *,
        lens=None,
        steps=8,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed=None,
        max_duration=4096,
        vocoder=None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        if getattr(self, "search_strategy", None) is None:
            return super().sample(
                cond=cond,
                text=text,
                duration=duration,
                lens=lens,
                steps=steps,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
                max_duration=max_duration,
                vocoder=vocoder,
                use_epss=use_epss,
                no_ref_audio=no_ref_audio,
                duplicate_test=duplicate_test,
                t_inter=t_inter,
                edit_mask=edit_mask,
            )

        self.eval()
        # print(f"Text: {text}")
        joined_text = ''.join(c for sub in text for c in sub)
        ref_text = joined_text.split('. ')[-1].strip()
        rms = None

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels
        ref_len = cond.shape[1]

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device

        if lens is None:
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        if isinstance(text, list):
            if self.vocab_char_map is not None:
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)

        cond_mask = lens_to_mask(lens)

        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1,
            duration,
        )

        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        cond = torch.nn.functional.pad(
            cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0
        )

        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = torch.nn.functional.pad(
            cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False
        )
        cond_mask = cond_mask.unsqueeze(-1)

        if batch > 1:
            mask = lens_to_mask(duration)
        else:
            mask = None

        if seed is not None:
            torch.manual_seed(seed)

        num_particles = 1
        if isinstance(self.search_strategy, (BestOfNSearch, SMCSearch, MRMSearch)):
            num_particles = self.search_strategy.N

        B = batch
        N = num_particles

        y0_list = []
        for b in range(B):
            per_particle = []
            for _ in range(N):
                per_particle.append(
                    torch.randn(
                        duration[b],
                        self.num_channels,
                        device=device,
                        dtype=cond.dtype,
                    )
                )
            per_particle = torch.nn.utils.rnn.pad_sequence(
                per_particle,
                padding_value=0,
                batch_first=True
            )
            y0_list.append(per_particle)

        y0 = torch.stack(y0_list, dim=0)
        y0 = y0.view(B * N, *y0.shape[2:])

        cond_orig = cond.clone()
        cond_mask_orig = cond_mask.clone()

        if N > 1:
            cond = cond.unsqueeze(1).expand(B, N, -1, -1)
            cond = cond.reshape(B * N, *cond.shape[2:])

            cond_mask = cond_mask.unsqueeze(1).expand(B, N, -1, -1)
            cond_mask = cond_mask.reshape(B * N, *cond_mask.shape[2:])

            if mask is not None:
                mask = mask.unsqueeze(1).expand(B, N, -1)
                mask = mask.reshape(B * N, mask.shape[-1])

            if text is not None:
                text = text.unsqueeze(1).expand(B, N, -1)
                text = text.reshape(B * N, text.shape[-1])

        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        def fn(t, x):
            if cfg_strength < 1e-5:
                v = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                )
            else:
                pred_cfg = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    cfg_infer=True,
                    cache=True,
                )
                pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
                v = pred + (pred - null_pred) * cfg_strength
            return v

        if use_epss:
            t = get_epss_timesteps(
                steps, device=device, dtype=cond.dtype
            )
        else:
            t = torch.linspace(
                0, 1, steps + 1,
                device=device,
                dtype=cond.dtype
            )

        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (
                torch.cos(torch.pi / 2 * t) - 1 + t
            )


        def reward_fn(x, reward_types=("mos",), weights=None):
            return self.reward(
                x0=x,
                ref_audio_len=ref_len,
                cond_mask=cond_mask,
                cond=cond,
                ref_rms=rms,
                reward_types=reward_types,
                weights=weights,
                ref_texts=ref_text,
            )

        final_particles, trajectory = self.search_strategy.run(
            model=self,
            y0=y0,
            timesteps=t,
            fn=fn,
            reward_fn=reward_fn,
            context=None,
        )

        sampled = final_particles[0]

        self.transformer.clear_cache()

        out = torch.where(cond_mask_orig, cond_orig, sampled)

        if vocoder is not None:
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        final_rewards = self.reward(
            out,
            ref_len,
            cond_mask_orig,
            cond_orig,
            ref_rms=rms
        )
        print("Final reward:", final_rewards)
        return out, trajectory
