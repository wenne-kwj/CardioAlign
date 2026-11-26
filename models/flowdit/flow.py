"""Independent 3-channel Gaussian source; x-prediction / velocity loss.

Time direction: 0=noise, 1=clean fluorescence. The loss uses the same clipped
denominator for target and prediction. The ODE uses the analytic denominator
1-t, evaluates only t<1, and uses an Euler final step. Thus an exact clean-image
oracle reaches its endpoint exactly, including with a one-step Euler solver.
"""
from contextlib import nullcontext
import math

import torch
from .pcc import clean_image_pcc_loss


def autocast(device, precision):
    if precision not in ("float32", "bfloat16"):
        raise ValueError("Unsupported numerical precision")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if precision == "bfloat16" else nullcontext()


class ConditionalFlow:
    def __init__(self, loss_denominator_floor=0.05, time_distribution="uniform",
                 logit_mean=-0.8, logit_std=0.8, sampling=None, uniform_probability=0.5,
                 pcc_weight=0.0, pcc_prediction_variance_floor=1e-6, pcc_target_variance_min=1e-12):
        if not 0 < loss_denominator_floor <= 1:
            raise ValueError("Invalid loss denominator floor")
        if time_distribution not in ("uniform", "logit_normal", "logit_normal_uniform_mixture") or logit_std <= 0:
            raise ValueError("Unsupported time distribution")
        if not 0 <= uniform_probability <= 1:
            raise ValueError("Uniform mixture probability must be in [0,1]")
        self.loss_denominator_floor = loss_denominator_floor
        self.time_distribution, self.logit_mean, self.logit_std = time_distribution, logit_mean, logit_std
        self.uniform_probability = uniform_probability
        if not math.isfinite(pcc_weight) or pcc_weight < 0:
            raise ValueError('PCC weight must be finite and nonnegative')
        if any(not math.isfinite(v) or v <= 0 for v in
               (pcc_prediction_variance_floor, pcc_target_variance_min)):
            raise ValueError('Invalid PCC variance safeguards')
        self.pcc_weight = pcc_weight
        self.pcc_prediction_variance_floor = pcc_prediction_variance_floor
        self.pcc_target_variance_min = pcc_target_variance_min
        self.sampling = dict(sampling or {"method": "euler", "steps": 50, "clip_x0": True})
        if self.sampling["method"] not in ("euler", "heun") or not 1 <= self.sampling["steps"] <= 10000:
            raise ValueError("Expected 1..10000 Euler/Heun steps")
        self.sampling_steps = self.sampling["steps"]

    @property
    def path_spec(self):
        return {"type": "gaussian_to_fluorescence_linear", "noise_scale": 1.0,
                "time_zero": "independent_three_channel_gaussian", "time_one": "clean_fluorescence",
                "model_coordinates": "2*x-1", "prediction": "clean_fluorescence"}

    def initial_state(self, bf, noise):
        """Source distribution hook; the original flow starts at pure noise."""
        return noise.float()

    def sample_times(self, batch, device, generator, mixture_generator=None):
        if self.time_distribution == "uniform":
            return torch.rand(batch, device=device, generator=generator)
        original = torch.sigmoid(torch.randn(batch, device=device, generator=generator) * self.logit_std + self.logit_mean)
        if self.time_distribution == "logit_normal":
            return original
        # A dedicated saved RNG preserves the original normal-time / IF-noise
        # stream, exactly as in the validated paired time-coverage experiment.
        mixing_rng = mixture_generator if mixture_generator is not None else generator
        uniform_branch = torch.rand(batch, device=device, generator=mixing_rng) < self.uniform_probability
        uniform = torch.rand(batch, device=device, generator=mixing_rng)
        return torch.where(uniform_branch, uniform, original)

    @staticmethod
    def interpolate(target, time, noise):
        if target.shape != noise.shape or target.ndim != 4 or target.shape[1] != 3 or time.shape != (target.shape[0],):
            raise ValueError("Noise must match the THREE-channel target exactly; broadcasting is forbidden")
        if not torch.isfinite(time).all() or (time < 0).any() or (time > 1).any():
            raise ValueError("Flow time must be finite in [0,1]")
        t = time.float()[:, None, None, None]
        return (1 - t) * noise.float() + t * target.float()

    def loss(self, model, bf, target, *, generator=None, mixture_generator=None, precision="float32", time=None, noise=None):
        if target.ndim != 4 or target.shape[1] != 3 or bf.shape != (target.shape[0], 1, *target.shape[-2:]):
            raise ValueError("Expected one BF channel and three target IF channels")
        if time is None:
            time = self.sample_times(target.shape[0], target.device, generator, mixture_generator)
        if noise is None:
            noise = torch.randn(target.shape, device=target.device, generator=generator, dtype=torch.float32)
        state = self.interpolate(target, time, self.initial_state(bf, noise))
        with autocast(target.device, precision):
            clean = model(state, time, bf)
        if clean.shape != target.shape:
            raise ValueError("Model must predict clean three-channel IF")
        error = clean.float() - target.float()
        denominator = (1 - time.float()).clamp_min(self.loss_denominator_floor)[:, None, None, None]
        # Algebraically identical to MSE((clean-state)/d, (target-state)/d),
        # with less cancellation and all sensitive arithmetic in FP32.
        flow_mse = (error / denominator).square().mean()
        details = {"flow_mse": flow_mse.detach(), "clean_mse": error.square().mean().detach(),
                   "sampled_times": time.detach()}
        if not self.pcc_weight:
            return flow_mse, details
        pcc, pcc_details = clean_image_pcc_loss(clean, target,
            self.pcc_prediction_variance_floor, self.pcc_target_variance_min)
        details.update(pcc_loss=pcc.detach(), **pcc_details)
        return flow_mse + self.pcc_weight * pcc, details

    @torch.inference_mode()
    def sample(self, model, bf, noise_source=None, precision="float32", *, generator=None):
        if bf.ndim != 4 or bf.shape[1] != 1:
            raise ValueError("Sampling accepts BF [B,1,H,W], never a fluorescence target")
        b, _, h, w = bf.shape
        noise = noise_source(0) if callable(noise_source) else noise_source
        if noise is None:
            noise = torch.randn((b, 3, h, w), device=bf.device, generator=generator)
        if noise.shape != (b, 3, h, w):
            raise ValueError("Initial noise must have exactly three independent channel slots")
        noise = noise.to(device=bf.device, dtype=torch.float32)
        state = self.initial_state(bf, noise).clone()
        if not torch.isfinite(state).all():
            raise FloatingPointError("Non-finite initial noise")
        original = getattr(model, "_orig_mod", model)
        with autocast(bf.device, precision):
            context = original.encode_condition(bf) if hasattr(original, "encode_condition") else None
        def velocity(value, t):
            times = torch.full((b,), t, device=bf.device, dtype=torch.float32)
            with autocast(bf.device, precision):
                clean = model(value, times, bf, context=context) if hasattr(original, "encode_condition") else model(value, times, bf)
            clean = clean.float()
            if clean.shape != value.shape:
                raise ValueError("Sampler model output shape mismatch")
            if self.sampling.get("clip_x0", True):
                clean = clean.clamp(-1, 1)
            return (clean - value) / (1 - t)
        for i in range(self.sampling_steps):
            t, next_t = i / self.sampling_steps, (i + 1) / self.sampling_steps
            dt = next_t - t
            v = velocity(state, t)
            proposal = state + dt * v
            if self.sampling["method"] == "heun" and i < self.sampling_steps - 1:
                state = state + dt * 0.5 * (v + velocity(proposal, next_t))
            else:
                state = proposal
        if not torch.isfinite(state).all():
            raise FloatingPointError("Non-finite flow sample")
        return state

    @property
    def nfe(self):
        return self.sampling_steps if self.sampling["method"] == "euler" else 2 * self.sampling_steps - 1
