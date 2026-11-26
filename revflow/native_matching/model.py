"""Native two-evaluation x-prediction and the fixed parent readout chart."""

import torch
from torch import nn


def guided_pair(model, latent, time, labels, hi_scale, lo_scale):
    hi, lo = latent.split([model.hi_channels, model.lo_channels], dim=-1)
    final, base = model.forward_stage(hi, time, "hi", model.hi_condition(labels))
    hi = final.float() + (hi_scale - 1) * (final.float() - base.float())
    condition = model.lo_condition(hi, torch.zeros_like(time))
    final, base = model.forward_stage(lo, time, "lo", condition)
    lo = final.float() + (lo_scale - 1) * (final.float() - base.float())
    return torch.cat((hi, lo), dim=-1)


def readouts(model):
    return {"hi": model.ddt_hi.linear, "lo": model.final_lo.linear}


class Chart(nn.Module):
    def __init__(self, state):
        super().__init__()
        for stage in ("hi", "lo"):
            for key in ("inverse", "bias", "mean", "std"):
                self.register_buffer(stage + "_" + key, state[stage][key].float())
            if not (state[stage]["std"] > 0).all():
                raise ValueError("Chart standard deviations must be positive")

    def forward(self, raw):
        with torch.autocast(raw.device.type, enabled=False):
            hi, lo = raw.float().split((self.hi_bias.numel(), self.lo_bias.numel()), dim=-1)
            hi = ((hi - self.hi_bias) @ self.hi_inverse - self.hi_mean) / self.hi_std
            lo = ((lo - self.lo_bias) @ self.lo_inverse - self.lo_mean) / self.lo_std
            return torch.cat((hi, lo), dim=-1)
