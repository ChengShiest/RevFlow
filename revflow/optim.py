"""Muon for trunk matrices, AdamW for conditioning, embeddings and vectors."""

import math

import torch


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, {"lr": lr})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                gradient = param.grad
                state = self.state[param]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(gradient)
                momentum = state["momentum"].mul_(.95).add_(gradient)
                direction = gradient.add(momentum, alpha=.95).bfloat16()
                transpose = direction.shape[0] > direction.shape[1]
                if transpose:
                    direction = direction.T
                direction = direction / (direction.norm() + 1e-7)
                for _ in range(5):
                    gram = direction @ direction.T
                    polynomial = -4.775 * gram + 2.0315 * gram @ gram
                    direction = 3.4445 * direction + polynomial @ direction
                update = direction.T if transpose else direction
                param.add_(update, alpha=-group["lr"])


def build_optimizers(model, config):
    matrices, other = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or any(term in name for term in ("embed", "norm", "modulation", "condition_")):
            other.append(parameter)
        else:
            matrices.append(parameter)
    return (Muon(matrices, config["muon_lr"]),
            torch.optim.AdamW(other, lr=config["adamw_lr"], betas=(.9, .95), weight_decay=.01))


def lr_multiplier(step, steps_per_epoch, config):
    warmup = config["warmup_epochs"] * steps_per_epoch
    total = config["epochs"] * steps_per_epoch
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1., (step - warmup) / max(1, total - warmup))
    floor = config["final_muon_lr"] / config["muon_lr"]
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2
