import torch


def shift_time(time, shift):
    return shift * time / (1 + (shift - 1) * time)


def noisy_latent(clean, shift):
    time = shift_time(torch.randn(len(clean), device=clean.device).sigmoid(), shift)
    noise = torch.randn_like(clean)
    t = time[:, None, None]
    return (1 - t) * clean + t * noise, time


def training_loss(model, hi, lo, labels, config):
    sigma = torch.empty(len(hi), device=hi.device).uniform_(*config["teacher_sigma"])
    sigma_grid = sigma[:, None, None]
    teacher = (1 - sigma_grid.square()).sqrt() * hi + sigma_grid * torch.randn_like(hi)
    noisy_hi, t_hi = noisy_latent(hi, config["hi_shift"])
    noisy_lo, t_lo = noisy_latent(lo, config["lo_shift"])
    final_hi, final_lo, base_hi, base_lo = model(noisy_hi, t_hi, noisy_lo, t_lo, labels, teacher, sigma)

    def velocity_mse(prediction, target, time):
        error = (prediction.float() - target) / time[:, None, None].clamp_min(.05)
        return error.square().mean()

    final = velocity_mse(final_hi, hi, t_hi) + .5 * velocity_mse(final_lo, lo, t_lo)
    base = 2 * velocity_mse(base_hi, hi, t_hi) + velocity_mse(base_lo, lo, t_lo)
    return final + .5 * base


@torch.inference_mode()
def sample(model, labels, flow, sampling, generator=None, initial_noise=None):
    batch, tokens = len(labels), model.side ** 2
    if initial_noise is None:
        initial_noise = [torch.randn(batch, tokens, width, device=labels.device, generator=generator)
                         for width in (model.hi_channels, model.lo_channels)]
    hi, lo = initial_noise
    for stage in ("hi", "lo"):
        if stage == "hi":
            x = hi.float()
            condition = model.hi_condition(labels)
        else:
            x = lo.float()
            condition = model.lo_condition(hi, torch.zeros(batch, device=labels.device))
        steps = sampling[f"{stage}_steps"]
        if steps < 1:
            raise ValueError("Each stage requires at least one Euler step")
        grid = shift_time(torch.linspace(1, 0, steps + 1, device=labels.device), flow[f"{stage}_shift"])
        # Decide the guidance windows once per stage to avoid host syncs inside the loop.
        active = (grid[:-1] > sampling[f"{stage}_min_time"]).tolist()
        scale = sampling[f"{stage}_scale"]
        for i in range(steps):
            final, base = model.forward_stage(x, grid[i].expand(batch), stage, condition)
            estimate = final.float()
            if active[i] and scale != 1:
                estimate = estimate + (scale - 1) * (estimate - base.float())
            velocity = (x - estimate) / grid[i].clamp_min(1e-5)
            x = x + (grid[i + 1] - grid[i]) * velocity
        if stage == "hi":
            hi = x
        else:
            lo = x
    return hi, lo
