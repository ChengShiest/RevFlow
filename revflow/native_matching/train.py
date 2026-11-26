"""Adapt an 80-epoch RevFlow DDT EMA by native representation matching."""

import argparse
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from revflow.checkpoints import atomic_save, load_weights, read_config
from revflow.models import RevFlow
from .field import distribution_gradient
from .model import Chart, guided_pair, readouts
from .support import Support


def optimizers_for(model, cfg):
    from gram_newton_schulz import Muon
    params = [p for p in model.parameters() if p.requires_grad]
    muon = Muon([p for p in params if p.ndim == 2], lr=cfg["muon_lr"], momentum=.95, nesterov=True,
                weight_decay=0., ns_coefficients_preset="POLAR_EXPRESS_COEFFICIENTS",
                ns_use_kernels=False, adjust_lr="rms_norm")
    adamw = torch.optim.AdamW([p for p in params if p.ndim != 2], lr=cfg["adamw_lr"],
                              betas=(.9, .95), weight_decay=0.)
    return muon, adamw


def generators(rank):
    seeds = dict(rows=420001, noise=420101, ref=951201, fm=951001, time=951101, cond=952301)
    rng = {k: torch.Generator().manual_seed(v + rank) for k, v in seeds.items()}
    rng["classes"] = torch.Generator().manual_seed(42)
    return rng


def average_gradients(params, world):
    if world == 1:
        return
    # Bound the size of temporary contiguous buffers during reduction.
    bucket = []
    size = 0
    def flush():
        flat = torch.cat([g.reshape(-1) for g in bucket])
        dist.all_reduce(flat)
        flat.div_(world)
        for grad, chunk in zip(bucket, flat.split([g.numel() for g in bucket])):
            grad.copy_(chunk.view_as(grad))
    for param in params:
        if param.grad is None:
            continue
        bucket.append(param.grad)
        size += param.numel()
        if size >= 8_000_000:
            flush()
            bucket.clear()
            size = 0
    if bucket:
        flush()


def fm_loss(model, z, labels, time, noise, sigma, teacher_noise):
    hi, lo = z.split([model.hi_channels, model.lo_channels], -1)
    eps_hi, eps_lo = noise.split([model.hi_channels, model.lo_channels], -1)
    t = time[:, None, None]
    s = sigma[:, None, None]
    teacher_hi = (1 - s.square()).sqrt() * hi + s * teacher_noise
    fh, fl, bh, bl = model((1 - t) * hi + t * eps_hi, time, (1 - t) * lo + t * eps_lo,
                           time, labels, teacher_hi, sigma)
    def error(prediction, clean):
        return ((prediction.float() - clean) / t).square().mean()
    high = error(fh, hi) + .25 * error(bh, hi)
    low = error(fl, lo) + .25 * error(bl, lo)
    return (model.hi_channels * high + model.lo_channels * low) / z.shape[-1]


def update(model, ema, chart, support, cfg, rng, device, rank, world, coefficient):
    fresh, class_count, micro = (cfg[k] for k in ("samples_per_class", "classes_per_update", "micro_batch"))
    classes = torch.randperm(model.num_classes, generator=rng["classes"])[:class_count].reshape(world, -1)[rank]
    rows = np.concatenate([support.rows[c][torch.randperm(len(support.rows[c]), generator=rng["rows"])[:fresh].numpy()]
                           for c in classes.tolist()])
    z, labels = support.get(rows, device)
    def normal(shape, name):
        return torch.randn(shape, generator=rng[name]).to(device)
    noise = normal(z.shape, "noise")
    reference_noise = normal(z.shape, "ref")
    ref_t = cfg["reference_time"]
    t = torch.ones(len(z), device=device)
    scales = (cfg["train_hi_scale"], cfg["train_lo_scale"])
    def generate(net, x, time, sl):
        with torch.autocast(device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
            return chart(guided_pair(net, x, time, labels[sl], *scales))
    slices = [slice(i, i + micro) for i in range(0, len(z), micro)]
    fake, reference = [], []
    with torch.no_grad():
        for sl in slices:
            fake.append(generate(model, noise[sl], t[sl], sl))
            real_noisy = (1 - ref_t) * z[sl] + ref_t * reference_noise[sl]
            reference.append(generate(ema, real_noisy, t[sl] * ref_t, sl))
    shape = (len(classes), fresh, *z.shape[1:])
    current = torch.cat(fake).reshape(shape)
    grad, coefficient = distribution_gradient(current, torch.cat(reference).reshape(shape),
                                              model.hi_channels, cfg["kernel"], coefficient)
    del reference, fake
    for sl in slices:
        replay = generate(model, noise[sl], t[sl], sl)
        replay.backward(grad.flatten(0, 1)[sl])
    del current, grad, replay, noise, reference_noise
    fm_noise = normal(z.shape, "fm")
    times = (.05 + .45 * torch.rand(len(z), generator=rng["time"])).to(device)
    sigma = (.1 + .2 * torch.rand(len(z), generator=rng["cond"])).to(device)
    teacher_noise = normal(z[..., :model.hi_channels].shape, "cond")
    total = 0.
    for sl in slices:
        with torch.autocast(device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
            loss = fm_loss(model, z[sl], labels[sl], times[sl], fm_noise[sl], sigma[sl], teacher_noise[sl])
        (loss * (cfg["fm_weight"] / len(slices))).backward()
        total += loss.detach() / len(slices)
    return coefficient, total


def save(path, model, ema, optimizers, chart, rng, rank, world, step, config, coefficient):
    state = {k: gen.get_state() for k, gen in rng.items()}
    states = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(state, states, dst=0)
    else:
        states = [state]
    if rank == 0:
        atomic_save(dict(step=step, config=config, model=model.state_dict(), ema=ema.state_dict(),
                         optimizers=[o.state_dict() for o in optimizers], rng=states,
                         coefficient=coefficient, chart=chart.state_dict()), path)
    if world > 1:
        dist.barrier()


def train(args, rank, world, device):
    config = read_config(args.config)
    cfg = config["native_matching"]
    torch.manual_seed(42)
    model = RevFlow(**config["model"])
    start, coefficient = 0, None
    saved = load_weights(args.resume) if args.resume else None
    if saved is not None:
        if saved["config"] != config or len(saved["rng"]) != world:
            raise ValueError("Resume with the original NM configuration and process count")
        model.load_state_dict(saved["model"])
    else:
        model.load_state_dict(load_weights(args.parent))
    for layer in readouts(model).values():
        layer.requires_grad_(False)
    support = Support(args.cache, model, args.stats, args.cache_stats)
    count, fresh, micro = cfg["classes_per_update"], cfg["samples_per_class"], cfg["micro_batch"]
    if (count > model.num_classes or count % world or fresh < 2 or micro < 1
            or count * fresh // world % micro or min(map(len, support.rows)) < fresh):
        raise ValueError("Class count, fresh support, world size and microbatch are incompatible")
    chart = Chart(load_weights(args.chart)).to(device)
    model.to(device).train()
    ema = deepcopy(model).requires_grad_(False).eval()
    optimizers = optimizers_for(model, cfg)
    rng = generators(rank)
    if saved is not None:
        ema.load_state_dict(saved["ema"])
        for optimizer, state in zip(optimizers, saved["optimizers"]):
            optimizer.load_state_dict(state)
            for group, original in zip(optimizer.param_groups, state["param_groups"]):
                group.update({k: v for k, v in original.items() if k != "params"})
        for key, generator in rng.items():
            generator.set_state(saved["rng"][rank][key])
        chart.load_state_dict(saved["chart"])
        start, coefficient = saved["step"], saved["coefficient"]
        del saved
    latest = args.output / "latest.pt"
    if latest.exists() and args.resume is None:
        raise FileExistsError(f"Use --resume {latest} or choose a fresh output directory")
    stop = cfg["steps"] if args.stop_after is None else args.stop_after
    if not start <= stop <= cfg["steps"]:
        raise ValueError("Requested update lies outside the remaining adaptation budget")
    params = [p for p in model.parameters() if p.requires_grad]
    for step in range(start + 1, stop + 1):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        coefficient, loss = update(model, ema, chart, support, cfg, rng, device, rank, world, coefficient)
        average_gradients(params, world)
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        for optimizer in optimizers:
            optimizer.step()
        with torch.no_grad():
            for target, source in zip(ema.parameters(), model.parameters()):
                if source.requires_grad:
                    target.mul_(cfg["ema_decay"]).add_(source, alpha=1 - cfg["ema_decay"])
        if rank == 0 and (step == 1 or step % 10 == 0):
            print(f"update {step}  FM {float(loss):.5f}  IMQ coefficient {coefficient:.8f}", flush=True)
        if step % cfg["save_every"] == 0 or step == stop:
            save(latest, model, ema, optimizers, chart, rng, rank, world, step, config, coefficient)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "chart", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--parent", type=Path, help="Tensor-only 80-epoch DDT EMA")
    source.add_argument("--resume", type=Path, help="A checkpoint from this NM trainer")
    parser.add_argument("--cache-stats", type=Path, help="Original stats for a differently normalized cache")
    parser.add_argument("--stats", type=Path, help="Parent flow stats for cache conversion")
    parser.add_argument("--stop-after", type=int, help="Save and exit at this absolute update")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        train(args, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
