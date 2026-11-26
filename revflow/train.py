"""Train a class-conditional RevFlow generator on ImageNet."""

import argparse
from copy import deepcopy
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from revflow.checkpoints import DEFAULT_CONFIG, atomic_save, load_weights, read_config
from revflow.data import ImageNet
from revflow.models import RevFlow
from revflow.optim import build_optimizers, lr_multiplier
from revflow.codec import FeatureEncoder, FlowStats
from revflow.flow import training_loss


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for name in ("data", "dino-repo", "dino-weights", "codec", "stats", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", type=Path, help="Resume a checkpoint written by this trainer")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_training(path, model, ema, optimizers, step, steps_per_epoch, config, rank, world, device):
    rng = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        rng["device"] = torch.cuda.get_rng_state(device)
    states = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(rng, states, dst=0)
    else:
        states = [rng]
    if rank == 0:
        atomic_save({"model": model.state_dict(), "ema": ema.state_dict(), "config": config,
                     "optimizers": [opt.state_dict() for opt in optimizers], "step": step,
                     "steps_per_epoch": steps_per_epoch, "rng": states}, path)
    if world > 1:
        dist.barrier()


def train(args, rank, world, device):
    config = read_config(args.config)
    recipe = config["training"]
    torch.manual_seed(recipe["seed"])
    model = RevFlow(**config["model"]).to(device)
    ema = deepcopy(model).requires_grad_(False).eval() if rank == 0 else None
    if world > 1:
        net = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                     broadcast_buffers=False)
        if device.type == "cuda":
            from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook
            net.register_comm_hook(None, bf16_compress_hook)
    else:
        net = model
    optimizers = build_optimizers(model, recipe)
    codec_weights = load_weights(args.codec)
    encoder = FeatureEncoder(args.dino_repo, args.dino_weights, codec_weights).to(device)
    del codec_weights
    stats = FlowStats(load_weights(args.stats)).to(device)
    dataset = ImageNet(args.data, augment=True)
    if len(dataset.classes) > model.num_classes:
        raise ValueError("Dataset has more classes than the model")
    if recipe["global_batch"] < world or recipe["global_batch"] % world:
        raise ValueError("Global batch must be a positive multiple of the process count")
    sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=recipe["seed"])
    loader_rng = torch.Generator()
    loader = DataLoader(dataset, batch_size=recipe["global_batch"] // world, sampler=sampler,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        generator=loader_rng, drop_last=True)
    if len(loader) == 0:
        raise ValueError("The dataset must contain at least one global batch")
    checkpoint = args.output / "latest.pt"
    if checkpoint.exists() and args.resume is None:
        raise FileExistsError(f"Use --resume {checkpoint} or choose another output directory")
    step = 0
    torch.manual_seed(recipe["seed"] + rank)
    if args.resume:
        saved = load_weights(args.resume)
        if (config != saved["config"] or len(saved["rng"]) != world
                or len(loader) != saved["steps_per_epoch"]):
            raise ValueError("Resume requires matching configuration, world size and epoch length")
        model.load_state_dict(saved["model"])
        if ema is not None:
            ema.load_state_dict(saved["ema"])
        for optimizer, state in zip(optimizers, saved["optimizers"]):
            optimizer.load_state_dict(state)
        rng = saved["rng"][rank]
        torch.set_rng_state(rng["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng["device"], device)
        step = saved["step"]
        del saved

    start_epoch, cursor = divmod(step, len(loader))
    for epoch in range(start_epoch, recipe["epochs"]):
        sampler.set_epoch(epoch)
        dataset.set_epoch(recipe["seed"], epoch)
        loader_rng.manual_seed(recipe["seed"] + epoch * world + rank)
        for index, (images, labels) in enumerate(loader):
            if epoch == start_epoch and index < cursor:
                continue
            scale = lr_multiplier(step, len(loader), recipe)
            for optimizer, lr in zip(optimizers, (recipe["muon_lr"], recipe["adamw_lr"])):
                optimizer.param_groups[0]["lr"] = lr * scale
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                with torch.no_grad():
                    hi, lo = stats.normalize(*encoder(images.to(device, non_blocking=True)))
                loss = training_loss(net, hi, lo, labels.to(device), config["flow"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["clip_grad"], error_if_nonfinite=True)
            for optimizer in optimizers:
                optimizer.step()
            if ema is not None:
                with torch.no_grad():
                    for target, source in zip(ema.parameters(), model.parameters()):
                        if source.requires_grad:
                            target.mul_(recipe["ema_decay"]).add_(source, alpha=1 - recipe["ema_decay"])
            step += 1
            if rank == 0 and (step == 1 or step % 100 == 0):
                print(f"epoch {epoch + 1}  step {step}  loss {loss.item():.5f}", flush=True)
            if step % recipe["checkpoint_interval"] == 0 or index + 1 == len(loader):
                save_training(checkpoint, model, ema, optimizers, step, len(loader), config, rank, world, device)


def main():
    args = arguments()
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        train(args, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
