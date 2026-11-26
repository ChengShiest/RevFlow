"""Class-conditional distribution gradients for native matching.

Each tensor is [class, sample, position, channel]. Each visit uses 32 fresh
images per class in the paper configuration.
"""

import math

import torch
import torch.distributed as dist


def off_diagonal(matrix):
    n = matrix.shape[-1]
    return (matrix.sum((-2, -1)) - matrix.diagonal(dim1=-2, dim2=-1).sum(-1)) / (n * (n - 1))


def distances(x, y):
    squared = x.square().sum(-1)[:, :, None] + y.square().sum(-1)[:, None, :]
    return (squared - 2 * torch.bmm(x, y.transpose(1, 2))).clamp_min(0) / x.shape[-1]


def energy_distance(fake, target):
    x, y = fake.flatten(2), target.flatten(2)
    d = x.shape[-1]
    xx, yy, xy = (a @ b.transpose(-1, -2) / d for a, b in ((x, x), (y, y), (x, y)))
    nx, ny = xx.diagonal(dim1=-2, dim2=-1), yy.diagonal(dim1=-2, dim2=-1)

    def smooth(a, b, gram):
        return (.1 ** 2 + (a[:, :, None] + b[:, None, :] - 2 * gram).clamp_min(0)).sqrt()

    return (2 * smooth(nx, ny, xy).mean((-2, -1))
            - off_diagonal(smooth(nx, nx, xx)) - off_diagonal(smooth(ny, ny, yy))).mean()


def window_views(values):
    classes, samples, tokens, channels = values.shape
    side = math.isqrt(tokens)
    if side * side != tokens or side % 16:
        raise ValueError("Statistical drifting requires a square grid divisible by 16")
    grid = values.reshape(classes, samples, side, side, channels)
    for window in (2, 4, 16):
        windows = grid.reshape(classes, samples, side // window, window, side // window, window, channels)
        means = windows.mean((3, 5))
        variance = (windows - means[:, :, :, None, :, None, :]).square().mean((3, 5))
        yield means.flatten(2, 3)
        yield (variance + 1e-6).sqrt().flatten(2, 3)


@torch.no_grad()
def drift_target(fake, reference):
    classes, n, locations, width = fake.shape
    current = fake.transpose(1, 2).reshape(-1, n, width)
    positive = reference.transpose(1, 2).reshape(-1, reference.shape[1], width)
    targets = torch.cat((current, positive), 1)
    dot = torch.bmm(current, targets.transpose(1, 2))
    distance = (current.square().sum(-1)[:, :, None] + targets.square().sum(-1)[:, None, :]
                - 2 * dot).clamp_min(1e-8).sqrt()
    mean_distance = distance.mean()
    scale = (mean_distance / math.sqrt(width)).clamp_min(1e-3)
    current, targets = current / scale, targets / scale
    distance = distance / mean_distance.clamp_min(1e-3)
    mask = torch.zeros(n, targets.shape[1], device=fake.device)
    mask[:, :n] = 100 * torch.eye(n, device=fake.device)
    distance = distance + mask
    total = torch.zeros_like(current)
    for temperature in (.02, .05, .2):
        logits = -distance / temperature
        affinity = (logits.softmax(-1) * logits.softmax(-2)).clamp_min(1e-6).sqrt()
        neg, pos = affinity[:, :, :n], affinity[:, :, n:]
        coeff = torch.cat((-neg * pos.sum(-1, keepdim=True), pos * neg.sum(-1, keepdim=True)), dim=-1)
        force = torch.bmm(coeff, targets) - coeff.sum(-1, keepdim=True) * current
        total += force / force.square().mean().sqrt().clamp_min(1e-4)
    return (current + total).reshape(classes, locations, n, width).transpose(1, 2), scale


def drift_loss(fake, reference):
    terms = []
    for view, real in zip(window_views(fake), window_views(reference)):
        goal, scale = drift_target(view, real)
        terms.append((view / scale - goal).square().mean())
    return sum(terms) / len(terms)


def kernel_loss(fake, reference, hi_channels, kind):
    """Average the hi and lo full-image marginal IMQ estimates with equal weight."""
    if kind != "image_marginals":
        raise ValueError(f"Native matching requires the image_marginals kernel, got {kind!r}")
    branches = []
    for section in (slice(None, hi_channels), slice(hi_channels, None)):
        x, y = fake[..., section].flatten(2), reference[..., section].flatten(2)
        yy = distances(y, y)
        bandwidth = off_diagonal(yy).clamp_min(.01)[:, None, None]
        branches.append([v / bandwidth for v in (distances(x, x), yy, distances(x, y))])
    kernels = []
    for hi, lo in zip(*branches):
        terms = []
        for width in (.5, 1., 2.):
            high, low = (1 + hi / width).rsqrt(), (1 + lo / width).rsqrt()
            terms.append(.5 * (high + low))
        kernels.append(torch.stack(terms).mean(0))
    xx, yy, xy = kernels
    return (off_diagonal(xx) + off_diagonal(yy) - 2 * xy.mean((-2, -1))).mean()


def norm_ratio(a, b):
    squares = torch.stack((a.double().square().sum(), b.double().square().sum()))
    if dist.is_initialized():
        dist.all_reduce(squares)
    if not torch.isfinite(squares).all() or not (squares > 0).all():
        raise ValueError("Nonfinite or zero NM gradient during norm calibration")
    return (squares[0] / squares[1]).sqrt()


def distribution_gradient(current, reference, hi_channels, kind, coefficient=None):
    if current.ndim != 4 or current.shape != reference.shape or current.shape[1] < 2:
        raise ValueError("Expected matching class bags with at least two images per class")
    x = current.detach().float().requires_grad_(True)
    y = reference.detach().float()
    energy, = torch.autograd.grad(energy_distance(x, y), x)
    drift, = torch.autograd.grad(drift_loss(x, y), x)
    drift = drift * norm_ratio(energy, drift).float()
    auxiliary, = torch.autograd.grad(kernel_loss(x, y, hi_channels, kind), x)
    if coefficient is None:
        coefficient = .5 * norm_ratio(drift, auxiliary).item()
    return drift + coefficient * auxiliary, coefficient
