# Copyright (c) Meta Platforms, Inc. and affiliates.

from omegaconf import DictConfig

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.aggregation import MeanMetric

import adjoint_samplers.utils.train_utils as train_utils
from adjoint_samplers.components.matcher import Matcher, AdjointVEMatcher


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def _v3_inject_mode_samples(matcher, cfg, device, is_asbs_init_stage):
    """Inject synthetic samples from a specific mode into the matcher's buffer.

    For v3 Family B1 (Dead Mode Revival via Data Injection).
    Generates x1 from the target mode's Gaussian, runs them through
    the adjoint computation, and adds to the buffer.

    This creates "as if the sampler visited the dead mode" data,
    testing whether the collapsed controller can suppress this signal.
    """
    if not isinstance(matcher, AdjointVEMatcher):
        print("[v3] WARNING: injection only supported for AdjointVEMatcher, skipping")
        return

    center = torch.tensor(cfg.v3_injection_mode_center, device=device, dtype=torch.float32)
    sigma = float(cfg.v3_injection_mode_sigma)
    frac = float(cfg.v3_injection_fraction)

    # How many injection samples (fraction of resample_batch_size)
    n_inject = max(1, int(frac * cfg.resample_batch_size))

    # Generate x1 from the dead mode's Gaussian
    x1 = center.unsqueeze(0) + sigma * torch.randn(n_inject, center.shape[0], device=device)

    # Generate corresponding x0 from the source (needed for VE buffer format)
    # We use the ref SDE's sample_posterior logic: for VE with no drift,
    # x0 is just a noisy version. But the buffer only needs (x0, x1, adjoint1)
    # and x0 is sampled independently from source for the posterior bridge.
    # So we just sample x0 from the source distribution.
    source_module = matcher.sde.ref_sde  # We need the actual source, passed via cfg
    # Actually, for VE matcher, x0 is just the initial noise sample.
    # We generate it from N(0, scale^2 I) matching the source config.
    source_scale = float(cfg.scale) if "scale" in cfg else 2.0
    x0 = source_scale * torch.randn(n_inject, center.shape[0], device=device)

    # Compute adjoint1 = grad_term_cost(x1) at the injected terminal points
    adjoint1 = matcher._compute_adjoint1(x1, is_asbs_init_stage).clone()

    # Add to buffer (same format as AdjointVEMatcher.populate_buffer)
    matcher.buffer.add({
        "x0": x0.detach().cpu(),
        "x1": x1.detach().cpu(),
        "adjoint1": adjoint1.detach().cpu(),
    })

    print(f"  [v3] Injected {n_inject} samples from mode at "
          f"{cfg.v3_injection_mode_center} (frac={frac})")


def train_one_epoch(
    matcher: Matcher,
    model: torch.nn.Module,
    source: torch.nn.Module,
    optimizer: Optimizer,
    lr_schedule: LRScheduler | None,
    epoch: int,
    device: str,
    cfg: DictConfig,
):
    # build dataloader
    B = cfg.resample_batch_size
    M = matcher.resample_size // (B * cfg.world_size)
    loss_scale = matcher.loss_scale

    is_asbs_init_stage = train_utils.is_asbs_init_stage(epoch, cfg)

    for _ in range(M):
        x0 = source.sample([B,]).to(device)
        timesteps = train_utils.get_timesteps(**cfg.timesteps).to(device)
        matcher.populate_buffer(x0, timesteps, is_asbs_init_stage)

    # ── v3 hooks: dead mode data injection (Family B1) ──
    if (
        "v3_injection_start_epoch" in cfg
        and cfg.v3_injection_start_epoch is not None
        and cfg.v3_injection_start_epoch <= epoch
        < cfg.v3_injection_start_epoch + cfg.get("v3_injection_duration", 0)
    ):
        _v3_inject_mode_samples(matcher, cfg, device, is_asbs_init_stage)

    dataloader = matcher.build_dataloader(cfg.train_batch_size)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    loader = iter(cycle(dataloader))

    model.train(True)
    for _ in range(cfg.train_itr_per_epoch):
        optimizer.zero_grad()

        data = next(loader)

        input, target = matcher.prepare_target(data, device)
        output = model(*input)

        loss = loss_scale * ((output - target)**2).mean()
        loss.backward()

        if cfg.clip_grad_norm:
            max_norm = cfg.clip_target_norm if "clip_target_norm" in cfg and cfg.clip_target_norm is not None else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        optimizer.step()

        epoch_loss.update(loss.item())
        if lr_schedule:
            lr_schedule.step()

    return float(epoch_loss.compute().detach().cpu())
