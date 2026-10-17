"""DDP support: communication timing, hang-safe launch settings, and a numerical-equivalence check.

Nothing here imports torch at module import time, so the dashboard server can load the package
without it.
"""

from __future__ import annotations

import datetime
import os
from typing import Any

from .tracker import TrainingTracker


def launch_env(timeout_s: int = 600, run_id: str | None = None) -> dict[str, str]:
    """Environment that makes a stuck collective fail loudly instead of hanging forever. Merge into
    the torchrun environment; also pass `process_group_timeout(timeout_s)` to init_process_group."""
    env = {
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",   # RCCL honours the NCCL-named torch settings
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": str(timeout_s),
        "TORCH_DISTRIBUTED_DEBUG": "OFF",
    }
    if run_id:
        env["DGPU_RUN_ID"] = run_id
    return env


def process_group_timeout(timeout_s: int = 600) -> datetime.timedelta:
    return datetime.timedelta(seconds=timeout_s)


def install_comm_timing(ddp_model: Any, tracker: TrainingTracker) -> bool:
    """Time every gradient all-reduce with CUDA events (read once per window, never synchronised
    per call). Returns False when the model is not a DDP instance or hooks are unavailable."""
    try:
        import torch
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
    except Exception:  # noqa: BLE001
        return False
    if not hasattr(ddp_model, "register_comm_hook") or not torch.cuda.is_available():
        return False

    def hook(state: Any, bucket: Any):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        future = default_hooks.allreduce_hook(state, bucket)

        def done(fut: Any):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            tracker.comm_events.append((start, end))
            return fut.value()[0]

        return future.then(done)

    try:
        ddp_model.register_comm_hook(None, hook)
    except RuntimeError:  # a hook is already registered
        return False
    return True


def _worker(rank: int, world_size: int, config: dict[str, Any], queue: Any) -> None:  # pragma: no cover - spawned
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(config["port"]))
    device = torch.device("cpu") if config["device"] == "cpu" else torch.device(config["device"], rank)
    dist.init_process_group(config["backend"], rank=rank, world_size=world_size)

    def make_model() -> Any:
        torch.manual_seed(1234)
        return torch.nn.Sequential(torch.nn.Embedding(32, 16), torch.nn.Linear(16, 16), torch.nn.Tanh(), torch.nn.Linear(16, 32))

    def global_batches() -> list[list[tuple[Any, Any]]]:
        """steps x accumulation x world_size (tokens, labels); ragged lengths so token-count
        normalisation matters (labels -100 mark padding)."""
        gen = torch.Generator().manual_seed(7)
        out = []
        for _ in range(config["steps"]):
            step = []
            for _ in range(config["accumulation"] * world_size):
                length = int(torch.randint(3, 12, (1,), generator=gen))
                tokens = torch.randint(0, 32, (2, length), generator=gen)
                labels = tokens.clone()
                labels[:, : int(torch.randint(0, 3, (1,), generator=gen))] = -100
                step.append((tokens, labels))
            out.append(step)
        return out

    batches = global_batches()
    model = make_model().to(device)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=None if device.type == "cpu" else [rank])
    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.1)
    losses = []
    for step in batches:
        optimizer.zero_grad()
        mine = [step[i * world_size + rank] for i in range(config["accumulation"])]
        local_tokens = sum(int((labels[:, 1:] != -100).sum()) for _, labels in mine)
        total = torch.tensor([local_tokens], dtype=torch.float64)
        dist.all_reduce(total)
        for tokens, labels in mine:
            tokens, labels = tokens.to(device), labels.to(device)
            logits = ddp(tokens)[:, :-1]
            token_loss = F.cross_entropy(logits.reshape(-1, 32), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
            # DDP averages gradients over ranks, so scale by world_size to get the token-weighted global mean.
            norm = float(total) / world_size if config["mode"] == "token_normalised" else None
            loss = token_loss / norm if norm else token_loss / max(1, int((labels[:, 1:] != -100).sum()))
            loss.backward()
        optimizer.step()
        losses.append(float(token_loss.detach()))
    if rank == 0:
        queue.put({"params": [p.detach().cpu().flatten() for p in model.parameters()]})
    dist.destroy_process_group()


def _reference(config: dict[str, Any], world_size: int) -> list[Any]:  # pragma: no cover - needs torch
    import torch
    import torch.nn.functional as F

    torch.manual_seed(1234)
    model = torch.nn.Sequential(torch.nn.Embedding(32, 16), torch.nn.Linear(16, 16), torch.nn.Tanh(), torch.nn.Linear(16, 32))
    gen = torch.Generator().manual_seed(7)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(config["steps"]):
        batch = []
        for _ in range(config["accumulation"] * world_size):
            length = int(torch.randint(3, 12, (1,), generator=gen))
            tokens = torch.randint(0, 32, (2, length), generator=gen)
            labels = tokens.clone()
            labels[:, : int(torch.randint(0, 3, (1,), generator=gen))] = -100
            batch.append((tokens, labels))
        total = sum(int((labels[:, 1:] != -100).sum()) for _, labels in batch)
        optimizer.zero_grad()
        for tokens, labels in batch:
            logits = model(tokens)[:, :-1]
            loss = F.cross_entropy(logits.reshape(-1, 32), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum") / total
            loss.backward()
        optimizer.step()
    return [p.detach().flatten() for p in model.parameters()]


def run_equivalence_check(world_size: int = 2, steps: int = 3, accumulation: int = 2, device: str = "cpu",
                          backend: str = "gloo", mode: str = "token_normalised", tolerance: float = 1e-5,
                          port: int = 29511) -> dict[str, Any]:
    """One DDP update path versus a single-process accumulated reference over the SAME global
    batches. Use mode='naive_mean' to see what a per-rank mean loss does with ragged batches.
    On the training machine run with device='cuda', backend='nccl' (RCCL) for the real check."""
    import torch
    import torch.multiprocessing as mp

    config = {"steps": steps, "accumulation": accumulation, "device": device, "backend": backend, "mode": mode, "port": port}
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    mp.spawn(_worker, args=(world_size, config, queue), nprocs=world_size, join=True)
    ddp_params = queue.get()["params"]
    reference = _reference(config, world_size)
    worst = max(float((a - b).abs().max()) for a, b in zip(ddp_params, reference))
    return {"max_abs_param_diff": worst, "tolerance": tolerance, "equivalent": worst <= tolerance, "mode": mode,
            "world_size": world_size, "device": device, "torch": torch.__version__}
