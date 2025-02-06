# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import logging
from collections import namedtuple
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

import fsspec
import torch
import torch.nn as nn
import wandb
from pydantic import BaseModel, ConfigDict

from bytelatent.distributed import get_is_master

logger = logging.getLogger()


class WandbArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_type: str | None = None
    dir: str | None = None
    project: str | None = None
    entity: str | None = None
    tags: list | None = None
    group: str | None = None
    name: str | None = None
    notes: str | None = None
    config_exclude_keys: list[str] | None = None
    config_include_keys: list[str] | None = None
    anonymous: str | None = None
    mode: str | None = None
    allow_val_change: bool | None = None
    resume: Union[bool, str] | None = None
    force: bool | None = None
    tensorboard: bool | None = None
    sync_tensorboard: bool | None = None
    monitor_gym: bool | None = None
    save_code: bool | None = None
    id: str | None = None
    fork_from: str | None = None
    resume_from: str | None = None


class LoggingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    freq: int = 10  # Log every freq optimizer steps
    acc_freq: int | None = None  # Log every acc_freq gradient accumulation steps

    wandb: WandbArgs | None = None


class MetricLogger:
    def __init__(
        self,
        outdir: Path,
        # args: TrainArgs
        args: Any | None = None,
        fs: fsspec.AbstractFileSystem | None = None,
    ):
        self.outdir = outdir
        self.jsonl_writer = None
        self.fs = fs
        self.args = args

    def open(self):
        if self.jsonl_writer is None:
            if self.fs is None:
                self.jsonl_writer = open(self.outdir, "a")
            else:
                self.jsonl_writer = self.fs.open(self.outdir, "a")
        if (
            self.args is not None
            and self.args.logging.wandb is not None
            and get_is_master()
        ):
            run = wandb.init(
                config=asdict(self.args),
                **asdict(self.args.logging.wandb),
            )

    def log(self, metrics: dict[str, Any]):
        if (
            self.args is not None
            and self.args.logging.wandb is not None
            and (wandb.run is not None)
        ):
            wandb.log(metrics, step=metrics["global_step"])

        metrics.update({"created_at": datetime.now(timezone.utc).isoformat()})
        print(json.dumps(metrics), file=self.jsonl_writer, flush=True)

    def close(self):
        if self.jsonl_writer is not None:
            self.jsonl_writer.close()
            self.jsonl_writer = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()


GPUMemStats = namedtuple(
    "GPUMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
        "power_draw",
    ],
)


class GPUMemoryMonitor:
    """
    Class to monitor GPU memory usage
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)  # device object
        self.device_name = torch.cuda.get_device_name(self.device)
        self.device_index = torch.cuda.current_device()
        self.device_capacity = torch.cuda.get_device_properties(
            self.device
        ).total_memory
        self.device_capacity_gib = self._to_gib(self.device_capacity)

        # reset stats, clear cache
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    def _to_gib(self, memory_in_bytes):
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib

    def _to_pct(self, memory):
        return 100 * memory / self.device_capacity

    def get_peak_stats(self):
        cuda_info = torch.cuda.memory_stats(self.device)

        max_active = cuda_info["active_bytes.all.peak"]
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = cuda_info["reserved_bytes.all.peak"]
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        num_retries = cuda_info["num_alloc_retries"]
        num_ooms = cuda_info["num_ooms"]
        power_draw = torch.cuda.power_draw()

        if num_retries > 0:
            logger.warning(f"{num_retries} CUDA memory allocation retries.")
        if num_ooms > 0:
            logger.warning(f"{num_ooms} CUDA OOM errors thrown.")

        return GPUMemStats(
            max_active_gib,
            max_active_pct,
            max_reserved_gib,
            max_reserved_pct,
            num_retries,
            num_ooms,
            power_draw,
        )

    def reset_peak_stats(self):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

    def __str__(self):
        mem_stats = self.get_peak_stats()
        display_str = f"{self.device_name} ({self.device_index}): {self.device_capacity_gib} GiB capacity, "
        display_str += (
            f"{mem_stats.max_reserved_gib} GiB peak, {mem_stats.max_reserved_pct}% peak"
        )
        return f"{display_str}"


def upload_train_to_wandb(
    ckpt_dir, project="lingua", entity="codegen-team", train=True, eval=True
):
    import json
    from pathlib import Path

    import wandb
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path(ckpt_dir) / "config.yaml")
    cfg = OmegaConf.to_container(cfg)

    if train:
        wandb.init(config=cfg, name=cfg["name"], project=project, entity=entity)

        with open(Path(ckpt_dir) / "metrics.jsonl") as f:
            for l in f:
                m = json.loads(l)
                wandb.log(m, step=m["global_step"])

        wandb.finish()

    if eval:
        wandb.init(config=cfg, name=cfg["name"], project=project, entity=entity)

        with open(Path(ckpt_dir) / "metrics.eval.jsonl") as f:
            for l in f:
                m = json.loads(l)
                wandb.log(
                    {
                        f"evals/{name.replace('/','.')}": value
                        for name, value in m.items()
                        if "/" in name
                    },
                    step=m["global_step"],
                )

        wandb.finish()


def get_num_params(model: nn.Module) -> int:
    """
    Get the total model params
    Args : only_trainable: whether to only count trainable params
    """
    numel = {n: p.numel() for n, p in model.named_parameters()}
    return sum(numel.values())
