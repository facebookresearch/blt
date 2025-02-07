# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from datetime import timedelta
from enum import Enum
from functools import lru_cache
import logging
import math
import sys
import time
from typing import Dict, List, Optional, Tuple
from torch import Tensor
import os
import pickle

import fsspec
import torch
import torch.distributed
import torch.nn.functional
import torch.nn.functional as F
from torch.distributed._tensor import DTensor

from torch.distributed.device_mesh import init_device_mesh
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

from bytelatent.args import TrainArgs
from bytelatent.distributed import (
    DistributedArgs,
    check_model_value_range,
    parallelize_model,
    setup_env,
    setup_torch_distributed,
)

logger = logging.getLogger()


def set_root_log_level(log_level: str):
    logger = logging.getLogger()
    level: int | str = log_level.upper()
    try:
        level = int(log_level)
    except ValueError:
        pass
    try:
        logger.setLevel(level)  # type: ignore
    except Exception:
        logger.warning(
            f"Failed to set logging level to {log_level}, using default 'NOTSET'"
        )
        logger.setLevel(logging.NOTSET)


class LogFormatter(logging.Formatter):
    """
    Custom logger for distributed jobs, displaying rank
    and preserving indent from the custom prefix format.
    """

    def __init__(self):
        self.start_time = time.time()
        self.rank = get_global_rank()
        self.show_rank = not get_is_slurm_job()  # srun has --label

    def formatTime(self, record):
        subsecond, seconds = math.modf(record.created)
        curr_date = (
            time.strftime("%y-%m-%d %H:%M:%S", time.localtime(seconds))
            + f".{int(subsecond * 1_000_000):06d}"
        )
        delta = timedelta(seconds=round(record.created - self.start_time))
        return f"{curr_date} - {delta}"

    def formatPrefix(self, record):
        fmt_time = self.formatTime(record)
        if self.show_rank:
            return f"{self.rank}: {record.levelname:<7} {fmt_time} - "
        else:
            return f"{record.levelname:<7} {fmt_time} - "

    def formatMessage(self, record, indent: str):
        content = record.getMessage()
        content = content.replace("\n", "\n" + indent)
        # Exception handling as in the default formatter, albeit with indenting
        # according to our custom prefix
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if content[-1:] != "\n":
                content = content + "\n" + indent
            content = content + indent.join(
                [l + "\n" for l in record.exc_text.splitlines()]
            )
            if content[-1:] == "\n":
                content = content[:-1]
        if record.stack_info:
            if content[-1:] != "\n":
                content = content + "\n" + indent
            stack_text = self.formatStack(record.stack_info)
            content = content + indent.join([l + "\n" for l in stack_text.splitlines()])
            if content[-1:] == "\n":
                content = content[:-1]

        return content

    def format(self, record):
        prefix = self.formatPrefix(record)
        indent = " " * len(prefix)
        content = self.formatMessage(record, indent)
        return prefix + content


def init_logger(
    log_file: str | None = None,
    *,
    name: str | None = None,
    level: str = "INFO",
    fs: fsspec.AbstractFileSystem | None = None,
):
    """
    Setup logging.

    Args:
        log_file: A file name to save file logs to.
        name: The name of the logger to configure, by default the root logger.
        level: The logging level to use.
    """
    set_root_log_level(level)
    logger = logging.getLogger(name)

    # stdout: everything
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.NOTSET)
    stdout_handler.setFormatter(LogFormatter())

    # stderr: warnings / errors and above
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(LogFormatter())

    # set stream handlers
    logger.handlers.clear()
    logger.handlers.append(stdout_handler)
    logger.handlers.append(stderr_handler)


@torch.no_grad()
def fixed_clip_grad_norm_(
    parameters: torch.Tensor | list[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    r"""Clip the gradient norm of an iterable of parameters.

    The norm is computed over the norms of the individual gradients of all parameters,
    as if the norms of the individual gradients were concatenated into a single vector.
    Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad.to(torch.bfloat16) for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)
    first_device = grads[0].device
    grouped_grads: Dict[
        Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]
    ] = _group_tensors_by_device_and_dtype(
        [grads]
    )  # type: ignore[assignment]

    norms: List[Tensor] = []
    for (device, _), ([device_grads], _) in grouped_grads.items():  # type: ignore[assignment]
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_grads, norm_type))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_grads])

    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(first_device) for norm in norms]), norm_type
    )

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for (device, _), ([device_grads], _) in grouped_grads.items():  # type: ignore[assignment]
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device)

    return total_norm


def get_no_recompute_ops():
    return None


@lru_cache()
def get_is_torch_run() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


@lru_cache()
def get_is_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ and not get_is_torch_run()


@lru_cache()
def get_global_rank() -> int:
    if get_is_torch_run():
        return int(os.environ["RANK"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_PROCID"])
    else:
        return 0


@lru_cache()
def get_local_rank() -> int:
    if get_is_torch_run():
        return int(os.environ["LOCAL_RANK"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_LOCALID"])
    else:
        return 0


@lru_cache()
def get_world_size() -> int:
    if get_is_torch_run():
        return int(os.environ["WORLD_SIZE"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_NTASKS"])
    else:
        return 1


@lru_cache()
def get_is_master() -> bool:
    return get_global_rank() == 0


def validate_train_args(args: TrainArgs, output_size: int):
    # assert args.model is not None or args.entropy_model is not None
    if args.entropy_model is not None:
        logger.info(f"Setting model output size to {args.entropy_model.vocab_size}")
        args.entropy_model.vocab_size = output_size

    assert args.dump_dir, "Dump dir not set"

    if (
        args.distributed.dp_replicate
        * args.distributed.dp_shard
        * args.distributed.tp_size
        != get_world_size()
    ):
        logging.info("Modifying TrainArgs distributed config")
        assert get_world_size() % args.distributed.dp_shard == 0
        logging.info("World size: %s", get_world_size())
        logging.info(
            "Existing setting: train_args.distributed.dp_shard=%s",
            args.distributed.dp_shard,
        )
        logging.info(
            "Setting train_args.distributed.dp_replicate=%s, was dp_replicate=%s",
            get_world_size() // args.distributed.dp_shard,
            args.distributed.dp_replicate,
        )
        args.distributed.dp_replicate = get_world_size() // args.distributed.dp_shard

        logging.info(
            "Changing dp_replicate from %s to %s, to account for tp_size=%s",
            args.distributed.dp_replicate,
            args.distributed.dp_replicate // args.distributed.tp_size,
            args.distributed.tp_size,
        )
        assert args.distributed.dp_replicate % args.distributed.tp_size == 0
        args.distributed.dp_replicate = (
            args.distributed.dp_replicate // args.distributed.tp_size
        )

        logger.warning(
            f"Setting Data Parallel size to {args.distributed.dp_replicate * args.distributed.dp_shard}"
        )
        assert (
            args.distributed.dp_replicate
            * args.distributed.dp_shard
            * args.distributed.tp_size
            == get_world_size()
        )

        if args.distributed.fsdp_type == "no_shard":
            assert (
                args.distributed.dp_shard == 1
                and args.distributed.dp_replicate == get_world_size()
            )

    if args.model is not None:
        args.model.max_seqlen = args.data.seq_len
    if args.entropy_model is not None:
        args.entropy_model.max_seqlen = args.data.seq_len

    if args.distributed.tp_size == 1:
        logger.warning(
            "Tensor parallelism has not been tested for a while, use at your own risk"
        )

    assert (
        args.probe_freq != args.profiling.mem_steps
    ), "Don't profile during probe step"
    assert (
        args.probe_freq != args.profiling.profile_steps
    ), "Don't profile during probe step"
    if args.logging.wandb is not None:
        args.logging.wandb.name = args.name

    if args.probe_freq is not None:
        assert (
            args.distributed.tp_size == 1
        ), "Probing not supported with tensor parallelism"
        assert (
            args.distributed.selective_activation_checkpointing is False
        ), "Probing not supported with selective activation checkpointing"


def compute_loss(p, y, mask, scale):
    tok_loss = scale * F.cross_entropy(
        p.flatten(0, 1), y.flatten(0, 1), reduction="none"
    )
    if mask is None:
        loss = tok_loss.mean()
    else:
        mask = mask.flatten(0, 1)
        tok_loss = tok_loss * mask
        loss = tok_loss.sum() / (mask.sum() + 1e-6)
    return loss, tok_loss


def get_device_mesh(distributed_args):
    tp_size = distributed_args.tp_size
    dp_replicate = distributed_args.dp_replicate
    dp_shard = distributed_args.dp_shard

    assert (
        dp_replicate * dp_shard * tp_size == get_world_size()
    ), f"dp_replicate * dp_shard * tp_size ({dp_replicate} * {dp_shard} * {tp_size}) != world_size ({get_world_size()})"

    dims = []
    names = []
    if dp_replicate >= 1:
        dims.append(dp_replicate)
        names.append("dp_replicate")
    if dp_shard > 1 or distributed_args.fsdp_type == "no_shard":
        dims.append(dp_shard)
        names.append("dp_shard")
    if tp_size > 1:
        dims.append(tp_size)
        names.append("tp")
    dims = tuple(dims)
    names = tuple(names)

    return init_device_mesh("cuda", mesh_shape=dims, mesh_dim_names=names)


def build_fsdp_grouping_plan():
    group_plan: tuple[int, bool] = []

    # Grouping and output seperately
    group_plan.append(("tok_embeddings", False))

    # Grouping by layers
    # for i in range(model_args.n_layers):
    #    group_plan.append((f"layers.{i}", False))

    group_plan.append(("output", True))

    return group_plan


class MinimalModel(torch.nn.Module):
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.tok_embeddings = torch.nn.Embedding(vocab_size, dim)

        # self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # self.layers = torch.nn.ModuleList()
        # for _ in range(args.n_layers):
        #    self.layers.append(TransformerBlock(args))

        self.output = torch.nn.Linear(
            dim,
            vocab_size,
            bias=False,
        )

    def forward(self, tokens):
        h = self.tok_embeddings(tokens)
        logits = self.output(h)
        # logits = self.output(self.norm(h))
        return logits

    def reset_parameters(self, init_std=None):
        pass

    def init_weights(self):
        pass


def train():
    args = TrainArgs(
        dump_dir="/tmp",
        name="debug_bf16",
        model=None,
        entropy_model=None,
        distributed=DistributedArgs(
            fsdp_type="full_shard",
            model_dtype="bf16",
            matmul_allow_tf32=False,
            selective_activation_checkpointing=False,
            tp_size=1,
        ),
    )
    tokenizer = args.data.tokenizer_args.build()
    validate_train_args(
        args,
        tokenizer.n_words,
    )
    dump_fs = fsspec.filesystem("file")
    init_logger(os.path.join(args.dump_dir, "train.log"), fs=dump_fs)
    setup_env(args.env)
    setup_torch_distributed(args.distributed)
    world_mesh = get_device_mesh(args.distributed)
    logger.info(f"Starting job: {args.name}")

    # build dataloader
    # need dp world size and rank
    dp_mesh = world_mesh["dp_replicate"]
    dp_degree = dp_mesh.size()
    dp_rank = dp_mesh.get_local_rank()
    if args.distributed.dp_shard > 1:
        dp_rank = dp_rank * dp_degree + world_mesh["dp_shard"].get_local_rank()
        dp_degree *= world_mesh["dp_shard"].size()

    logger.info(f"Running on dp rank : {dp_rank}")
    logger.info(f"Running on dp size : {dp_degree}")

    torch.manual_seed(args.seed)
    logger.info("Building model")

    # Initializing Model in meta device allows us to initialize models much bigger than 1 gpu's memory
    with torch.device("meta"):
        model = MinimalModel(768, tokenizer.n_words)

    model = parallelize_model(
        model,
        world_mesh,
        args.model,
        args.distributed,
        fsdp_grouping_plan=build_fsdp_grouping_plan(),
        tp_parallelize=None,
        no_recompute_ops=get_no_recompute_ops(),
    )

    # Once we shard the model on different gpus we can actually initialize the model
    # First we create empty tensors of the correct shapes
    model = model.to_empty(device="cuda")
    # Then we init the model. Please make sure this function initializes *ALL* parameters
    # and buffers, otherwise you will have random values in the unitialized tensors
    # which will silently fail (give nan gradients for example)

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(42)
        model.init_weights()
    check_model_value_range(model, range=10.0, std=1.0)

    # data_loader = args.data.build_from_rank(dp_rank, dp_degree)

    # train loop
    model.train()
    # data_loader = train_state.data_loader_state.build()
    # batch_iterator = data_loader.create_iter()
    # batch = next(batch_iterator)
    # with open(f"/storage/home/par/toy-data/batch_{dp_rank}.pickle", "wb") as f:
    #     pickle.dump(batch, f)
    with open(f"/storage/home/par/toy-data/batch_{dp_rank}.pickle", "rb") as f:
        batch = pickle.load(f)

    batch_x = torch.from_numpy(
        batch.x,
    ).cuda()
    batch_y = torch.from_numpy(batch.y).cuda()
    mask = None if batch.mask is None else torch.from_numpy(batch.mask).cuda()
    pred = model(batch_x)
    loss, _ = compute_loss(pred, batch_y, mask, 1.0)

    # We scale loss with grad_acc_steps so the gradient is the same
    # regardless of grad_acc_steps
    loss = loss / args.grad_acc_steps

    # backward on scaled loss to create scaled gradients
    loss.backward()
    # For logging we undo that scaling
    loss = loss.detach() * args.grad_acc_steps

    world_size = get_world_size()
    if 1 < world_size <= 8 and False:
        # For some reason, there are errors in reduces due to
        # not working for non-bf16 numbers. This function is a patched
        # version that converts gradients to bf16 before computing norms.
        # The error only happens in distributed training on one node,
        # hence the guard
        grad_norm = fixed_clip_grad_norm_(
            model.parameters(), max_norm=args.optim.clip, foreach=True
        )
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=args.optim.clip, foreach=True
        )

    grad_norm = (
        grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
    ).item()

    # if isinstance(data_loader, MultiprocessIterator):
    #    logger.info("Closing MP iterator before exiting")
    #    data_loader.shutdown()


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgsgs

    @dataclass
    class LMTransformerArgsgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate TrainArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call train.py with train.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in TrainArgs dataclass.
    """
    train()


if __name__ == "__main__":
    main()
