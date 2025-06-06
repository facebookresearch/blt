# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from typing import Optional, Tuple, Union

import torch

from bytelatent.base_transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    cross_entropy,
)
from bytelatent.model.utils import check_param_device, create_causal_mask, DTYPE_MAP
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from xformers.ops import AttentionBias

logger = logging.getLogger()

try:
    from apex.normalization.fused_layer_norm import FusedRMSNorm

    RMSNorm = FusedRMSNorm
except (ImportError, ModuleNotFoundError):
    logging.debug("Apex not found. Using nn.RMSNorm")
    RMSNorm = nn.RMSNorm


def attention_flops_per_token(n_layers, seq_len, dim, causal):
    # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(
    num_non_embed_params: int, n_layers: int, dim: int, seq_len: int
) -> int:
    return 6 * num_non_embed_params + attention_flops_per_token(
        n_layers, seq_len, dim, True
    )


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


class LMTransformerArgs(BaseTransformerArgs):
    seed: int = 42

    vocab_size: int = -1
    weight_tying: bool = False

    sliding_window: int | None = None


class LMTransformer(
    BaseTransformer,
    PyTorchModelHubMixin,
    repo_url="https://github.com/facebookresearch/blt",
    paper_url="https://arxiv.org/abs/2412.09871",
    pipeline_tag="text-generation",
    license="other",
    license_name="fair-noncommercial-research-license",
    license_link="https://huggingface.co/facebook/blt/blob/main/LICENSE",
    coders={
        LMTransformerArgs: (
            lambda x: {"args": x.model_dump()},
            lambda data: LMTransformerArgs(**data),
        )
    },
):
    def __init__(self, args: LMTransformerArgs):
        super().__init__(args)
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window

        assert args.vocab_size > 0

        self.tok_embeddings = torch.nn.Embedding(
            args.vocab_size,
            args.dim,
            device=args.init_device,
            dtype=DTYPE_MAP[args.init_dtype],
        )

        self.norm = RMSNorm(
            args.dim,
            eps=args.norm_eps,
            device=args.init_device,
            dtype=DTYPE_MAP[args.init_dtype],
        )

        self.output = nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False,
            device=args.init_device,
            dtype=DTYPE_MAP[args.init_dtype],
        )

        if args.weight_tying:
            self.output.weight = self.embeddings.tok_embeddings.weight

        # Sanity check
        check_param_device(self, args.init_device)

    def push_to_hub(self, *args, **kwargs):
        raise ValueError(
            "For meta authors: Do not push BLT weights with this, save weights with save_pretrained() then push them manually to HF hub to ensure the repository metadata is correct."
        )

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str | None = None,
    ):
        if attn_impl is None:
            attn_impl = self.attn_impl
        bsz, seqlen = token_values.shape

        h = self.tok_embeddings(token_values)

        mask = (
            mask
            if mask is not None
            else create_causal_mask(
                seqlen,
                attn_impl,
                self.attn_bias_type,
                sliding_window=self.sliding_window,
                tokens=token_values,
                eos_id=self.eos_id,
            )
        )
        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)

        logits = self.output(self.norm(h))
        if target is not None:
            return cross_entropy(logits, target)
        else:
            return logits

    def reset_parameters(self, init_std=None):
        self.norm.reset_parameters()

    def init_weights(self):
        self.reset_parameters()
        init_std = self.dim ** (-0.5)
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        super().init_weights()

        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )


# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


# Optional and only used for fully shard options (fsdp) is choose. Highly recommanded for large models
def build_fsdp_grouping_plan(model_args: LMTransformerArgs):
    group_plan: Tuple[int, bool] = []

    if isinstance(model_args, LMTransformerArgs):
        group_plan.append(("tok_embeddings", False))

        for i in range(model_args.n_layers):
            group_plan.append((f"layers.{i}", False))

        group_plan.append(("output", True))
    else:
        for i in range(model_args.n_layers_local_encoder):
            group_plan.append((f"local_encoder.layers.{i}", False))
            group_plan.append((f"local_encoder.cross_attn_layers.{i}", False))
        for i in range(model_args.n_layers_local_decoder):
            group_plan.append((f"local_decoder.layers.{i}", False))
            group_plan.append((f"local_decoder.cross_attn_layers.{i}", False))
        for i in range(model_args.n_layers_global):
            group_plan.append((f"global_transformer.layers.{i}", False))

        for i in range(len(model_args.encoder_hash_byte_group_size)):
            group_plan.append((f"encoder_hash_tok_embedding.{i}", False))

    return group_plan


# Optional and only used for model/tensor parallelism when tp_size > 1
def tp_parallelize(model, tp_mesh, model_args: LMTransformerArgs, distributed_args):
    assert model_args.dim % distributed_args.tp_size == 0
    assert model_args.vocab_size % distributed_args.tp_size == 0
    assert model_args.n_heads % distributed_args.tp_size == 0
    assert (model_args.n_kv_heads or 0) % distributed_args.tp_size == 0
    assert model_args.n_heads % (model_args.n_kv_heads or 1) == 0

    # Embedding layer tp
    main_plan = {}
    main_plan["tok_embeddings"] = ColwiseParallel(
        input_layouts=Replicate(), output_layouts=Shard(1)
    )
    main_plan["norm"] = SequenceParallel()
    main_plan["output"] = ColwiseParallel(
        input_layouts=Shard(1), output_layouts=Replicate()
    )

    parallelize_module(
        model,
        tp_mesh,
        main_plan,
    )

    # Attention layers tp
    for layer in model.layers:
        layer_plan = {}

        layer_plan["attention"] = PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        )
        layer_plan["attention_norm"] = SequenceParallel()
        layer_plan["attention.wq"] = ColwiseParallel()
        layer_plan["attention.wk"] = ColwiseParallel()
        layer_plan["attention.wv"] = ColwiseParallel()
        layer_plan["attention.wo"] = RowwiseParallel(output_layouts=Shard(1))

        # Feedforward layers tp
        layer_plan["feed_forward"] = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        )
        layer_plan["ffn_norm"] = SequenceParallel()
        layer_plan["feed_forward.w1"] = ColwiseParallel()
        layer_plan["feed_forward.w3"] = ColwiseParallel()
        layer_plan["feed_forward.w2"] = RowwiseParallel(output_layouts=Shard(1))

        parallelize_module(
            layer,
            tp_mesh,
            layer_plan,
        )

        # Adjusting the number of heads and kv heads according to the tp size
        attn_layer = layer.attention
        attn_layer.n_heads = attn_layer.n_heads // distributed_args.tp_size
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // distributed_args.tp_size
