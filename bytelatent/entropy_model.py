# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
import logging
import os

import torch

from bytelatent.transformer import LMTransformer, LMTransformerArgs

logger = logging.getLogger()


def load_entropy_model(
    entropy_model_checkpoint_dir, state_dict_path, device="cpu", dtype="bf16"
):
    with open(os.path.join(entropy_model_checkpoint_dir, "params.json")) as fr:
        reloaded = json.loads(fr.read())

    # torch.set_default_dtype(dtype)
    model_params = reloaded["entropy_model"]
    logger.warning(
        "Update checkpoint to load attn and sliding window args from checkpoint"
    )
    entropy_model_args = LMTransformerArgs(
        dim=model_params["dim"],
        n_layers=model_params["n_layers"],
        n_heads=model_params["n_heads"],
        max_seqlen=model_params["max_seqlen"],
        ffn_dim_multiplier=model_params["ffn_dim_multiplier"],
        vocab_size=model_params["vocab_size"],
        attn_bias_type="local_block_causal",
        attn_impl="xformers",
        sliding_window=512,
        init_device=device,
        init_dtype=dtype,
    )
    entropy_model = LMTransformer(entropy_model_args)

    entropy_model.load_state_dict(
        torch.load(state_dict_path, map_location=device)["model"], strict=False
    )
    entropy_model.to(device)
    entropy_model = entropy_model.eval()
    # no grads for the model:
    for n, param in entropy_model.named_parameters():
        param.requires_grad = False

    return entropy_model, entropy_model_args
