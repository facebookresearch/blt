# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
import os

import torch
from torch.nn.attention.flex_attention import create_block_mask
from xformers.ops import fmha

logger = logging.getLogger()


def patch_reduce(h, max_num_patches, reduction, patch_ids):
    """
    Reduce variable length patches to single embedding per patch
    Note: this works with variable number of patches for different sequences in the batch
    It handles variable length patches by assuming that patch_lengths will be 0 for any
    extra patches on the *right*. Since there can be a variable number of patches
    this function also return the number of patches for each sequence in the batch.
    Any embeddings on the right that are not allocated to a patch
    (i.e. if the sum(patch_lengths[i]) < seq_len for any i)
    will be sent to a dummy patch, which is trimmed before returning.
    """
    bs, seq_len, emb_dim = h.shape

    patch_ids = patch_ids.unsqueeze(-1).expand(-1, -1, h.shape[-1])

    reduced_embs = torch.zeros(
        (bs, max_num_patches, emb_dim), dtype=h.dtype, device=h.device
    )
    reduced_embs = reduced_embs.scatter_reduce(
        src=h,
        dim=1,
        index=patch_ids,
        reduce=reduction,
        include_self=False,
    )
    reduced_embs = reduced_embs[:, :max_num_patches, :]

    return reduced_embs


def concat_downsample(h, patch_lengths, patch_size):
    # The assumption in this function is that seq_len = patch_size * num_patches.
    bs, seq_len, emb_dim = h.shape
    patch_end_ids = torch.cumsum(patch_lengths, dim=1)
    patch_ids = patch_end_ids.unsqueeze(-1) - torch.arange(patch_size, 0, -1).to(
        patch_end_ids.device
    )
    # Is clamp ok here?
    patch_ids = patch_ids.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, h.shape[-1])
    patch_ids = patch_ids.view(bs, -1, emb_dim)
    # after gather h.shape = [batch_size, seq_len, dim]
    h = torch.gather(h, 1, patch_ids)
    h = h.reshape(bs, patch_lengths.shape[1], patch_size * h.size(-1))
    return h


def pooling_downsample(h, max_num_patches, pooling_mode, patch_ids):
    cat = []
    if "avg" in pooling_mode or "mean" in pooling_mode:
        cat.append(patch_reduce(h, max_num_patches, "mean", patch_ids))
    if "min" in pooling_mode:
        cat.append(patch_reduce(h, max_num_patches, "amin", patch_ids))
    if "max" in pooling_mode:
        cat.append(patch_reduce(h, max_num_patches, "amax", patch_ids))
    assert len(cat) > 0
    h = torch.cat(cat, dim=-1)
    return h


def downsample(
    h,
    num_patches,
    patch_lengths=None,
    patch_ids=None,
    downsampling_by_pooling=None,
    patch_size=4,
):
    """
    Downsampling:
        a. concatenating embeddings in the patch
            Note: with dynamic patching, patch the last patch_size tokens.
        b. pooling embeddings in the patch
    """
    # input: h.shape = [batch_size, seq_len, dim]
    # input: pool h.shape = [batch_size, seq_len / patch_size, dim]
    # if we don't use the cros_attn, we pool so that we convert bytes rep to patch rep
    if downsampling_by_pooling is not None and len(downsampling_by_pooling) > 0:
        # By pooling
        max_num_patches = num_patches
        assert patch_ids is not None
        h = pooling_downsample(h, max_num_patches, downsampling_by_pooling, patch_ids)
    else:
        # TODO: remove this condition
        # By concatenating (fixed lengths patching)
        assert patch_lengths is not None
        h = concat_downsample(h, patch_lengths, patch_size)
    return h


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def tokens_to_seqlen(batch: torch.Tensor, eos_id: int):
    """
    0 0 0 1 0 0 0 1 0 0 0
    0 1 0 0 0 1 0 0 0 0 0
    -> 4 4 3 2 4 5
    """
    mask = batch == eos_id
    mask[:, -1] = True  # virtual eos at the end of each row

    # 0 0 0 1 0 0 0 1 0 0 X
    # 0 1 0 0 0 1 0 0 0 0 X
    row, col = torch.where(mask)

    # row = 0, 0, 0, 1, 1, 1
    # col = 3, 7, 10, 1, 5, 10
    seqlens = (col[1:] - col[:-1]) + (row[1:] - row[:-1]) * mask.shape[1]
    # seqlens = (4, 3, -9, 4, 5) + (0, 0, 11, 0, 0) = (4, 3, 2, 4, 5)
    return [int(col[0].item() + 1)] + seqlens.tolist()


def create_causal_mask(
    seqlen,
    attn_impl: str,
    attn_bias_type: str | None,
    *,
    eos_id: int | None = None,
    tokens: torch.Tensor | None = None,
    sliding_window: int | None = None,
):
    if attn_impl == "xformers":
        if attn_bias_type is None:
            return fmha.attn_bias.LowerTriangularMask()
        elif attn_bias_type == "causal":
            assert sliding_window is None
            return fmha.attn_bias.LowerTriangularMask()
        elif attn_bias_type == "block_causal":
            assert sliding_window is None
            assert eos_id is not None
            assert tokens is not None
            return fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(
                q_seqlen=tokens_to_seqlen(tokens, eos_id)
            )
        elif attn_bias_type == "local_block_causal":
            assert sliding_window is not None
            assert eos_id is not None
            assert tokens is not None
            return fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(
                q_seqlen=tokens_to_seqlen(tokens, eos_id)
            ).make_local_attention(sliding_window)
        else:
            return fmha.attn_bias.LocalAttentionFromBottomRightMask(
                window_left=sliding_window - 1, window_right=0
            )
    elif attn_impl == "sdpa":
        BLT_SUPPRESS_ATTN_ERROR = int(os.environ.get("BLT_SUPPRESS_ATTN_ERROR", 0))

        if attn_bias_type == "causal":
            return "causal"

        if BLT_SUPPRESS_ATTN_ERROR == 1:
            return "causal"
        else:
            raise ValueError(
                "SDPA attention being used, which doesn't have specialized attention implementations for block_causal and local_block_causal attention. To suppress this error and run the model anyway, set the environment variable BLT_SUPPRESS_ATTN_ERROR=1"
            )
    elif attn_impl == "flex_attention":
        return create_block_mask(causal_mask, None, None, seqlen, seqlen)
    elif attn_impl == "fmha":
        return None
    else:
        raise NotImplementedError(
            f"Attention {attn_impl} with {sliding_window} sliding window not implemented"
        )


def check_param_device(model, device_type: str = "cpu"):
    for name, param in model.named_parameters():
        assert (
            param.device.type == device_type
        ), f"Parameter {name} is on {param.device.type}, not on {device_type}"
