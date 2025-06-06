# Copyright (c) Meta Platforms, Inc. and affiliates.

from enum import auto, Enum
from typing import Any, Optional

import torch

from bytelatent.base_transformer import (
    BaseTransformerArgs,
    InitStdFactor,
    SequenceModelWithOutput,
)
from bytelatent.data.patcher import Patcher, PatcherArgs
from bytelatent.model.latent_transformer import GlobalTransformer
from bytelatent.model.local_models import LocalDecoder, LocalEncoder, LocalModelArgs
from bytelatent.model.utils import check_param_device, downsample, DTYPE_MAP
from bytelatent.tokenizers.constants import BOE_ID, BOS_ID, EOS_ID, OFFSET, PAD_ID
from huggingface_hub import PyTorchModelHubMixin

from numpy.random import f
from pydantic import model_validator
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from typing_extensions import Self


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


def setattrs(_self, **kwargs):
    for k, v in kwargs.items():
        setattr(_self, k, v)


def get_encoder_dim_token_emb(args):
    if args.dim_token is not None:
        dim_token_emb = args.dim_token
    elif args.use_local_encoder_transformer:
        dim_token_emb = args.dim_local_encoder
    else:
        dim_token_emb = args.dim_global // args.patch_size
    return dim_token_emb


def get_encoder_dim_patch_emb(args):
    dim_patch_emb = None
    if args.cross_attn_encoder:
        if args.cross_attn_init_by_pooling:
            dim_patch_emb = args.dim_local_encoder
        else:
            dim_patch_emb = args.dim_global
    return dim_patch_emb


def get_global_dim_patch_emb(args):
    dim_token_emb = get_encoder_dim_token_emb(args)
    if args.cross_attn_encoder:
        dim_patch_emb = dim_token_emb * args.cross_attn_k
    elif (
        args.downsampling_by_pooling is None
        or not args.downsampling_by_pooling
        or len(args.downsampling_by_pooling) == 0
    ):
        dim_patch_emb = dim_token_emb * args.patch_size
    else:
        dim_patch_emb = dim_token_emb * sum(
            [
                pooling in args.downsampling_by_pooling
                for pooling in ["avg", "min", "max"]
            ]
        )
    return dim_patch_emb


def get_decoder_dim_token_emb(args):
    if args.share_encoder_decoder_emb:
        dim_token_emb = get_encoder_dim_token_emb(args)
    elif args.dim_token is not None:
        dim_token_emb = args.dim_token
    else:
        dim_token_emb = args.dim_local_decoder
    return dim_token_emb


def parse_ngram_to_size(ngram_to_size_str: str | None) -> dict[int, int]:
    if ngram_to_size_str is None:
        return None
    ngram_to_size = {}
    for entry in ngram_to_size_str.split(","):
        ngram, size = entry.split(":")
        ngram = int(ngram)
        size = int(size)
        ngram_to_size[ngram] = size
    return ngram_to_size


def fill_tokens(tokens, patch_size, fill_id):
    batch_size, seq_len = tokens.shape
    if seq_len % patch_size == 0:
        return tokens
    else:
        remaining = patch_size - seq_len % patch_size
        final_padding = tokens.new(batch_size, remaining).fill_(fill_id)
        return torch.cat((tokens, final_padding), dim=1)


def decoder_patch_ids_from_lengths(patch_lengths, nb_boe, seq_len):
    first_patch_length = patch_lengths[0, 0]
    assert torch.all(
        first_patch_length == patch_lengths[:, 0]
    ), "first patch should always be the same size (1 for dynamic, patch_size for static)."
    assert (
        first_patch_length - nb_boe == 1
    ), f"First patch (patch length: {first_patch_length}) should have one non-boe token (boe toks: {nb_boe})"
    # Remove first patch from patch_ids for local decoder inputs and shift the last patch.
    # decoder_patch_lengths = patch_lengths[:, 1:].clone()
    # decoder_patch_lengths = add_to_last_nonzero_patch(decoder_patch_lengths, 1)
    decoder_patch_lengths = patch_lengths[:, 1:]
    assert (
        decoder_patch_lengths.sum() + (nb_boe + 1) * patch_lengths.shape[0]
        == patch_lengths.sum()
    ), f"{decoder_patch_lengths.sum() + (nb_boe + 1) * patch_lengths.shape[0]} != {patch_lengths.sum()}"
    assert torch.all(decoder_patch_lengths >= 0), f"{decoder_patch_lengths}"
    decoder_patch_ids = patch_ids_from_lengths(
        patch_lengths=decoder_patch_lengths, seq_len=seq_len
    )
    return decoder_patch_ids


primes = [
    1000000007,
    5915587277,
    1500450271,
    3267000013,
    5754853343,
    4093082899,
    9576890767,
    3628273133,
    2860486313,
    5463458053,
    3367900313,
]


def rolling_polynomial_hash(t, hash_func_nb: int = 0):
    if hash_func_nb >= len(primes):
        print(f"len(primes): {len(primes)}, hash_func_nb: {hash_func_nb}")

    prime = torch.tensor(primes[hash_func_nb], dtype=torch.int64, device=t.device)
    prime_powers = torch.stack([prime**i for i in range(t.shape[-1])])
    return torch.sum(t * prime_powers, dim=-1)


def get_rolling_polynomial_hash_fn(hash_func_nb: int = 0, group_size: int = 2):
    prime = torch.tensor(primes[hash_func_nb], dtype=torch.int64)
    prime_powers = torch.stack([prime**i for i in range(group_size)])

    def rolling_polynomial_hash_fn(t):
        return torch.sum(t * prime_powers, dim=-1)

    return rolling_polynomial_hash_fn


def byte_group_hash_function(
    x: torch.Tensor, group_size: int = 2, hash_func_nb: int = 0, max_hash: int = 30000
):
    """
    Returns a hash of the input x and maps it to a value in the range [0, max_hash].

    expects: x of shape (batch_size, seq_len) with values as ids in the token vocab.
    returns a tensor  of shape (batch_size, seq_len) with values in the range [0, max_hash].

    Note: max hash can make a big difference on the number of collisions.
    """
    with torch.no_grad():
        bs, seq_len = x.shape
        # x_numpy = x.numpy()
        # hash_values = torch.zeros(bs, seq_len, dtype=torch.int64, requires_grad=False)
        # for i in range(bs):
        #     for j in range(seq_len):
        #         start = max(j, j-group_size+1)
        #         end = j+1
        #         hash_values[i, j] = hash_array(x_numpy[i, start:end], max_hash)

        prefix = torch.zeros(bs, group_size - 1, dtype=torch.int64, device=x.device)
        x = torch.cat([prefix, x], dim=1)
        windows = x.unfold(1, group_size, 1)
        # hashes = get_rolling_polynomial_hash_fn(hash_func_nb, group_size)(windows)
        hashes = rolling_polynomial_hash(windows, hash_func_nb)
        hash_values_range = hashes % max_hash
    hash_values_range.requires_grad = False
    return hash_values_range


def create_patch_mask_from_ids(
    patch_ids, num_patches, window=None, patches_as_queries=False
):
    """
    Creates a tensor of shape [bs, seq_len, num_patches] where each element at position (i, j, k)
    is True if the patch id at position (i, j) is less than or equal to k.
    Args:
        patch_ids (torch.Tensor): Tensor of shape [bs, seq_len] containing patch ids.
        num_patches (int): Total number of patches.
        window (int): If not None, only considers patches within a window of size window.
        patches_as_queries (bool): If True, the patches are used as queries
    Returns:
        torch.Tensor: Tensor of shape [bs, q_len, kv_len] with the desired mask.
    """
    bs, seq_len = patch_ids.shape
    if not patches_as_queries:
        q_ids = patch_ids.unsqueeze(-1).expand(bs, seq_len, num_patches)
        kv_ids = (
            torch.arange(num_patches, device=patch_ids.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(bs, seq_len, num_patches)
        )
    else:
        kv_ids = patch_ids.unsqueeze(1).expand(bs, num_patches, seq_len)
        q_ids = (
            torch.arange(num_patches, device=patch_ids.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(bs, num_patches, seq_len)
        )
    if window is None:
        mask = q_ids == kv_ids
    else:
        mask = (kv_ids <= q_ids) & (q_ids < kv_ids + window)
    return mask


GLOBAL = set()


def cross_attn_mask(
    patch_ids,
    patch_lengths,
    N,
    patches_as_queries=False,
    cross_attn_k=1,
    window=None,
    block_mask=True,
):
    bs = patch_ids.shape[0]
    with torch.no_grad():
        # Create the patch mask
        cross_mask = create_patch_mask_from_ids(
            patch_ids,
            patch_lengths.shape[1],
            window=window,
            patches_as_queries=patches_as_queries,
        ).repeat_interleave(cross_attn_k, dim=1 if patches_as_queries else -1)
        q_len = patch_lengths.shape[1] * cross_attn_k if patches_as_queries else N
        kv_len = N if patches_as_queries else patch_lengths.shape[1] * cross_attn_k
        assert cross_mask.shape == (
            bs,
            q_len,
            kv_len,
        ), f"{cross_mask.shape} != {(bs, q_len, kv_len)}"
        if block_mask:
            # This appears to resolve occasional nondeterministic RuntimeErrors
            # in the create_block_mask call. I have no idea why.
            cross_mask_copy = cross_mask.clone()

            def patch_mask(b, h, q_idx, kv_idx):
                return cross_mask_copy[b, q_idx, kv_idx]

            # print(f"cross_mask: {cross_mask.shape}")
            # print(f"bs: {bs}, q_len: {q_len}, kv_len: {kv_len}")
            # print(cross_mask[0, 0, 0])
            # for i in range(bs):
            #     for j in range(q_len):
            #         for k in range(kv_len):
            #             y = cross_mask[i, j, k]

            # import pickle

            # with open("cross_mask.pkl", "wb") as f:
            #     pickle.dump(cross_mask, f)

            # global GLOBAL
            # s = f"bs_{bs}_q_len_{q_len}_kv_len_{kv_len}"
            # if s not in GLOBAL:
            #     GLOBAL.add(s)
            #     print(f"bs_{bs}_q_len_{q_len}_kv_len_{kv_len}")
            # else:
            #     print(f"bs_{bs}_q_len_{q_len}_kv_len_{kv_len} (skipped)")

            # if q_len >= 51 and kv_len >= 96:
            #     breakpoint()

            block_mask = create_block_mask(
                patch_mask,
                B=bs,
                H=None,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                _compile=True,
            )

            # print(f"block_mask_shape: {block_mask.shape}")

            return block_mask
        else:
            return torch.where(
                cross_mask, torch.tensor(0.0), torch.tensor(float("-inf"))
            ).unsqueeze(
                1
            )  # [bs, 1, q_len, kv_len]


def get_blt_input(
    tokens: torch.Tensor,
    enforce_patch_size_multiple: bool,
    nb_boe: torch.Tensor,
    patch_size: int,
    boe_id: int,
):
    """
        This function returns X_et, X_gt and X_dt, the encoder, global, and decoder
    tokens respectively.

    Consider the input and target sequences:
    X=[3,4,5,6,7,eos,bos,8,9,10,eos,bos,11,12,13]
    Y=[4,5,6,7,eos,bos,8,9,10,eos,bos,11,12,13,14]
    with patch_size=4

    Note 1: that there will be no special tokens introduced at the patch level.
    Note 2: X_e needs to be trimmed to be passed to Global

    Current without boe:
    X_et = [[boe,boe,boe,boe] [3,4,5,6],      [7,eos,bos,8],    [9,10,eos,bos] [11,12,13, pad]]
    X_g =  [[boe,boe,boe,boe] [3,4,5,6],      [7,eos,bos,8],    [9,10,eos,bos] [11,12,13, pad]] # remove last glob patch
    X_dt = [[3,4,5,6]         [7,eos,bos,8],  [9,10,eos,bos],   [11,12,13]]
    Y =    [[4,5,6,7]         [eos,bos,8,9],  [10,eos,bos,11],  [12,13,14]]

    --> lag fix:
    X_et = [[boe,boe,boe,3]   [4,5,6,7],      [eos,bos,8,9],    [10,eos,bos,11] [12,13,pad,pad]]
    X_g =  [[boe,boe,boe,3]   [4,5,6,7],      [eos,bos,8,9],    [10,eos,bos,11]]
    X_dt = [[3,4,5,6]         [7,eos,bos,8],  [9,10,eos,bos],   [11,12,13]]
    Y =    [[4,5,6,7]    	  [eos,bos,8,9],  [10,eos,bos,11],  [12,13,14]]

    Dynamic (current):
    X = [3,4,5,6,7,eos,bos,8,9,10,eos,bos]
    Y = [4,5,6,7,eos,bos,8,9,10,eos,bos,11]

    entropy patching:
    input: 7, bos, 9, 10
    pred (high entropy): eos, 8, 10, eos

    X_et = [[boe,3,4,5,6,7,eos,bos,8,9,10,eos,bos]
    X_g =  [[boe],      [3,4,5,6], [7,eos],[bos,8],[9],     [10,eos]]
    X_dt = [[3,4,5,6],  [7,eos],   [bos,8],[9],    [10,eos],[bos]]
    Y =    [4,5,6,7,eos,bos,8,9,10,eos,bos,11]

    --> lag fix no boe (force single byte first patch):
    X_et = [[3,4,5,6,7,eos,bos,8,9,10,eos,bos,11,12]
    X_g =  [[3],        [4,5,6,7], [eos,bos],[8,9], [10],       [eos,bos],      [11,12]] # remove last global patch
    X_dt = [[3,4,5,6],  [7,eos],   [bos,8], [9],    [10,eos],   [bos,11,12]]
    Y =    [4,5,6,7,    eos,bos,    8,9,    10,     eos,bos,    11,12,13]

    input: 4, 7, bos, 9, 10
    pred (high entropy): 5, eos, 8, 10, eos

    X_et = [[3,4,5,6,7,eos,bos,8,9,10,eos,bos,11,12]
    X_g =  [[3],        [4]   ,   [5,6,7], [eos,bos],[8,9], [10],       [eos,bos],      [11,12]] # remove last global patch
    X_dt = [[3]         [4,5,6],  [7,eos],   [bos,8], [9],    [10,eos],   [bos,11,12]]
    Y =    [4,]         [5,6,7,    eos,bos,    8,9,    10,     eos,bos,    11,12,13]

    Handle the last byte properly.
    patch_lengths = [1, 1,         3,      2,         2      1           2               2         1]
    X_et = [[3,4,5,6,7,eos,bos,8,9,10,eos,bos,11,12]
    X_g =  [[3],        [4]   ,   [5,6,7], [eos,bos],[8,9], [10],       [eos,bos],      [11,12]] # do not remove last global patch
    X_dt = [[3]         [4,5,6],  [7,eos],   [bos,8], [9],    [10,eos],   [bos,11]       [12]]
    Y =    [4,]         [5,6,7,    eos,bos,    8,9,    10,     eos,bos,    11,12,        13]]


    bpe delim
    X_et = [[3,4,5,6,7,<d>,eos,bos,<d>,8,9,<d>,10,<d>,eos,bos,11,12]
    X_g =  [[3],          [4,5,6,7,<d>],     [eos,bos,<d>], ..
    X_dt = [[3,4,5,6,7],  [<d>,eos,bos],     [<d>,bos,8], ..
    Y =    [4,5,6,7,<d>,    eos,bos,<d>       8,9,<d>, ..


    Note 1: that there will be no special tokens introduced at the patch level.
    Note 2: X_e needs to be trimmed to be passed to Global
    """
    batch_size, seq_len = tokens.shape
    local_encoder_tokens = tokens
    local_decoder_tokens = tokens

    if nb_boe > 0:
        padded_patch = tokens.new(batch_size, nb_boe).fill_(boe_id)
        local_encoder_tokens = torch.cat((padded_patch, local_encoder_tokens), dim=1)
    # global_tokens = tokens.new(batch_size, ((seq_len-1) // patch_size)+1).fill_(boe_id)

    # create global tokens, contains boe tokens and eos
    # padded_local_encoder_tokens = fill_tokens(local_encoder_tokens, patch_size, boe_id)
    # patches = padded_local_encoder_tokens.view(batch_size, -1, patch_size)
    # global_tokens = (patches.eq(eos_id).any(dim=2).int() * eos_id)[:, 1:]
    # global_tokens += global_tokens.eq(0).int() * boe_id
    # TODO: fix this when we want to use block causal in the global.

    if enforce_patch_size_multiple and local_encoder_tokens.shape[-1] % patch_size != 0:
        local_encoder_tokens = fill_tokens(local_encoder_tokens, patch_size, boe_id)

    return local_encoder_tokens, None, local_decoder_tokens


def patch_ids_from_lengths(patch_lengths, seq_len):
    bs, num_patches = patch_lengths.shape
    # Create a tensor of cumulative sums of the patch lengths
    cum_d = torch.cat(
        [
            torch.zeros(bs, 1, dtype=patch_lengths.dtype, device=patch_lengths.device),
            patch_lengths.cumsum(dim=-1),
        ],
        dim=-1,
    )
    patch_ids = (cum_d.unsqueeze(-1) <= torch.arange(seq_len, device=cum_d.device)).sum(
        dim=-2
    ) - 1
    assert not (
        torch.max(patch_ids) > patch_lengths.shape[-1] or torch.min(patch_ids) < 0
    ), f"{torch.max(patch_ids)} > {patch_lengths.shape[-1]} or {torch.min(patch_ids)} < 0"
    return patch_ids


class ByteLatentTransformerArgs(BaseTransformerArgs):
    # Basic model configuration
    seed: int = 42
    vocab_size: int = -1
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    # TODO: What is the purpose of this parameter?
    weight_tying: bool = False
    patch_in_forward: bool = False

    # Architecture and dimensions
    dim_token: int | None = None
    dim_global: int = 512
    dim_local_decoder: int = 512
    dim_local_encoder: int = 512
    n_layers_global: int = 8
    n_layers_local_decoder: int = 8
    n_layers_local_encoder: int = 8

    # Tokenization and patching
    patch_size: float | None = None
    patching_mode: str | None = None
    patching_threshold: float | None = None
    patching_threshold_add: float | None = None
    monotonicity: bool = False
    patching_batch_size: int = 1
    patching_device: str = "cuda"
    max_patch_length: int | None = None

    # Encoder/Decoder configuration
    tie_local_encoder_decoder_logits: bool = False
    use_local_encoder_transformer: bool = False
    encoder_lm_loss: bool = False
    max_encoder_seq_length: int | None = None
    pad_to_max_length: bool = False
    encoder_enable_byte_ngrams: bool = False
    encoder_enable_byte_group_hash: bool = False
    ngram_vocab_sizes: int | None = None

    # Cross attention configurations
    cross_attn_encoder: bool = False
    cross_attn_decoder: bool = False
    cross_attn_window_encoder: int | None = None
    cross_attn_window_decoder: int | None = None
    cross_attn_k: int | None = None
    cross_attn_nheads: int | None = None
    cross_attn_all_layers_decoder: bool = False
    cross_attn_all_layers_encoder: bool = False
    cross_attn_use_flex_attention: bool = True
    cross_attn_init_by_pooling: bool = False

    # Encoder hash configurations
    encoder_hash_byte_group_size: Any | None = None
    encoder_hash_byte_group_vocab: int = 30000
    encoder_hash_byte_group_nb_functions: int = 3

    # Model behavior and optimization
    log_patch_lengths: bool = False
    non_linearity: str = "swiglu"
    use_rope: bool = True
    recompute_fc1_out: bool = False
    recompute_fc3_out: bool = False
    recompute_attn: bool = True
    custom_bwd: bool = False
    layer_ckpt: str = "all"

    # Initialization and attention
    init_use_gaussian: bool = True
    init_use_depth: str = "current"
    attn_bias_type: str = "causal"
    alpha_depth: str = "disabled"
    max_length: int = 2048

    # Norm configuration
    norm_eps: float = 1e-5
    norm_affine: bool = True
    pre_norm: bool = True
    norm_type: str = "rmsnorm"

    # Additional configurations
    multiple_of: int = 256
    ffn_dim_multiplier: float = 1.0
    dropout: float = 0
    output_size: int = -1

    # Additional parameters from ModelArgs
    architecture: str = "vanilla"
    share_encoder_decoder_emb: bool = True
    global_local_decoder_residual_layer: str | None = None

    tokenize_with_bpe_delimiter: bool = False
    patching_thresholds_str: str | None = None
    tie_local_encoder_decoder: bool = False
    encoder_preds_low_entropy_toks: float | None = None
    encoder_preds_random_toks: float | None = None
    dim_token_emb: int | None = None
    dim_patch_emb: int | None = None

    encoder_ngram_table_dir: str | None = None
    encoder_ngram_to_size_str: str | None = None

    # Model architecture params
    entropy_model_checkpoint_dir: str | None = None
    entropy_model_is_ngram_model: bool = False
    downsampling_by_pooling: str | None = None
    n_heads_global: int = 8
    n_heads_local_decoder: int = 8
    n_heads_local_encoder: int = 8
    n_kv_heads: int | None = None
    n_kv_heads_global: int | None = None
    conv_kernel_size: int | None = None
    local_attention_window_len: int | None = None

    # Performance optimization
    sequence_parallel: bool = False
    loss_parallel: bool = False
    fuse_sequence_parallel: bool = False
    use_fsdp: bool = True
    attn_to_keep: str = "all"

    # Parameter mixing
    pm_size: int = 0

    # Logging
    full_logging_n_layers: int = 4

    @model_validator(mode="after")
    def check_hash_byte_sizes(self) -> Self:
        if (
            self.encoder_hash_byte_group_size is not None
            and type(self.encoder_hash_byte_group_size) == str
        ):
            self.encoder_hash_byte_group_size = [
                int(x)
                for x in self.encoder_hash_byte_group_size.split(",")
                if len(x) > 0
            ]
        return self


class GlobalTransformerArgs(ByteLatentTransformerArgs):
    # Global encoder specific dimensions
    dim_token_emb: int | None = None
    dim_patch_emb: int | None = None

    def __post_init__(self):
        # Override base args with global encoder specific values
        self.dim = self.dim_global
        self.n_layers = self.n_layers_global
        self.n_heads = self.n_heads_global
        self.n_kv_heads = self.n_kv_heads_global
        self.local_attention_window_len = None
        self.cross_attn_encoder = False
        self.cross_attn_decoder = False


class LocalDecoderArgs(ByteLatentTransformerArgs):
    # Local decoder specific dimensions
    dim_token_emb: int | None = None
    dim_patch_emb: int | None = None

    def __post_init__(self):
        # Override base args with local decoder specific values
        self.dim = self.dim_local_decoder
        self.n_layers = self.n_layers_local_decoder
        self.n_heads = self.n_heads_local_decoder
        self.cross_attn_encoder = False
        self.cross_attn_init_by_pooling = False
        self.attn_bias_type = "local_block_causal"


def create_global_transformer(args: ByteLatentTransformerArgs) -> GlobalTransformer:
    global_args = args.model_copy(
        deep=True,
        update=dict(
            dim=args.dim_global,
            n_layers=args.n_layers_global,
            n_heads=args.n_heads_global,
            n_kv_heads=args.n_kv_heads_global,
            local_attention_window_len=None,
            dim_token_emb=get_global_dim_patch_emb(args),
            dim_patch_emb=None,
            cross_attn_encoder=False,
            cross_attn_decoder=False,
        ),
    )

    return GlobalTransformer(global_args)


def create_local_encoder(args: ByteLatentTransformerArgs) -> LocalEncoder:
    local_encoder_args = LocalModelArgs(
        # Updated args
        dim=args.dim_local_encoder,
        n_layers=args.n_layers_local_encoder,
        n_heads=args.n_heads_local_encoder,
        dim_token_emb=get_encoder_dim_token_emb(args),
        dim_patch_emb=get_encoder_dim_patch_emb(args),
        cross_attn_encoder=args.cross_attn_encoder,
        cross_attn_decoder=False,
        cross_attn_k=args.cross_attn_k if args.cross_attn_encoder else None,
        cross_attn_init_by_pooling=args.cross_attn_init_by_pooling,
        # Defaults
        head_dim=args.head_dim,
        max_seqlen=args.max_encoder_seq_length,
        dropout=args.dropout,
        vocab_size=args.vocab_size + args.pm_size,
        norm_eps=args.norm_eps,
        patch_size=args.patch_size,
        sliding_window=args.local_attention_window_len,
        use_rope=args.use_rope,
        rope_theta=args.rope_theta,
        rope_use_fp32_in_outer_product=args.rope_use_fp32_in_outer_product,
        init_base_std=args.init_base_std,
        init_std_factor=args.init_std_factor,
        n_kv_heads=args.n_kv_heads,
        attn_impl=args.attn_impl,
        attn_bias_type="local_block_causal",
        multiple_of=args.multiple_of,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
        patching_mode=args.patching_mode,
        use_local_encoder_transformer=args.use_local_encoder_transformer,
        downsampling_by_pooling=args.downsampling_by_pooling,
        encoder_hash_byte_group_size=args.encoder_hash_byte_group_size,
        cross_attn_all_layers_encoder=args.cross_attn_all_layers_encoder,
        cross_attn_all_layers_decoder=args.cross_attn_all_layers_decoder,
        cross_attn_nheads=args.cross_attn_nheads,
        eos_id=args.eos_id,
        init_device=args.init_device,
        init_dtype=args.init_dtype,
    )

    return LocalEncoder(local_encoder_args)


def create_local_decoder(args: ByteLatentTransformerArgs) -> LocalDecoder:
    # First deep copy the original args
    local_decoder_args = LocalModelArgs(
        dim=args.dim_local_decoder,
        n_layers=args.n_layers_local_decoder,
        n_heads=args.n_heads_local_decoder,
        dim_token_emb=get_decoder_dim_token_emb(args),
        dim_patch_emb=args.dim_global,
        cross_attn_encoder=False,
        cross_attn_decoder=args.cross_attn_decoder,
        cross_attn_init_by_pooling=False,  # states are already defined
        cross_attn_k=args.cross_attn_k if args.cross_attn_decoder else None,
        # Defaults
        head_dim=args.head_dim,
        max_seqlen=args.max_encoder_seq_length,
        dropout=args.dropout,
        vocab_size=args.vocab_size + args.pm_size,
        norm_eps=args.norm_eps,
        patch_size=args.patch_size,
        sliding_window=args.local_attention_window_len,
        use_rope=args.use_rope,
        rope_theta=args.rope_theta,
        rope_use_fp32_in_outer_product=args.rope_use_fp32_in_outer_product,
        init_base_std=args.init_base_std,
        init_std_factor=args.init_std_factor,
        n_kv_heads=args.n_kv_heads,
        attn_impl=args.attn_impl,
        attn_bias_type="local_block_causal",
        multiple_of=args.multiple_of,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
        patching_mode=args.patching_mode,
        use_local_encoder_transformer=args.use_local_encoder_transformer,
        downsampling_by_pooling=args.downsampling_by_pooling,
        encoder_hash_byte_group_size=args.encoder_hash_byte_group_size,
        cross_attn_all_layers_encoder=args.cross_attn_all_layers_encoder,
        cross_attn_all_layers_decoder=args.cross_attn_all_layers_decoder,
        cross_attn_nheads=args.cross_attn_nheads,
        eos_id=args.eos_id,
        init_device=args.init_device,
        init_dtype=args.init_dtype,
    )

    return LocalDecoder(local_decoder_args)


class EmbeddingType(Enum):
    HASH_TOK = auto()
    NGRAM = auto()


def init_embeddings(
    args,
    embedding_type: EmbeddingType,
    local_encoder_dim: int,
    encoder_hash_byte_group_size: list = None,
):
    if (
        embedding_type == EmbeddingType.HASH_TOK
        and args.encoder_hash_byte_group_size is None
    ):
        return None
    if embedding_type == EmbeddingType.NGRAM and args.encoder_ngram_to_size_str is None:
        return None

    embeddings = []

    if embedding_type == EmbeddingType.HASH_TOK:
        emb_dim = local_encoder_dim
        encoder_hash_byte_group_vocab = args.encoder_hash_byte_group_vocab
        for _ in range(args.encoder_hash_byte_group_nb_functions):
            for _ in encoder_hash_byte_group_size:
                embeddings.append(
                    nn.Embedding(
                        encoder_hash_byte_group_vocab,
                        emb_dim,
                        device=args.init_device,
                        dtype=DTYPE_MAP[args.init_dtype],
                    )
                )

    elif embedding_type == EmbeddingType.NGRAM:
        encoder_ngram_to_size = parse_ngram_to_size(args.encoder_ngram_to_size_str)
        emb_dim = local_encoder_dim
        OFFSET = 4  # This should be passed as parameter if it's variable
        for ngram_vocab_size in encoder_ngram_to_size.values():
            embeddings.append(
                nn.Embedding(
                    ngram_vocab_size + OFFSET,
                    emb_dim,
                    device=args.init_device,
                    dtype=DTYPE_MAP[args.init_dtype],
                )
            )

    return nn.ModuleList(embeddings)


def compute_hash_embeddings(
    local_encoder_tokens: torch.Tensor,
    local_encoder,
    encoder_hash_tok_embedding: nn.ModuleList,
    encoder_hash_byte_group_nb_functions: int,
    encoder_hash_byte_group_size: list,
    encoder_hash_byte_group_vocab: int,
) -> torch.Tensor:
    """
    Compute embeddings using hash token embeddings.

    Args:
        local_encoder_tokens: Input tokens tensor
        local_encoder: Encoder object with tok_embeddings method
        encoder_hash_tok_embedding: ModuleList of hash token embeddings
        encoder_hash_byte_group_nb_functions: Number of hash functions
        encoder_hash_byte_group_size: List of byte group sizes
        encoder_hash_byte_group_vocab: Vocabulary size for hash embeddings

    Returns:
        torch.Tensor: Combined embeddings
    """
    if encoder_hash_tok_embedding is None:
        return None

    local_encoder_embeds = local_encoder.tok_embeddings(local_encoder_tokens)

    i = 0
    for func_nb in range(encoder_hash_byte_group_nb_functions):
        for byte_group_size in encoder_hash_byte_group_size:
            hash_ids = byte_group_hash_function(
                local_encoder_tokens,
                byte_group_size,
                hash_func_nb=func_nb,
                max_hash=encoder_hash_byte_group_vocab,
            )
            hash_tok_embedding = encoder_hash_tok_embedding[i]
            local_encoder_embeds = local_encoder_embeds + hash_tok_embedding(hash_ids)
            i += 1

    assert i == len(encoder_hash_tok_embedding)
    return local_encoder_embeds


class ByteLatentTransformer(
    nn.Module,
    SequenceModelWithOutput,
    PyTorchModelHubMixin,
    repo_url="https://github.com/facebookresearch/blt",
    paper_url="https://arxiv.org/abs/2412.09871",
    pipeline_tag="text-generation",
    license="other",
    license_name="fair-noncommercial-research-license",
    license_link="https://huggingface.co/facebook/blt/blob/main/LICENSE",
    coders={
        ByteLatentTransformerArgs: (
            lambda x: {"args": x.model_dump()},
            lambda data: ByteLatentTransformerArgs(**data),
        )
    },
):
    """
    The ByteLatentTransformer (BLT) is a byte-level language model architecture that processes byte sequences
    by dynamically segmenting them into patches. It uses a combination of local encoders, global transformers,
    and local decoders to efficiently encode and decode byte sequences, leveraging patch-based processing for
    improved performance and inference efficiency.
    """

    def __init__(self, args: ByteLatentTransformerArgs):
        super(ByteLatentTransformer, self).__init__()

        # General configuration
        self.weight_tying = args.weight_tying
        self.patch_size = args.patch_size
        self.patching_mode = args.patching_mode
        self.boe_id, self.bos_id, self.pad_id, self.eos_id = (
            BOE_ID,
            BOS_ID,
            PAD_ID,
            EOS_ID,
        )
        self.downsampling_by_pooling = args.downsampling_by_pooling
        self.patching_threshold = args.patching_threshold
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen

        # Cross attention configuration
        self.cross_attn_encoder = args.cross_attn_encoder
        self.cross_attn_decoder = args.cross_attn_decoder
        self.cross_attn_k = args.cross_attn_k
        self.cross_attn_window_encoder = args.cross_attn_window_encoder
        self.cross_attn_window_decoder = args.cross_attn_window_decoder
        self.cross_attn_use_flex_attention = args.cross_attn_use_flex_attention

        # Encoder hash configuration
        self.encoder_hash_byte_group_size = args.encoder_hash_byte_group_size
        self.encoder_hash_byte_group_vocab = args.encoder_hash_byte_group_vocab
        self.encoder_hash_byte_group_nb_functions = (
            args.encoder_hash_byte_group_nb_functions
        )

        # ByteLatent modules
        self.local_encoder = create_local_encoder(args)
        self.global_transformer = create_global_transformer(args)
        self.local_decoder = create_local_decoder(args)
        self.encoder_hash_tok_embedding = init_embeddings(
            args,
            EmbeddingType.HASH_TOK,
            local_encoder_dim=self.local_encoder.dim,
            encoder_hash_byte_group_size=self.encoder_hash_byte_group_size,
        )
        self.encoder_ngram_embedding = init_embeddings(
            args,
            EmbeddingType.NGRAM,
            local_encoder_dim=self.local_encoder.dim,
            encoder_hash_byte_group_size=None,
        )

        # Encoder ngram embedding tables
        self.encoder_ngram_embedding = None
        if args.encoder_enable_byte_ngrams:
            self.encoder_ngram_embedding = nn.ModuleList()
            assert args.ngram_vocab_sizes is not None
            self.encoder_ngram_to_size = parse_ngram_to_size(
                args.encoder_ngram_to_size_str
            )
            ngram_emb_dim = self.local_encoder.dim
            for ngram_vocab_size in self.encoder_ngram_to_size.values():
                self.encoder_ngram_embedding.append(
                    nn.Embedding(
                        ngram_vocab_size + OFFSET,
                        ngram_emb_dim,
                        device=args.init_device,
                        dtype=dtype_map[args.init_dtype],
                    )
                )

        # Output layer
        assert args.vocab_size > 0, "vocab_size must be greater than 0"

        # Patcher module
        if args.patch_in_forward:
            self.patcher = Patcher(
                PatcherArgs(
                    patch_size=args.patch_size,
                    patching_mode=args.patching_mode,
                    patching_threshold=args.patching_threshold,
                    patching_threshold_add=args.patching_threshold_add,
                    monotonicity=args.monotonicity,
                    max_patch_length=args.max_patch_length,
                )
            )

        # Sanity check
        check_param_device(self, args.init_device)

    def push_to_hub(self, *args, **kwargs):
        raise ValueError(
            "For meta authors: Do not push BLT weights with this, save weights with save_pretrained() then push them manually to HF hub to ensure the repository metadata is correct."
        )

    def get_output_seq_len(self):
        return self.max_seqlen

    def forward(
        self,
        tokens: torch.Tensor,
        patch_lengths: Optional[torch.Tensor] = None,
        ngram_ids: Optional[torch.Tensor] = None,
    ):
        # Ensure ngram_ids is either a tensor or None
        assert (
            isinstance(ngram_ids, torch.Tensor) or ngram_ids is None
        ), f"ngram_ids must be a tensor or None, but was: {type(ngram_ids)}"

        bs, N = tokens.shape  # Batch size and sequence length

        # Get megabyte inputs
        nb_boe = int(0 if self.patching_mode != "" else self.patch_size - 1)
        local_encoder_tokens, _, local_decoder_tokens = get_blt_input(
            tokens=tokens,
            enforce_patch_size_multiple=False,
            nb_boe=nb_boe,
            patch_size=self.patch_size,
            boe_id=self.boe_id,
        )

        # Patching
        if patch_lengths is None:
            assert (
                getattr(self, "patcher", None) is not None
            ), "Patcher not defined and no patch_lengths passed."
            patch_lengths, tok_scores = self.patcher.patch(
                local_encoder_tokens,
                include_next_token=True,
                threshold=self.patcher.threshold,
            )
        else:
            if nb_boe > 0:
                patch_lengths[:, 0] += nb_boe

        assert torch.min(patch_lengths) >= 0

        # Generate patch IDs from patch_lengths
        patch_ids = patch_ids_from_lengths(
            patch_lengths, local_encoder_tokens.shape[-1]
        )
        assert torch.max(patch_ids) + 1 <= torch.max(
            (patch_lengths != 0).sum(dim=-1)
        ), f"{torch.max(patch_ids) + 1} > {torch.max((patch_lengths != 0).sum(dim=-1))}"

        cross_attn_mask_enc = None
        # Cross-attention encoder
        if self.cross_attn_encoder:
            cross_attn_mask_enc = cross_attn_mask(
                patch_ids,
                patch_lengths,
                N,
                patches_as_queries=True,
                cross_attn_k=self.cross_attn_k,
                window=self.cross_attn_window_encoder,
                block_mask=self.cross_attn_use_flex_attention,
            )

        # Hashing and embedding
        local_encoder_embeds = compute_hash_embeddings(
            local_encoder_tokens=local_encoder_tokens,
            local_encoder=self.local_encoder,
            encoder_hash_tok_embedding=self.encoder_hash_tok_embedding,
            encoder_hash_byte_group_nb_functions=self.encoder_hash_byte_group_nb_functions,
            encoder_hash_byte_group_size=self.encoder_hash_byte_group_size,
            encoder_hash_byte_group_vocab=self.encoder_hash_byte_group_vocab,
        )

        # N-gram table embeddings
        if self.encoder_ngram_embedding is not None:
            assert ngram_ids is not None, "ngram_ids must be provided"
            if local_encoder_embeds is None:
                local_encoder_embeds = self.local_encoder.tok_embeddings(
                    local_encoder_tokens
                )
            assert len(ngram_ids) == len(
                self.encoder_ngram_embedding
            ), f"ngram_ids.shape[0]={ngram_ids.shape[0]} versus len(encoder_ngram_embedding)={len(self.encoder_ngram_embedding)}, ngram_ids.shape={ngram_ids.shape}"
            for i in range(ngram_ids.shape[0]):
                ngram_embedding = self.encoder_ngram_embedding[i]
                ngram_embeds = ngram_embedding(ngram_ids[i])
                assert (
                    local_encoder_embeds.shape == ngram_embeds.shape
                ), f"Shape mismatch: {local_encoder_embeds.shape} vs {ngram_embeds.shape}, ngram_ids.shape={ngram_ids.shape}"
                local_encoder_embeds = local_encoder_embeds + ngram_embeds

        # Local encoder
        (h_encoder, h_cross), cache_encoder = self.local_encoder(
            tokens=local_encoder_tokens,
            embeds=local_encoder_embeds,
            patch_embeds=None,
            cross_mask=cross_attn_mask_enc,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
        )

        # Downsampling
        if not self.cross_attn_encoder:
            assert (
                patch_ids.shape[1] == h_encoder.shape[1]
            ), f"{patch_ids.shape[1]} != {h_encoder.shape[1]}"
            h = downsample(
                h_encoder,
                patch_lengths.shape[1],
                patch_lengths,
                patch_ids,
                downsampling_by_pooling=self.downsampling_by_pooling,
                patch_size=self.patch_size,
            )
        else:
            # Reshape h_cross
            h = h_cross.view(bs, patch_lengths.shape[1], -1)

        # Global transformer
        global_tokens = tokens.new(h.shape[0], h.shape[1]).fill_(self.boe_id)
        rows, cols = torch.where(local_encoder_tokens == self.eos_id)
        eos_patch_ids = patch_ids[rows, cols]
        global_tokens[rows, eos_patch_ids] = self.eos_id

        h, _ = self.global_transformer(
            embeds=h,
            tokens=global_tokens,
        )

        # Unpatching
        dec_embeds = h_encoder[:, nb_boe : nb_boe + N, :]

        # Generate decoder patch IDs
        decoder_patch_ids = decoder_patch_ids_from_lengths(
            patch_lengths, nb_boe, local_decoder_tokens.shape[-1]
        )
        assert (
            torch.max(decoder_patch_ids) + 1 <= h.shape[1]
        ), f"{torch.max(decoder_patch_ids) + 1} > {h.shape[1]}"
        assert (
            decoder_patch_ids.shape[1] == dec_embeds.shape[1]
        ), f"{decoder_patch_ids.shape[1]} != {dec_embeds.shape[1]}"

        # Cross-attention decoder
        if not self.cross_attn_decoder:
            h = torch.gather(
                h, 1, decoder_patch_ids.unsqueeze(-1).expand(-1, -1, h.shape[-1])
            )
            cross_attn_mask_dec = None
            assert local_decoder_tokens.shape == h.shape[:-1]
        else:
            cross_attn_mask_dec = cross_attn_mask(
                decoder_patch_ids,
                patch_lengths,
                N,
                patches_as_queries=False,
                cross_attn_k=self.cross_attn_k,
                window=self.cross_attn_window_decoder,
                block_mask=self.cross_attn_use_flex_attention,
            )

        # Local decoder
        output, _ = self.local_decoder(
            embeds=dec_embeds,
            patch_embeds=h,
            tokens=local_decoder_tokens,
            cross_mask=cross_attn_mask_dec,
        )
        return output

    def init_weights(self):
        self.local_encoder.init_weights()
        self.global_transformer.init_weights()
        self.local_decoder.init_weights()

        emb_std = self.local_encoder.dim ** (-0.5)
        for emb in self.encoder_hash_tok_embedding:
            nn.init.trunc_normal_(
                emb.weight,
                mean=0.0,
                std=emb_std,
                a=-3 * emb_std,
                b=3 * emb_std,
            )
