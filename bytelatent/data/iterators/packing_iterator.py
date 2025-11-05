# Copyright (c) Meta Platforms, Inc. and affiliates.
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from bytelatent.data.data_types import Batch, BltSequence
from bytelatent.data.iterators.abstract_iterator import (
    PydanticIteratorState,
    StatefulIterator,
)
from bytelatent.data.iterators.sampling_iterator import SamplingIteratorState


class PackingMode(str, Enum):
    BYTES = "bytes"
    PATCHING = "patching"


class PackingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int
    seq_len: int
    pad_id: int
    max_length: int | None
    pad_to_max_length: bool
    enable_byte_ngrams: bool
    packing_mode: PackingMode


class PackingIteratorState(PydanticIteratorState):
    model_config = ConfigDict(extra="forbid")
    sequence_iterator_state: SamplingIteratorState
    packing_args: PackingArgs

    def build(self) -> "PackingIterator":
        return PackingIterator(
            sequence_iterator=self.sequence_iterator_state.build(),
            packing_args=self.packing_args,
        )


def _merge_patch_seq_masks(bs: int, slen: int, mask_seqs: list[list[bool]]):
    assert len(mask_seqs) == bs
    lens = [len(m) for m in mask_seqs]
    if all(all(m) for m in mask_seqs) and all(lens[0] == l for l in lens):
        return np.ones((bs, slen), dtype=bool)
    assert slen == max(lens) - 1, f"slen={slen} != max(lens)-1={max(lens) - 1}"
    mask = np.zeros((bs, slen), dtype=bool)
    for i, m in enumerate(mask_seqs):
        if m is None:
            print(
                "Did not implement None mask, the mask should be True for all toks, so we need to pass that to this function."
            )
            raise NotImplementedError
        mask[i][: len(mask_seqs[i]) - 1] = mask_seqs[i][1:]
    return mask


def truncate_batch(
    batch: Batch,
    max_length: int,
    pad_id: int,
    pad_to_max_length: bool = False,
    *,
    enable_byte_ngrams: bool,
):
    """
    Truncate the x to a given size, making sure we remove the corresponding patch sizes in patch_lenghts
    and fixing the batch.mask.

    batch.patch_lengths has unchanged shape
    x,y, and mask may reduce in size
    """
    if batch.patch_lengths is None:
        return batch

    seq_lengths = batch.patch_lengths.sum(axis=1)
    max_length_adj = max_length + 1
    if np.any(seq_lengths > max_length_adj):
        for i in range(batch.x.shape[0]):
            if seq_lengths[i] > max_length_adj:
                # Find id of patch that tips over max_length + 1
                count, j = 0, 0
                while count + batch.patch_lengths[i, j] <= max_length_adj:
                    count += batch.patch_lengths[i, j]
                    j += 1
                # Edit the batch
                assert j < batch.patch_lengths.shape[1]
                batch.x[i, max_length:] = pad_id
                batch.y[i, max_length:] = pad_id
                if batch.mask is not None:
                    batch.mask[i, max_length:] = False
                batch.patch_lengths[i, j:] = 0
                batch.patch_lengths[i, j] = max_length_adj - count

        # Truncate if necessary.
        if max_length < batch.x.shape[1]:
            batch.x = batch.x[:, :max_length]
            batch.y = batch.y[:, :max_length]
            if batch.mask is not None:
                batch.mask = batch.mask[:, :max_length]

    # Right pad to max_length if necessary
    elif pad_to_max_length:
        if batch.x.shape[1] < max_length:
            # NOTE: this has to be done on an actual patch.
            non_zero_indices = (batch.patch_lengths != 0).sum(axis=1) - 1
            non_zero_indices = np.maximum(0, non_zero_indices)
            batch.patch_lengths[range(len(batch.patch_lengths)), non_zero_indices] += (
                max_length - batch.x.shape[1]
            )
            # TODO: We could get rid of many of these complications by moving this funciton directly in the dataloader.
            x = np.full((batch.x.shape[0], max_length), pad_id, dtype=batch.x.dtype)
            x[:, : batch.x.shape[1]] = batch.x
            batch.x = x
        if batch.y.shape[1] < max_length:
            y = np.full((batch.y.shape[0], max_length), pad_id, dtype=batch.y.dtype)
            y[:, : batch.y.shape[1]] = batch.y
            batch.y = y
        if batch.mask is not None and batch.mask.shape[1] < max_length:
            mask = np.full(
                (batch.mask.shape[0], max_length), False, dtype=batch.mask.dtype
            )
            mask[:, : batch.mask.shape[1]] = batch.mask
            batch.mask = mask

    assert batch.x.shape[1] <= max_length
    assert batch.y.shape[1] <= max_length
    assert batch.mask is None or batch.mask.shape[1] <= max_length
    assert np.all(max_length_adj - batch.patch_lengths.sum(axis=1) == 0)
    if pad_to_max_length:
        assert batch.x.shape[1] == max_length
        assert batch.y.shape[1] == max_length
        assert batch.mask is None or batch.mask.shape[1] == max_length
    if enable_byte_ngrams:
        raise NotImplementedError()
        # (num_ngram, batch_size, seq_len)
        ngram_ids = np.array(tokenizer.encode_token_ngrams(batch.x))
        assert ngram_ids.shape[2] == batch.x.shape[1]
    else:
        ngram_ids = None
    batch.ngram_ids = ngram_ids


class PackingIterator(StatefulIterator[Batch, PackingIteratorState]):
    def __init__(
        self,
        sequence_iterator: StatefulIterator[BltSequence, Any],
        *,
        packing_args: PackingArgs,
    ):
        self.sequence_iterator = sequence_iterator
        self.packing_args = packing_args

    def get_state(self):
        return PackingIteratorState(
            sequence_iterator_state=self.sequence_iterator.get_state(),
            packing_args=self.packing_args,
        )

    def create_iter(self):
        if self.packing_args.packing_mode == PackingMode.BYTES:
            return self._create_iter_from_bytes()
        elif self.packing_args.packing_mode == PackingMode.PATCHING:
            return self._create_iter_from_patch_lengths()
        else:
            raise ValueError(f"Invalid patching mode: {self.packing_args.packing_mode}")

    def _create_iter_from_bytes(self):
        sequence_iter = self.sequence_iterator.create_iter()
        batch_size = self.packing_args.batch_size
        pad_id = self.packing_args.pad_id
        seq_len = self.packing_args.seq_len
        while True:
            tokens: list[list[int]] = []
            masks: list[list[bool]] = []
            stop_iteration = False
            try:
                for _ in range(self.packing_args.batch_size):
                    sequence = next(sequence_iter)
                    _tokens = sequence.tokens
                    _mask = sequence.mask
                    assert (
                        sequence.patch_lengths is None
                    ), "patch_lengths should not be used in byte packing"
                    tokens.append(_tokens)
                    masks.append(_mask)
            except StopIteration:
                # At this point, there will be no new sequences, so we need to stop
                # after yielding the already accumulated data (one batch).
                # In this case, either:
                # 1. We have a complete batch, so things go as normal
                # 2. We have an incomplete batch, but due to creating a right sized batch,
                #   then filling the values in, this case is automatically handled.
                stop_iteration = True

            x = np.full((batch_size, seq_len), fill_value=pad_id)
            y = np.full((batch_size, seq_len), fill_value=pad_id)
            m = np.zeros((batch_size, seq_len), dtype=np.bool)

            for i, tok_seq in enumerate(tokens):
                x[i, : len(tok_seq) - 1] = tok_seq[:-1]
                y[i, : len(tok_seq) - 1] = tok_seq[1:]
                m[i, : len(tok_seq)] = masks[i][1:]

            batch = Batch(x=x, y=y, mask=m)
            assert (
                batch.mask is None or np.sum(x != pad_id) == batch.mask.sum()
            ), f"{np.sum(x != pad_id)} != {batch.mask.sum()}"
            yield batch
            if stop_iteration:
                break

    def _create_iter_from_patch_lengths(self):
        sequence_iter = self.sequence_iterator.create_iter()
        batch_size = self.packing_args.batch_size
        pad_id = self.packing_args.pad_id
        seq_len = self.packing_args.seq_len
        pad_to_max_length = self.packing_args.pad_to_max_length
        enable_byte_ngrams = self.packing_args.enable_byte_ngrams
        max_length = self.packing_args.max_length
        assert max_length is not None
        final_leftover_batch = False
        while True:
            tokens: list[list[int]] = []
            masks: list[list[bool]] = []
            patch_lengths: list[list[int]] = []
            stop_iteration = False
            try:
                for _ in range(self.packing_args.batch_size):
                    sequence = next(sequence_iter)
                    _tokens = sequence.tokens
                    _mask = sequence.mask
                    _patch_lengths = sequence.patch_lengths
                    assert (
                        _patch_lengths is not None
                    ), "patch lengths are required for packing based on patches."
                    # Reminder: seq_len is in terms of patches
                    assert len(sequence.patch_lengths) == self.packing_args.seq_len
                    last_patch_length = 0
                    if _patch_lengths[0] > 1:
                        last_patch_length = _patch_lengths[-1]
                        _patch_lengths[0] -= 1
                        _patch_lengths = [1] + _patch_lengths[:-1]
                    tokens.append(_tokens[: len(_tokens) - last_patch_length])
                    masks.append(_mask[: len(_mask) - last_patch_length])
                    patch_lengths.append(_patch_lengths)
            except StopIteration:
                stop_iteration = True

            if len(tokens) == 0 and stop_iteration:
                break

            x_patch_lengths = np.array(patch_lengths)
            assert (
                x_patch_lengths.shape[1] == seq_len
            ), f"{x_patch_lengths.shape[1]} vs {seq_len}"
            # pad batch to same length
            tok_seq_len = max([len(toks) for toks in tokens]) - 1
            x = np.full((batch_size, tok_seq_len), fill_value=pad_id)
            y = np.full((batch_size, tok_seq_len), fill_value=pad_id)

            for i, tok_seq in enumerate(tokens):
                x[i, : len(tok_seq) - 1] = tok_seq[:-1]
                y[i, : len(tok_seq) - 1] = tok_seq[1:]
                # Adjust patch lengths to match x
                x_patch_lengths[i, -1] += tok_seq_len - (len(tok_seq) - 1)

            if x_patch_lengths.shape[0] < batch_size:
                if final_leftover_batch:
                    raise ValueError(
                        "There should only be one partial batch, but found multiple"
                    )
                final_leftover_batch = True
                assert len(masks) == len(x_patch_lengths)
                n_missing = batch_size - x_patch_lengths.shape[0]
                # Repeat the last patch length to validly pad it out, but
                # update the mask to ignore the row
                x_patch_lengths = np.vstack(
                    [
                        x_patch_lengths,
                        np.repeat(x_patch_lengths[-1:, :], n_missing, axis=0),
                    ]
                )
                for _ in range(n_missing):
                    masks.append([0] * tok_seq_len)
                assert len(masks) == batch_size

            assert x_patch_lengths.shape == (
                batch_size,
                seq_len,
            ), f"{x_patch_lengths.shape} vs {(batch_size, seq_len)}"

            if enable_byte_ngrams:
                raise NotImplementedError()
            else:
                ngram_ids = None

            batch = Batch(
                x=x,
                y=y,
                patch_lengths=x_patch_lengths,
                ngram_ids=ngram_ids,
                mask=_merge_patch_seq_masks(batch_size, tok_seq_len, masks),
            )

            assert (
                x_patch_lengths.sum() == x.size + batch_size
            ), f"{x_patch_lengths.sum()} != {x.size + batch_size}"
            assert (
                batch.mask is None or np.sum(x != pad_id) == batch.mask.sum()
            ), f"{np.sum(x != pad_id)} != {batch.mask.sum()}"
            assert np.all(
                x_patch_lengths[:, 0] == 1
            ), f"first patch should always be 1, {x_patch_lengths[:, 0]}"
            # cuda_gb_allocated = (torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024)
            # cuda_gb_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
            # print(f"dataloader cuda_gb_allocated: {cuda_gb_allocated}, cuda_gb_reserved: {cuda_gb_reserved}")
            truncate_batch(
                batch,
                max_length=max_length,
                pad_id=pad_id,
                pad_to_max_length=pad_to_max_length,
                enable_byte_ngrams=enable_byte_ngrams,
            )
            yield batch
            if stop_iteration:
                break
