import numpy as np
import pytest

from bytelatent.data.data_types import BltSequence
from bytelatent.data.iterators.abstract_iterator import StatefulIterator
from bytelatent.data.iterators.packing_iterator import (
    PackingArgs,
    PackingIterator,
    PackingMode,
    _merge_patch_seq_masks,
)


class DummySequenceIterator(StatefulIterator):
    def __init__(
        self,
        *,
        seq_len: int,
        n_seqs: int,
        patch_lengths: list[int] | None = None,
        pad_id: int = 0,
    ):
        self.seq_len = seq_len
        self.n_seqs = n_seqs
        self.patch_lengths = patch_lengths
        self.pad_id = pad_id

    def get_state(self):
        raise NotImplementedError()

    def create_iter(self):
        for i in range(self.n_seqs):
            if self.patch_lengths is None:
                tokens = np.arange(
                    i * self.seq_len + 1, (i + 1) * self.seq_len + 1
                ).tolist()
                mask = [True] * self.seq_len  # type: ignore
                assert len(tokens) == self.seq_len
            else:
                n = sum(self.patch_lengths)
                tokens = np.arange(i * n + 1, (i + 1) * n + 1).tolist()
                assert len(tokens) == n
                mask = [True] * n
                assert len(mask) == len(tokens)
            yield BltSequence(
                tokens=tokens,
                mask=mask,
                patch_lengths=self.patch_lengths,
            )


def create_bytes_iter(*, seq_len: int, n_seqs: int, batch_size: int, pad_id: int):
    sequence_iterator = DummySequenceIterator(seq_len=seq_len, n_seqs=n_seqs)
    packing_iterator = PackingIterator(
        sequence_iterator,
        packing_args=PackingArgs(
            batch_size=batch_size,
            seq_len=seq_len,
            pad_id=pad_id,
            packing_mode=PackingMode.BYTES,
            max_length=None,
            pad_to_max_length=False,
            enable_byte_ngrams=False,
        ),
    )
    return packing_iterator.create_iter()


def create_patches_iter(
    *,
    seq_len: int,
    n_seqs: int,
    batch_size: int,
    pad_id: int,
    patch_lengths: list[int] | None,
    max_length: int,
):
    sequence_iterator = DummySequenceIterator(
        # seq_len=number of bytes, which for blt/patches, is max_length since seq_len is
        # in terms of number of patches
        seq_len=max_length,
        n_seqs=n_seqs,
        patch_lengths=patch_lengths,
    )
    packing_iterator = PackingIterator(
        sequence_iterator,
        packing_args=PackingArgs(
            batch_size=batch_size,
            seq_len=seq_len,
            pad_id=pad_id,
            packing_mode=PackingMode.PATCHING,
            max_length=max_length,
            pad_to_max_length=True,
            enable_byte_ngrams=False,
        ),
    )
    return packing_iterator.create_iter()


def test_last_batch_correctness_bytes():
    seq_len = 1024
    n_seqs = 10
    batch_size = 4
    pad_id = 0
    iterator = create_bytes_iter(
        seq_len=seq_len, n_seqs=n_seqs, batch_size=batch_size, pad_id=pad_id
    )
    batches = []
    n_nonpad = 0
    n_nonmask = 0
    for b in iterator:
        assert b.x.shape[0] == batch_size
        assert b.x.shape[1] == seq_len
        n_nonpad += (b.x != pad_id).sum()
        if b.mask is None:
            n_nonmask += b.x.size
        else:
            n_nonmask += b.mask.sum()
        batches.append(b)
    assert len(batches) == 3
    assert n_nonpad == n_nonmask == seq_len * n_seqs
    # The second half of the last batch should be all pads
    assert batches[-1].mask[2:].sum() == 0


def test_edgecase_batch_correctness_bytes():
    seq_len = 1024
    n_seqs = 10
    batch_size = 12
    pad_id = 0
    iterator = create_bytes_iter(
        seq_len=seq_len, n_seqs=n_seqs, batch_size=batch_size, pad_id=pad_id
    )
    batches = []
    n_nonpad = 0
    n_nonmask = 0
    for b in iterator:
        assert b.x.shape[0] == batch_size
        assert b.x.shape[1] == seq_len
        n_nonpad += (b.x != pad_id).sum()
        if b.mask is None:
            n_nonmask += b.x.size
        else:
            n_nonmask += b.mask.sum()
        batches.append(b)
    assert len(batches) == 1
    assert n_nonpad == n_nonmask == seq_len * n_seqs
    # The second half of the last batch should be all pads
    assert batches[0].mask[10:].sum() == 0


def test_exact_batch_correctness_bytes():
    seq_len = 1024
    n_seqs = 12
    batch_size = 4
    pad_id = 0
    iterator = create_bytes_iter(
        seq_len=seq_len, n_seqs=n_seqs, batch_size=batch_size, pad_id=pad_id
    )
    batches = []
    n_nonpad = 0
    n_nonmask = 0
    for b in iterator:
        assert b.x.shape[0] == batch_size
        assert b.x.shape[1] == seq_len
        n_nonpad += (b.x != pad_id).sum()
        if b.mask is None:
            n_nonmask += b.x.size
        else:
            n_nonmask += b.mask.sum()
        batches.append(b)
    assert len(batches) == 4
    assert n_nonpad == n_nonmask == seq_len * n_seqs


def test_exact_batch_correctness_patches():
    # First patch length is forced to be 1
    patch_lengths = [1, 255, 256, 256, 256]
    # Recall: This is in terms of bytes
    max_length = 1024
    # Recall: This is in terms of patches
    seq_len = 5
    n_seqs = 12
    batch_size = 4
    pad_id = 0
    iterator = create_patches_iter(
        seq_len=seq_len,
        n_seqs=n_seqs,
        batch_size=batch_size,
        pad_id=pad_id,
        patch_lengths=patch_lengths,
        max_length=max_length,
    )
    batches = []
    n_nonpad = 0
    n_nonmask = 0
    for batch in iterator:
        assert batch.x.shape[0] == batch_size
        assert batch.x.shape[1] == max_length
        n_nonpad += (batch.x != pad_id).sum()
        if batch.mask is None:
            n_nonmask += batch.x.size
        else:
            n_nonmask += batch.mask.sum()
        batches.append(batch)

    assert len(batches) == 3

    # max_length - 1 is due to chopping off the last byte for
    # having a y target
    assert n_nonpad == n_nonmask == (max_length - 1) * n_seqs


def test_short_batch_correctness_patches():
    # First patch length is forced to be 1
    # Total=48
    patch_lengths = [1, 11, 12, 12, 12]
    # Recall: This is in terms of bytes
    max_length = 1024
    # Recall: This is in terms of patches
    seq_len = 5
    n_seqs = 12
    batch_size = 4
    pad_id = 0
    iterator = create_patches_iter(
        seq_len=seq_len,
        n_seqs=n_seqs,
        batch_size=batch_size,
        pad_id=pad_id,
        patch_lengths=patch_lengths,
        max_length=max_length,
    )
    batches = []
    n_nonpad = 0
    n_nonmask = 0
    for batch in iterator:
        assert batch.x.shape[0] == batch_size
        assert batch.x.shape[1] == max_length
        n_nonpad += (batch.x != pad_id).sum()
        if batch.mask is None:
            n_nonmask += batch.x.size
        else:
            n_nonmask += batch.mask.sum()
        batches.append(batch)

    assert len(batches) == 3

    # We'll still always have one byte chopped off the end
    assert n_nonpad == n_nonmask == ((sum(patch_lengths) - 1) * n_seqs)


def test_long_batch_correctness_patches():
    # First patch length is forced to be 1
    # Total=48
    patch_lengths = [1, 255, 256, 256, 1024]
    # Recall: This is in terms of bytes
    max_length = 1024
    # Recall: This is in terms of patches
    seq_len = 5
    n_seqs = 12
    batch_size = 4
    pad_id = 0
    iterator = create_patches_iter(
        seq_len=seq_len,
        n_seqs=n_seqs,
        batch_size=batch_size,
        pad_id=pad_id,
        patch_lengths=patch_lengths,
        max_length=max_length,
    )
    batches = []
    n_nonpad = 0
    n_nonmask = 0
    for batch in iterator:
        assert batch.x.shape[0] == batch_size
        assert batch.x.shape[1] == max_length
        n_nonpad += (batch.x != pad_id).sum()
        if batch.mask is None:
            n_nonmask += batch.x.size
        else:
            n_nonmask += batch.mask.sum()
        batches.append(batch)

    assert len(batches) == 3

    # No chop here since the next byte is available
    assert n_nonpad == n_nonmask == max_length * n_seqs


def test_merge_patch_seq_masks():
    batch_size = 4
    seq_len = 1024
    masks = []
    masks.append([True] * 1025)
    masks.append([True] * 512)
    masks.append([True] * 256)
    masks.append([True] * 10)
    expected_mask = np.zeros((batch_size, seq_len), dtype=bool)
    expected_mask[0, :] = True
    expected_mask[1, :511] = True
    expected_mask[2, :255] = True
    expected_mask[3, :9] = True
    merged_mask = _merge_patch_seq_masks(batch_size, seq_len, masks)
    assert (merged_mask == expected_mask).all()

    with pytest.raises(AssertionError):
        masks = []
        masks.append([True] * 1024)
        masks.append([True] * 512)
        masks.append([True] * 256)
        masks.append([True] * 10)
        _merge_patch_seq_masks(batch_size, seq_len, masks)
