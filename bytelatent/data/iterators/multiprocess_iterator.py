# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
import logging
import multiprocessing as mp
from multiprocessing.synchronize import Event as EventClass
from queue import Empty, Full

import numpy as np
from pydantic import BaseModel, ConfigDict

from bytelatent.data.data_types import Batch
from bytelatent.data.iterators.abstract_iterator import IteratorState, StatefulIterator
from bytelatent.data.iterators.packing_iterator import PackingIteratorState

logger = logging.getLogger()


class MultiprocessIteratorState(BaseModel, IteratorState):
    model_config = ConfigDict(extra="forbid")
    base_iterator_state: PackingIteratorState
    n_batches_to_prefetch: int
    serialized_prefetch_buffer: str

    def build(self):
        base_iterator = self.base_iterator_state.build()
        data = json.loads(self.serialized_prefetch_buffer)
        prefetch_buffer = [Batch.from_python_dict(item) for item in data]
        return MultiprocessIterator(
            base_iterator,
            n_batches_to_prefetch=self.n_batches_to_prefetch,
            prefetch_buffer=prefetch_buffer,
        )


def start_work_from_state(
    batch_queue: mp.Queue,
    state_queue: mp.Queue,
    stop_event: EventClass,
    state_dumped_event: EventClass,
    state: IteratorState,
):
    logging.info("Worker thread: Starting base_iterator work")
    stateful_iterator = state.build()
    iterator = stateful_iterator.create_iter()
    for item in iterator:
        while not stop_event.is_set():
            try:
                # Attempt to put on queue or timeout to try again (maybe main thread is busy)
                batch_queue.put(item, timeout=0.1)
                # On success, stop trying
                break
            except Full:
                pass
        if stop_event.is_set():
            # Signal the end of output, this ensures that even if the queue takes a while to
            # buffer, that the main thread receives everything (and tosses this fake batch)
            logging.debug(
                "Worker thread: Stop event detected, outputting is_final=True batch"
            )
            logging.debug("Worker thread: batch_queue full=%s", batch_queue.full())
            batch_queue.put(
                Batch(
                    x=np.zeros((1, 1)),
                    y=np.zeros((1, 1)),
                    is_final=True,
                    mask=None,
                    patch_lengths=None,
                    ngram_ids=None,
                )
            )
            logging.debug(
                "Worker thread: is_final=True batch put in queue, breaking from loop."
            )
            break

    try:
        logging.debug("Worker thread: outputting state")
        state_queue.put(stateful_iterator.get_state(), timeout=1)
        logging.debug("Worker thread: state dump complete")
        state_dumped_event.set()
        logging.debug("Worker thread: set state_dump_event")
    except Full:
        raise ValueError(
            "Attempted to dump state into the state queue, but it was full"
        )


class MultiprocessIterator(StatefulIterator):
    """
    Design sketch of the multiprocess iterator:

    Given the base_iterator, the only thing we do with this is call get_state()
    so that we can pass that through to the background worker process.

    The background process will receive this, rebuild the iterator, then start yielding from it.

    However, in order to implement MultiprocessIterator.get_state(), we need to be able to accurately get
    (1) the state of the iterator in the worker process
    (2) the currently buffered items in the Queue

    To do this, we use:
    - batch_queue: This is the prefetch buffer the worker yields to and the main loop yields from
    - state_queue: This size 1 queue will be how the worker sends the iterator state once it has halted iterating.
        It must hold the state in addition to the last batch, if the queue was full at the time the stop event is sent.
    - stop_iterating_event: Once this is issued from the main loop, the worker will stop iterating and enter cleanup.
        During cleanup, the iterator will send the state of the current iterator to the main loop,
        in addition to possibly the last batch if the batch_queue was full at the time
    - state_dumped_event: When the main loop issues the stop_iterating_event, it will wait until the state_dumped_event to attempt
        to get state from the state_queue. It must do this since the worker may take some time to create and send the state.
        Once received by the main loop, the main loop can safely store the Queue (plus maybe the last batch) as the prefetch buffer,
        get the worker iterator's state, and terminate the background process + delete associated objects.

    At this point, calling create_iter() again will bootstrap everything from the stored state and the old iterator will throw an error
    since it will not iterate anymore (so the caller must call create_iter() again to get a python iterator).

    """

    def __init__(
        self,
        base_iterator: StatefulIterator,
        *,
        n_batches_to_prefetch: int,
        prefetch_buffer: list | None = None
    ):
        self.base_iterator = base_iterator
        self.n_batches_to_prefetch = n_batches_to_prefetch
        if prefetch_buffer is None:
            prefetch_buffer = []
        self.prefetch_buffer = prefetch_buffer
        self.batch_queue = None
        self.state_queue = None
        self.producer = None
        self.stop_iterating_event = None
        self.state_dumped_event = None
        self.force_shutdown = False

    def shutdown(self):
        if self.producer is not None:
            # This properly shuts things down
            self.producer.kill()
            self.force_shutdown = True

    def get_state(self) -> MultiprocessIteratorState:
        """
        This is slightly unusual in effectively destroying the current iterator, its necessary
        to halt the background process and allow it to write the state to the main loop
        in order to not lose data
        """
        if self.force_shutdown:
            raise ValueError(
                "State will be invalid if shutdown was forced before state persisted."
            )
        if self.producer is None:
            serialized_prefetch_buffer = json.dumps(
                [b.to_python_dict() for b in self.prefetch_buffer]
            )
            return MultiprocessIteratorState(
                base_iterator_state=self.base_iterator.get_state(),
                n_batches_to_prefetch=self.n_batches_to_prefetch,
                serialized_prefetch_buffer=serialized_prefetch_buffer,
            )
        else:
            logging.debug("Main thread: Sending stop iteration event")
            self.stop_iterating_event.set()
            logging.debug(
                "Main thread: Emptying the batch_queue until batch.is_final=True is found."
            )
            self.prefetch_buffer = []
            final_batch_received = False
            while True:
                try:
                    batch = self.batch_queue.get(timeout=1)
                    if batch.is_final:
                        logging.debug(
                            "Main thread: is_final=True batch found, stopping fetch from batch_queue"
                        )
                        final_batch_received = True
                        break
                    self.prefetch_buffer.append(batch)
                except Empty:
                    logging.warning("Main thread: batch_queue is abnormally empty")
            assert final_batch_received

            logging.debug("Main thread: Waiting for state_dumped event")
            self.state_dumped_event.wait()

            try:
                base_iterator_state = self.state_queue.get(timeout=1)
                assert isinstance(base_iterator_state, IteratorState)
            except Empty:
                raise ValueError(
                    "Attempted to get the state, but it was unexpectantly missing"
                )

            self.base_iterator = base_iterator_state.build()
            self.producer.close()
            self.producer = None
            self.batch_queue = None
            self.state_queue = None
            self.stop_iterating_event = None
            self.state_dumped_event = None

            return MultiprocessIteratorState(
                base_iterator_state=self.base_iterator.get_state(),
                n_batches_to_prefetch=self.n_batches_to_prefetch,
                serialized_prefetch_buffer=json.dumps(
                    [b.to_python_dict() for b in self.prefetch_buffer]
                ),
            )

    def create_iter(self):
        if self.force_shutdown:
            raise ValueError(
                "Iterator may be invalid if shutdown was forced before state persisted."
            )
        logging.info("Main thread: Creating MP iterator")
        # First yield from the stored prefetch buffer.
        if self.prefetch_buffer is not None:
            while len(self.prefetch_buffer) > 0:
                item = self.prefetch_buffer.pop(0)
                yield item
            self.prefetch_buffer = None

        assert (
            self.producer is None
        ), "Cannot create two parallel iterators at once, call get_state() then remake to have two."

        # using mp context manager avoids excessive CPU loading
        ctx = mp.get_context("forkserver")
        self.batch_queue = ctx.Manager().Queue(maxsize=self.n_batches_to_prefetch)

        # We should only ever one state, which is output at the detection of a stop event
        self.state_queue = ctx.Manager().Queue(maxsize=1)

        self.stop_iterating_event = ctx.Event()
        self.state_dumped_event = ctx.Event()

        self.producer = mp.Process(
            name="blt_data_loader",
            target=start_work_from_state,
            args=(
                self.batch_queue,
                self.state_queue,
                self.stop_iterating_event,
                self.state_dumped_event,
                self.base_iterator.get_state(),
            ),
        )
        logger.info("Async dataloader started")
        self.producer.start()

        while True:
            if self.producer.exitcode is not None:
                raise RuntimeError(
                    "Data loader quit unexpectedly, real error has been raised previously"
                )
            try:
                batch = self.batch_queue.get(timeout=0.1)
                assert isinstance(batch, Batch)
                assert (
                    not batch.is_final
                ), "is_final should only be used during get_state() being called"
                yield batch
            except Empty:
                pass
            if self.producer is None:
                raise ValueError(
                    "Attempted to call this iterator after calling get_state(). You must call create_iter() to make a new iterator instead."
                )
