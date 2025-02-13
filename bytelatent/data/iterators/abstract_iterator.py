# Copyright (c) Meta Platforms, Inc. and affiliates.
import abc
from typing import Any, Generator, Generic, TypeVar

T = TypeVar("T")
C = TypeVar("C")


class StatefulIterator(Generic[T, C], abc.ABC):

    @abc.abstractmethod
    def get_state(self) -> C:
        pass

    @abc.abstractmethod
    def create_iter(self) -> Generator[T, Any, None]:
        pass


class IteratorState(Generic[C]):
    @abc.abstractmethod
    def build(self) -> StatefulIterator[T, C]:
        pass


def get_state_and_refresh(iterator: StatefulIterator):
    # Re-init dataloader and iterator is necessary since get_state()
    # on mp iterator shuts down MP to correctly persist state and it needs
    # to be restarted.
    state = iterator.get_state()
    data_loader = state.build()
    py_iterator = data_loader.create_iter()
    return state, data_loader, py_iterator
