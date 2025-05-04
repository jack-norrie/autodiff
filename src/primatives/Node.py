import numpy as np
from typing import Callable, Self


class Node:
    def __init__(
        self,
        value: np.ndarray,
        _parents: tuple[Self, ...] | None = None,
        _backward: Callable[..., Self] | None = None,
    ):
        self.value = value
        self.grad = 0

        self._parents = tuple() if _parents is None else _parents
        self._backward = lambda: 0 if _backward is None else _backward

    def __repr__(self) -> str:
        return f"Node({repr(self.value)})"

    def _get_topo_sort(self, topo_sort: list[Self], seen: set[Self]) -> None:
        seen.add(self)
        for parent in self._parents:
            if parent not in seen:
                parent._get_topo_sort(topo_sort=topo_sort, seen=seen)
        topo_sort.append(self)

    def get_topo_sort(self):
        topo_sort: list[Self] = []
        seen: set[Self] = set()

        self._get_topo_sort(topo_sort, seen)

        return topo_sort[::-1]

    def backward(self):
        topo_sort = self.get_topo_sort()

        for v in topo_sort:
            v._backward()
