import numpy as np
from typing import Callable, Self


class Node:
    def __init__(
        self,
        value: np.ndarray,
        _parents: tuple[Self, ...] | None = None,
        _backward: Callable[[], tuple[np.ndarray, ...]] | None = None,
    ):
        self.value = value
        self.grad = 0

        # Implementation detials for backpropogation - _backward produces node wise gradients per _parent
        self._parents = tuple() if _parents is None else _parents
        self._backward = lambda: (np.array(0.0),) if _backward is None else _backward

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
        # Send gradients back in topological order,
        # such that all children send gradients back before parent is processed
        topo_sort = self.get_topo_sort()
        for u in topo_sort:
            for v, node_grad in zip(u._parents, u._backward()):
                v.grad += u.grad * node_grad
