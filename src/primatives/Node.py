import numpy as np
from typing import Self


class Node:
    def __init__(self, value: np.ndarray, _parents: None | tuple[Self, ...] = None):
        self.value = value
        self.grad = 0
        self._parents = tuple() if _parents is None else _parents

    def __repr__(self) -> str:
        return f"Node({repr(self.value)})"

    def _get_topo_sort(self, topo: list[Self], seen: set[Self]) -> None:
        seen.add(self)
        for parent in self._parents:
            if parent not in seen:
                parent._get_topo_sort(topo=topo, seen=seen)
        topo.append(self)

    def get_topo_sort(self):
        topo: list[Self] = []
        seen: set[Self] = set()

        self._get_topo_sort(topo, seen)

        return topo[::-1]

    def backward(self):
        topo_sort = self.get_topo_sort()
        print(topo_sort)
