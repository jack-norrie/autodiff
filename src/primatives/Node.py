import numpy as np
from typing import Self


class Node:
    def __init__(self, value: np.ndarray, _parents: None | tuple[Self, ...] = None):
        self.value = value
        self.grad = 0
        self._parents = tuple() if _parents is None else _parents

    def __repr__(self) -> str:
        return f"Node({repr(self.value)})"

    def backward(self):
        topo: list[Node] = []
        seen: set[Node] = set()

        def get_topological_sort(v: Node) -> None:
            seen.add(v)
            for parent in v._parents:
                if parent not in seen:
                    get_topological_sort(parent)
            topo.append(v)

        get_topological_sort(self)

        print(topo)
