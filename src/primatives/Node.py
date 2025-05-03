import numpy as np
from typing import Self


class Node:
    def __init__(self, value: np.ndarray, _parents: None | tuple[Self, ...] = None):
        self.value = value
        self.grad = 0
        self._parents = tuple() if _parents is None else _parents

    def __repr__(self) -> str:
        return f"Node({repr(self.value)})"
