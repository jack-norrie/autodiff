import numpy as np


class Node:
    def __init__(self, value: np.ndarray, _children: None | list = None):
        self.value = value
        self.grad = 0
        self._children = [] if _children is None else _children

    def __repr__(self) -> str:
        return f"Node({repr(self.value)})"
