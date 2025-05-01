import numpy as np


class Node:
    def __init__(self, value: np.ndarray, _depdencies: None | list = None):
        self.value = value
        self.grad = 0
        self._depdencies = [] if _depdencies is None else _depdencies

    def __repr__(self) -> str:
        return f"Node({repr(self.value)})"
