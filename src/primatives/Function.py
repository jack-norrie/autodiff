from abc import ABC, abstractmethod
from src.primatives.Node import Node
import numpy as np


class Function(ABC):
    @classmethod
    def __call__(cls, *args: tuple[Node]) -> Node:
        z = cls.forward(*args)

        # Add paranets and backwards function for backprop
        z._parents = args
        z._backward = cls.backward

        return z

    @staticmethod
    @abstractmethod
    def forward(*args: tuple[Node, ...]) -> Node:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def backward() -> tuple[np.ndarray, ...]:
        raise NotImplementedError
