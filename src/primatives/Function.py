from abc import ABC, abstractmethod
from src.primatives.Node import Node


class Function(ABC):
    @classmethod
    def __call__(cls, *args: tuple[Node, ...]) -> Node:
        return cls.forward(*args)

    @staticmethod
    @abstractmethod
    def forward(*args: tuple[Node, ...]) -> Node:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def backward(*args: tuple[Node, ...]) -> Node:
        raise NotImplementedError
