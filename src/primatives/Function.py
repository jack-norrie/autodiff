from abc import ABC, abstractmethod
from src.primatives.Node import Node


class Function(ABC):
    @classmethod
    def __call__(cls, *args) -> Node:
        return cls.forward(*args)

    @staticmethod
    @abstractmethod
    def forward(*args) -> Node:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def backward(*args) -> Node:
        raise NotImplementedError
