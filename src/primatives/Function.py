from abc import ABC, abstractmethod
from src.primatives.Node import Node


class Function(ABC):
    def __init__(self):
        self.state = {}

    @abstractmethod
    def forward(self, *args: list[Node]) -> Node:
        raise NotImplementedError

    @abstractmethod
    def backward(self, *args: list[Node]) -> Node:
        raise NotImplementedError
