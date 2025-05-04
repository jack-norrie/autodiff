from src.primatives import Function, Node
import numpy as np


class Add(Function):
    @staticmethod
    def forward(x: Node, y: Node) -> Node:
        z = Node(x.value + y.value)
        return z

    @staticmethod
    def backward() -> tuple[np.ndarray, ...]:
        return (np.array(1.0), np.array(1.0))


add = Add()
