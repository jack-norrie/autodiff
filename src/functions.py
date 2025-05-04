from src.primatives import Function, Node
import numpy as np


class Add(Function):
    @staticmethod
    def forward(x: Node, y: Node) -> Node:
        z = Node(x.value + y.value)
        return z

    @staticmethod
    def backward(x: Node, y: Node) -> tuple[np.ndarray, ...]:
        return (np.array(1.0), np.array(1.0))


add = Add()


class Mul(Function):
    @staticmethod
    def forward(x: Node, y: Node) -> Node:
        z = Node(x.value * y.value)
        return z

    @staticmethod
    def backward(x: Node, y: Node) -> tuple[np.ndarray, ...]:
        return (x.value, x.value)


mul = Mul()
