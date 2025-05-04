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
        return (y.value, x.value)


mul = Mul()


class Square(Function):
    @staticmethod
    def forward(x: Node) -> Node:
        return Node(x.value**2)

    @staticmethod
    def backward(x: Node) -> tuple[np.ndarray, ...]:
        return (2 * x.value,)


square = Square()
