import math
from src.auto import Function, Vertex


class Sigmoid(Function):
    @staticmethod
    def forward(x: Vertex) -> Vertex:
        return Vertex(1.0 / (1.0 + math.exp(-x.value)))

    @staticmethod
    def backward(x: Vertex) -> tuple[float]:
        sig_x = 1.0 / (1.0 + math.exp(-x.value))
        return (sig_x * (1.0 - sig_x),)


sigmoid = Sigmoid()


class Tanh(Function):
    @staticmethod
    def forward(x: Vertex) -> Vertex:
        return Vertex(math.tanh(x.value))

    @staticmethod
    def backward(x: Vertex) -> tuple[float]:
        tanh_x = math.tanh(x.value)
        return (1.0 - tanh_x * tanh_x,)


tanh = Tanh()


class ReLU(Function):
    @staticmethod
    def forward(x: Vertex) -> Vertex:
        return Vertex(max(0, x.value))

    @staticmethod
    def backward(x: Vertex) -> tuple[float]:
        return (1.0 if x.value >= 0.0 else 0.0,)


relu = ReLU()
