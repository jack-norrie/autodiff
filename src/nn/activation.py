from src.functions import Function
from src.primatives import Vertex


class ReLU(Function):
    def forward(x: Vertex) -> Vertex:
        return Vertex(max(0, x.value))

    def backward(x: Vertex) -> tuple[float]:
        return Vertex(1.0 if x.value >= 0.0 else 0.0)


relu = ReLU()
