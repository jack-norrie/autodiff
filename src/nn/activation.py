from src.auto import Function, Vertex


class ReLU(Function):
    def forward(self, x: Vertex) -> Vertex:
        return Vertex(max(0, x.value))

    def backward(self, x: Vertex) -> tuple[float]:
        return (1.0 if x.value >= 0.0 else 0.0,)


relu = ReLU()
