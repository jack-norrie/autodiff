from src.primatives import Function, Vertex


class Add(Function):
    @staticmethod
    def forward(x: Vertex, y: Vertex) -> Vertex:
        z = Vertex(x.value + y.value)
        return z

    @staticmethod
    def backward(x: Vertex, y: Vertex) -> tuple[float, float]:
        return (1.0, 1.0)


add = Add()


class Mul(Function):
    @staticmethod
    def forward(x: Vertex, y: Vertex) -> Vertex:
        z = Vertex(x.value * y.value)
        return z

    @staticmethod
    def backward(x: Vertex, y: Vertex) -> tuple[float, float]:
        return (y.value, x.value)


mul = Mul()


class Square(Function):
    @staticmethod
    def forward(x: Vertex) -> Vertex:
        return Vertex(x.value**2)

    @staticmethod
    def backward(x: Vertex) -> tuple[float]:
        return (2 * x.value,)


square = Square()
