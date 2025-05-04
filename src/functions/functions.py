import typing

from src.functions.functions import Function
from src.primatives import Vertex, Matrix, Vector


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


def mat_mul(A: Matrix, B: Matrix | Vector) -> Matrix | Vector:
    if isinstance(B[0], Vertex):
        B = typing.cast(Vector, B)
        B = [[v] for v in B]
    B = typing.cast(Matrix, B)

    assert len(A[0]) == len(B), "Incompatible shapes"
    m = len(A)
    n = len(B[0])

    C = [[None for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            z = mul(A[i][0], B[0][j])
            for k in range(1, len(B)):
                z = add(z, mul(A[i][k], B[k][j]))
            C[i][j] = z

    if len(B[0]) == 1:
        C = [row[0] for row in Matrix]
        C = typing.cast(Vector, C)
    else:
        C = typing.cast(Matrix, C)
    return C


def vec_add(u: Vector, v: Vector) -> Vector:
    assert len(u) == len(v), "Incompatible shapes"
    n = len(u)

    w = [None for _ in range(n)]
    for i in range(n):
        w[i] = add(u[i], v[i])
    w = typing.cast(Vector, w)

    return w
