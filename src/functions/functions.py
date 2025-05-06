import typing

from src.Function import Function
from src.primitives import Vertex, Matrix, Vector


def vec_dot(u: Vector, v: Vector) -> Vertex:
    assert len(u) == len(v), f"Incompatible shapes: {len(u)} and {len(v)}"
    n = len(u)

    w = mul(u[0], v[0])
    for i in range(1, n):
        w = add(w, mul(u[i], v[i]))

    return w


def mat_mul(A: Matrix, B: Matrix | Vector) -> Matrix | Vector:
    if isinstance(B[0], Vertex):
        B = typing.cast(Vector, B)
        B = [[v] for v in B]
    B = typing.cast(Matrix, B)

    assert len(A[0]) == len(B), f"Incompatible shapes: {len(A[0])} and {len(B)}"
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
        C = [row[0] for row in C]
        C = typing.cast(Vector, C)
    else:
        C = typing.cast(Matrix, C)
    return C


def vec_add(u: Vector, v: Vector) -> Vector:
    assert len(u) == len(v), f"Incompatible shapes: {len(u)} and {len(v)}"
    n = len(u)

    w = [None for _ in range(n)]
    for i in range(n):
        w[i] = add(u[i], v[i])
    w = typing.cast(Vector, w)

    return w
