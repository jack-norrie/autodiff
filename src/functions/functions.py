from math import prod
from src.auto.Function import Function
from src.auto.Vertex import Vertex


class Add(Function):
    @staticmethod
    def forward(*args) -> Vertex:
        z = Vertex(sum(v.value for v in args))
        return z

    @staticmethod
    def backward(*args) -> tuple[float, ...]:
        return (1.0,) * len(args)


add = Add()


class Mult(Function):
    @staticmethod
    def forward(*args) -> Vertex:
        z = Vertex(prod(v.value for v in args))
        return z

    @staticmethod
    def backward(*args) -> tuple[float, ...]:
        n = len(args)

        pre = [1 for _ in range(n)]
        for i in range(1, n):
            pre[i] = args[i - 1].value * pre[i - 1]

        post = [1 for _ in range(n)]
        for i in range(n - 2, -1, -1):
            post[i] = args[i + 1].value * post[i + 1]

        return tuple(pre[i] * post[i] for i in range(n))


mult = Mult()


# def mat_mul(A: Matrix, B: Matrix | Vector) -> Matrix | Vector:
#     if isinstance(B[0], Vertex):
#         B = typing.cast(Vector, B)
#         B = [[v] for v in B]
#     B = typing.cast(Matrix, B)
#
#     assert len(A[0]) == len(B), f"Incompatible shapes: {len(A[0])} and {len(B)}"
#     m = len(A)
#     n = len(B[0])
#
#     C = [[None for _ in range(n)] for _ in range(m)]
#     for i in range(m):
#         for j in range(n):
#             z = mul(A[i][0], B[0][j])
#             for k in range(1, len(B)):
#                 z = add(z, mul(A[i][k], B[k][j]))
#             C[i][j] = z
#
#     if len(B[0]) == 1:
#         C = [row[0] for row in C]
#         C = typing.cast(Vector, C)
#     else:
#         C = typing.cast(Matrix, C)
#     return C
#
#
