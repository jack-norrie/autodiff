import math
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


class Square(Function):
    @staticmethod
    def forward(v: Vertex) -> Vertex:
        z = Vertex(v.value**2)
        return z

    @staticmethod
    def backward(v: Vertex) -> tuple[float, ...]:
        return (2 * v.value,)


square = Square()


class Sin(Function):
    @staticmethod
    def forward(v: Vertex) -> Vertex:
        z = Vertex(math.sin(v.value))
        return z

    @staticmethod
    def backward(v: Vertex) -> tuple[float, ...]:
        return (math.cos(v.value),)


sin = Sin()


class Cos(Function):
    @staticmethod
    def forward(v: Vertex) -> Vertex:
        z = Vertex(math.cos(v.value))
        return z

    @staticmethod
    def backward(v: Vertex) -> tuple[float, ...]:
        return (-math.sin(v.value),)


cos = Cos()


class Tan(Function):
    @staticmethod
    def forward(v: Vertex) -> Vertex:
        z = Vertex(math.tan(v.value))
        return z

    @staticmethod
    def backward(v: Vertex) -> tuple[float, ...]:
        return (1 / (math.cos(v.value) ** 2),)


tan = Tan()


class Exp(Function):
    @staticmethod
    def forward(v: Vertex) -> Vertex:
        z = Vertex(math.exp(v.value))
        return z

    @staticmethod
    def backward(v: Vertex) -> tuple[float, ...]:
        return (math.exp(v.value),)


exp = Exp()


class Log(Function):
    @staticmethod
    def forward(v: Vertex) -> Vertex:
        z = Vertex(math.log(v.value))
        return z

    @staticmethod
    def backward(v: Vertex) -> tuple[float, ...]:
        return (1 / v.value,)


log = Log()
