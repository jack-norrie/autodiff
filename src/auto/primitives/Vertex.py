import math
from typing import Callable, Self

from src.auto.Function import Function


class Vertex:
    def __init__(
        self,
        value: float,
        _parents: tuple[Self, ...] | None = None,
        _backward: Callable[[tuple[Self, ...]], tuple[float, ...]] | None = None,
    ):
        self.value = value
        self.grad = 0

        # Implementation detials for backpropogation - _backward produces node wise gradients per _parent
        self._parents = tuple() if _parents is None else _parents
        self._backward = lambda *n: (0,) if _backward is None else _backward

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(self.value)})"

    def _get_topo_sort(self, topo_sort: list[Self], seen: set[Self]) -> None:
        seen.add(self)
        for parent in self._parents:
            if parent not in seen:
                parent._get_topo_sort(topo_sort=topo_sort, seen=seen)
        topo_sort.append(self)

    def get_topo_sort(self):
        topo_sort: list[Self] = []
        seen: set[Self] = set()

        self._get_topo_sort(topo_sort, seen)

        return topo_sort[::-1]

    def backward(self):
        # Set the top level node gradient to one, i.e. its gradient with respect to itself
        self.grad = 1

        # Send gradients back in topological order,
        # such that all children send gradients back before parent is processed
        topo_sort = self.get_topo_sort()
        for u in topo_sort:
            node_grads = u._backward(*u._parents)
            for v, node_grad in zip(u._parents, node_grads):
                v.grad += u.grad * node_grad

    def zero_grad(self):
        seen = set()

        def dfs(root: Self):
            root.grad = 0
            seen.add(root)

            for parent in root._parents:
                if parent not in seen:
                    dfs(parent)

        dfs(self)

    def __add__(self, other: Self) -> Self:
        return add(self, other)

    def __sub__(self, other: Self) -> Self:
        return sub(self, other)

    def __mul__(self, other: Self) -> Self:
        return mul(self, other)

    def __truediv__(self, other: Self) -> Self:
        return div(self, other)

    def __pow__(self, other: Self) -> Self:
        return pow(self, other)

    def __neg__(self) -> Self:
        return neg(self)


class Add(Function):
    @staticmethod
    def forward(x: Vertex, y: Vertex) -> Vertex:
        z = Vertex(x.value + y.value)
        return z

    @staticmethod
    def backward(x: Vertex, y: Vertex) -> tuple[float, float]:
        return (1.0, 1.0)


add = Add()


class Sub(Function):
    @staticmethod
    def forward(x: Vertex, y: Vertex) -> Vertex:
        z = Vertex(x.value - y.value)
        return z

    @staticmethod
    def backward(x: Vertex, y: Vertex) -> tuple[float, float]:
        return (1.0, -1.0)


sub = Sub()


class Mul(Function):
    @staticmethod
    def forward(x: Vertex, y: Vertex) -> Vertex:
        z = Vertex(x.value * y.value)
        return z

    @staticmethod
    def backward(x: Vertex, y: Vertex) -> tuple[float, float]:
        return (y.value, x.value)


mul = Mul()


class Div(Function):
    @staticmethod
    def forward(x: Vertex, y: Vertex) -> Vertex:
        z = Vertex(x.value / y.value)
        return z

    @staticmethod
    def backward(x: Vertex, y: Vertex) -> tuple[float, float]:
        return (1 / y.value, -x.value / (y.value**2))


div = Div()


class Neg(Function):
    @staticmethod
    def forward(x: Vertex) -> Vertex:
        return Vertex(-x.value)

    @staticmethod
    def backward(x: Vertex) -> tuple[float]:
        return (-1.0,)


neg = Neg()
