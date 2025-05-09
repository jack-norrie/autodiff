import math
from typing import Callable, Self

from src.auto.Function import Function


class Vertex:
    """
    The core class for automatic differentiation in the computational graph.

    A Vertex represents a node in the computational graph, storing both the value
    and gradient information. It supports automatic differentiation through the
    backward method and provides operator overloading for arithmetic operations.
    """

    def __init__(
        self,
        value: float,
        _parents: tuple[Self, ...] | None = None,
        _backward: Callable[[tuple[Self, ...]], tuple[float, ...]] | None = None,
    ):
        """
        Initialize a Vertex with a value and optional parent nodes.

        Args:
            value: The scalar value of this vertex
            _parents: Parent vertices in the computational graph
            _backward: Function to compute gradients with respect to parents
        """
        self.value = value
        self.grad = 0

        # Implementation detials for backpropogation - _backward produces node wise gradients per _parent
        self._parents = tuple() if _parents is None else _parents
        self._backward = lambda *n: (0,) if _backward is None else _backward

    def __repr__(self) -> str:
        """
        Return a string representation of the Vertex.

        Returns:
            str: String representation including the class name and value
        """
        return f"{type(self).__name__}({repr(self.value)})"

    def _get_topo_sort(self, topo_sort: list[Self], seen: set[Self]) -> None:
        """
        Helper method for topological sorting of the computational graph.

        Performs a depth-first search to build a topological ordering.

        Args:
            topo_sort: List to store the topological ordering
            seen: Set of already visited vertices
        """
        seen.add(self)
        for parent in self._parents:
            if parent not in seen:
                parent._get_topo_sort(topo_sort=topo_sort, seen=seen)
        topo_sort.append(self)

    def get_topo_sort(self):
        """
        Get a topological sorting of the computational graph.

        Returns:
            list[Self]: Vertices in topological order (reversed for backpropagation)
        """
        topo_sort: list[Self] = []
        seen: set[Self] = set()

        self._get_topo_sort(topo_sort, seen)

        return topo_sort[::-1]

    def backward(self):
        """
        Perform backpropagation to compute gradients through the computational graph.

        This method:
        1. Sets the gradient of this vertex to 1 (seed gradient)
        2. Traverses the computational graph in topological order
        3. Computes and accumulates gradients for all parent vertices
        """
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
        """
        Reset all gradients in the computational graph to zero.

        This method performs a depth-first traversal of the computational graph
        and sets the gradient of each vertex to zero.
        """
        seen = set()

        def dfs(root: Self):
            root.grad = 0
            seen.add(root)

            for parent in root._parents:
                if parent not in seen:
                    dfs(parent)

        dfs(self)

    def _parse_other(self, other: Self | float | int) -> Self:
        if isinstance(other, float) or isinstance(other, int):
            other = type(self)(other)
        return other

    def __add__(self, other: Self | float | int) -> Self:
        other = self._parse_other(other)
        return add(self, other)

    def __sub__(self, other: Self | float | int) -> Self:
        other = self._parse_other(other)
        return sub(self, other)

    def __mul__(self, other: Self | float | int) -> Self:
        other = self._parse_other(other)
        return mul(self, other)

    def __truediv__(self, other: Self | float | int) -> Self:
        other = self._parse_other(other)
        return div(self, other)

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
