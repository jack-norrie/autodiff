from collections.abc import Sequence
import typing
from operator import add, sub, mul, truediv
from typing import Callable
from typing import Self
import src.functions.functions as F

from src.auto import Vertex


class Vector(Sequence):
    def __init__(self, *args):
        if len(args) == 0:
            raise ValueError("No data supplied.")

        if isinstance(args[0], Sequence):
            assert len(args) == 1, (
                "Constructor either takes verticies via variadic arguments, or an iterator."
            )
            parsed_args = list(args[0])
        else:
            parsed_args = list(args)

        for i in range(len(parsed_args)):
            if isinstance(parsed_args[i], Vertex):
                continue
            elif isinstance(parsed_args[i], float):
                parsed_args[i] = Vertex(parsed_args[i])
            else:
                raise ValueError(
                    "All passed arguments must be either of type Vertex or float."
                )

        self._data: tuple[Vertex, ...] = tuple(parsed_args)

    def __getitem__(self, item: int) -> Vertex:
        return self._data[item]

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data})"

    def _element_wise_operation(
        self, other: Self | float | int, op: Callable[[Vertex, Vertex], Vertex]
    ) -> Self:
        if isinstance(other, float) or isinstance(other, int):
            other = type(self)([Vertex(other)] * len(self))  # n references
        other = typing.cast(Self, other)

        if len(self) != len(other):
            raise ValueError(
                f"Vectors must be same length: {len(self)} != {len(other)}"
            )
        n = len(self)

        out = []
        for i in range(n):
            out.append(op(self[i], other[i]))

        return type(self)(out)

    def __add__(self, other: Self | float | int) -> Self:
        return self._element_wise_operation(other, add)

    def __radd__(self, other: float) -> Self:
        return self._element_wise_operation(other, add)

    def __sub__(self, other: Self | float | int) -> Self:
        return self._element_wise_operation(other, sub)

    def __rsub__(self, other: float) -> Self:
        return type(self)([Vertex(other) - v for v in self])

    def __mul__(self, other: Self | float | int) -> Self:
        return self._element_wise_operation(other, mul)

    def __rmul__(self, other: Self | float | int) -> Self:
        return self._element_wise_operation(other, mul)

    def __truediv__(self, other: Self | float | int) -> Self:
        return self._element_wise_operation(other, truediv)

    def __rtruediv__(self, other: float) -> Self:
        return type(self)([Vertex(other) / v for v in self])

    def __neg__(self) -> Self:
        return type(self)([-v for v in self._data])

    def dot(self, other: Self) -> Vertex:
        return F.add(*(self * other))
