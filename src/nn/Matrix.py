import typing
from collections.abc import Sequence
from operator import add, mul, sub, truediv
from typing import Callable, Self, overload

from src.auto import Vertex
from src.nn.Vector import Vector


class Matrix(Sequence):
    def __init__(self, data: Sequence[Sequence[float | Vector]]):
        assert len(data) > 0, "At least one row of data must be supplied."
        m = len(data)

        cols = set()
        for r in range(m):
            cols.add(len(data[r]))
        assert len(cols) == 1, (
            f"Inconsistent column numbers in supplied data, n_cols: {cols}"
        )
        n = cols.pop()

        parsed = [[None for _ in range(n)] for _ in range(m)]
        for r in range(m):
            for c in range(n):
                item = data[r][c]
                if isinstance(item, Vertex):
                    parsed[r][c] = item
                elif isinstance(item, float):
                    parsed[r][c] = Vertex(item)
                else:
                    raise ValueError(
                        "All passed arguments must be either of type Vertex or float."
                    )

        self._rows: tuple[Vector, ...] = tuple(Vector(row) for row in parsed)

    @overload
    def __getitem__(self, key: int) -> Vector: ...

    @overload
    def __getitem__(self, key: tuple[int, int]) -> Vertex: ...

    def __getitem__(self, key: int | tuple[int, int]) -> Vertex | Vector:
        if isinstance(key, tuple):
            assert len(key) == 2, "Tuple keys must have length 2."
            r, c = key
            return self._rows[r][c]
        elif not isinstance(key, float):
            return self._rows[key]
        else:
            raise ValueError("Key must be int or tuple of ints.")

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self._rows), len(self._rows[0])

    def _element_wise_operation(
        self, other: Self | float, op: Callable[[Vector, Vector], Vector]
    ) -> Self:
        if isinstance(other, float):
            n_rows, n_cols = self.shape
            other = type(self)([Vector([(other)] * n_cols)] * n_rows)  # n references
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

    def __add__(self, other: Self | float) -> Self:
        return self._element_wise_operation(other, add)

    def __radd__(self, other: float) -> Self:
        return self._element_wise_operation(other, add)

    def __sub__(self, other: Self | float) -> Self:
        return self._element_wise_operation(other, sub)

    def __rsub__(self, other: float) -> Self:
        return type(self)([other - v for v in self])

    def __mul__(self, other: Self | float) -> Self:
        return self._element_wise_operation(other, mul)

    def __rmul__(self, other: float) -> Self:
        return self._element_wise_operation(other, mul)

    def __truediv__(self, other: Self | float) -> Self:
        return self._element_wise_operation(other, truediv)

    def __rtruediv__(self, other: float) -> Self:
        return type(self)([other / v for v in self])

    def __neg__(self) -> Self:
        return type(self)([-v for v in self._rows])
