from re import S
import typing
from collections.abc import Sequence
from operator import add, mul, sub, truediv
from typing import Callable, Self, overload

from src.auto import Vertex
from src.nn.Vector import Vector


class Matrix(Sequence):
    def __init__(self, data: Sequence[Sequence[float | Vertex]]):
        assert len(data) > 0, "At least one row of data must be supplied."
        m = len(data)

        cols = set()
        for r in range(m):
            cols.add(len(data[r]))
        assert len(cols) == 1, (
            f"Inconsistent column numbers in supplied data, n_cols: {cols}"
        )
        n = cols.pop()

        parsed = []
        for r in range(m):
            row = []
            for c in range(n):
                item = data[r][c]
                if isinstance(item, Vertex):
                    row.append(item)
                elif isinstance(item, float):
                    row.append(Vertex(item))
                else:
                    raise ValueError(
                        "All passed arguments must be either of type Vertex or float."
                    )
            parsed.append(row)

        self._rows: tuple[Vector, ...] = tuple(Vector(row) for row in parsed)
        self._cols: tuple[Vector, ...] = tuple(
            Vector(tuple(row[c] for row in parsed)) for c in range(n)
        )

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

    @overload
    def __matmul__(self, other: Vector) -> Vector: ...

    @overload
    def __matmul__(self, other: Self) -> Self: ...

    def __matmul__(self, other: Self | Vector) -> Self | Vector:
        is_vec = isinstance(other, Vector)
        if is_vec:
            other = type(self)([[v] for v in other])

        n, k1 = self.shape
        k2, m = other.shape
        assert k1 == k2, f"Incompatible matrix multiplication dims {k1}!={k2}"

        out = []
        for r in range(n):
            row = []
            for c in range(m):
                row.append(self._rows[r].dot(other._cols[c]))
            out.append(row)

        if is_vec:
            return Vector([v[0] for v in out])

        return type(self)(out)
