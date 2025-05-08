import math
import random
import typing

from src.auto import Function
from src.nn import Matrix, Vector
from src.nn.components import Component


class Linear(Component):
    def __init__(
        self,
        in_dim,
        out_dim,
        bias: bool = True,
        activation: Function | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        random.seed(seed)
        self._parameters["W"] = Matrix(
            [
                [random.gauss(0, math.sqrt(2 / in_dim)) for _ in range(in_dim)]
                for _ in range(out_dim)
            ]
        )
        if bias:
            self._parameters["b"] = Vector([0.0 for _ in range(out_dim)])

        self._activation = activation

    def forward(self, x: Vector) -> Vector:
        z = self.parameters["W"] @ x
        z = typing.cast(Vector, z)

        if "b" in self.parameters:
            z = z + self.parameters["b"]

        if self._activation is not None:
            z = Vector([self._activation(v) for v in z])

        return z
