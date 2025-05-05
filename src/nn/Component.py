import math
import random
import typing
from abc import abstractmethod

from src.functions import Function, mat_mul, vec_add
from src.primitives import Vector, Vertex


class Component:
    def __init__(self, *args, **kwargs) -> None:
        self._parameters = {}

    def __call__(self, x: Vector) -> Vector:
        return self.forward(x)

    @property
    def parameters(self) -> dict:
        return self._parameters

    @abstractmethod
    def forward(self, x: Vector) -> Vector:
        raise NotImplementedError


class Sequential(Component):
    def __init__(self, layers: list[Component]) -> None:
        super().__init__()
        self._layers = layers
        for i, layer in enumerate(layers):
            self._parameters[str(i)] = layer._parameters

    def forward(self, x: Vector) -> Vector:
        z = x
        for layer in self._layers:
            z = layer(z)
        return z


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
        self._parameters["W"] = [
            [Vertex(random.gauss(0, math.sqrt(2 / in_dim))) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        if bias:
            self._parameters["b"] = [Vertex(0.0) for _ in range(out_dim)]

        self._activation = activation

    def forward(self, x: Vector) -> Vector:
        z = mat_mul(self.parameters["W"], x)
        z = typing.cast(Vector, z)

        if "b" in self.parameters:
            z = vec_add(z, self.parameters["b"])

        if self._activation is not None:
            z = [self._activation(v) for v in z]

        return z
