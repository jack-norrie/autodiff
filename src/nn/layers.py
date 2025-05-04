import random
import typing
from abc import abstractmethod

from src.functions import mat_mul, vec_add
from src.nn.activation import relu
from src.primatives import Matrix, Vector, Vertex


class Layer:
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


class Sequential(Layer):
    def __init__(self, layers: list[Layer]) -> None:
        super().__init__()
        self._layers = layers
        for i, layer in enumerate(layers):
            self._parameters[str(i)] = layer._parameters

    def forward(self, x: Vector) -> Vector:
        z = x
        for layer in self._layers:
            z = layer(z)
        return z


class Linear(Layer):
    def __init__(self, in_dim, out_dim, bias: bool = True, seed: int = 42) -> None:
        super().__init__()

        random.seed(seed)
        self._parameters["W"] = [
            [Vertex(random.uniform(-1, 1)) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        if bias:
            self._parameters["b"] = [
                Vertex(random.uniform(-1, 1)) for _ in range(out_dim)
            ]

    def forward(self, x: Vector) -> Vector:
        z = mat_mul(self.parameters["W"], x)
        z = typing.cast(Vector, z)

        if "b" in self.parameters:
            z = vec_add(z, self.parameters["b"])

        return z
