import random
import typing
from abc import abstractmethod

from src.functions import mat_mul, vec_add
from src.nn.activation import relu
from src.primatives import Matrix, Vector, Vertex


class Layer:
    def __init__(self, *args, **kwargs) -> None:
        self._paramaters: dict[str, Vertex | Vector | Matrix] = {}

    @property
    def paramaters(self) -> dict:
        return self._paramaters

    @abstractmethod
    def forward(self, x: Vector) -> Vector:
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, in_dim, out_dim, bias: bool = True, seed: int = 42) -> None:
        random.seed(seed)
        self._paramaters = {
            "W": [
                [Vertex(random.uniform(-1, 1)) for _ in range(in_dim)]
                for _ in range(out_dim)
            ],
        }
        if bias:
            self._paramaters["b"] = [
                Vertex(random.uniform(-1, 1)) for _ in range(in_dim)
            ]

    def forward(self, x: Vector) -> Vector:
        z = mat_mul(self.paramaters["W"], x)
        z = typing.cast(Vector, z)

        if "b" in self.paramaters:
            z = vec_add(z, self.paramaters["b"])

        return z
