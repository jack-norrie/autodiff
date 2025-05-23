import math
import random
import typing

from src.auto import Function
from src.nn.Matrix import Matrix
from src.nn.Vector import Vector
from src.nn.components import Component
from src.nn.initialisers import WeightInitialiser, He


class Linear(Component):
    """
    A linear (fully connected) layer in a neural network.

    This layer performs a linear transformation: y = Wx + b,
    optionally followed by an activation function.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        bias: bool = True,
        activation: Function | None = None,
        weight_initialiser: WeightInitialiser = He(),
        seed: int = 42,
    ) -> None:
        """
        Initialize a linear layer.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            bias: Whether to include a bias term
            activation: Optional activation function to apply
            weight_initialiser: Strategy for initializing weights
            seed: Random seed for reproducibility
        """
        super().__init__()

        random.seed(seed)
        W = Matrix(
            [
                [random.gauss(0, math.sqrt(2 / in_dim)) for _ in range(in_dim)]
                for _ in range(out_dim)
            ]
        )
        weight_initialiser(W)
        self._parameters["W"] = W

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
