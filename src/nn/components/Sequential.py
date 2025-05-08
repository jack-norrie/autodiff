from src.nn import Vector
from src.nn.components import Component


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
