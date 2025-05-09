from src.nn import Vector
from src.nn.components import Component


class Sequential(Component):
    """
    A sequential container of neural network components.

    This component chains multiple components together, passing the output
    of one component as input to the next.
    """

    def __init__(self, layers: list[Component]) -> None:
        """
        Initialize a Sequential component with a list of layers.

        Args:
            layers: List of Component objects to be applied in sequence
        """
        super().__init__()
        self._layers = layers
        for i, layer in enumerate(layers):
            self._parameters[str(i)] = layer._parameters

    def forward(self, x: Vector) -> Vector:
        z = x
        for layer in self._layers:
            z = layer(z)
        return z
