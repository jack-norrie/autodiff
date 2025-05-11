from abc import abstractmethod, ABC

from src.nn import Vector


class Component(ABC):
    """
    Base class for all neural network components.

    This abstract class defines the interface for neural network components
    such as layers and provides a mechanism for parameter management.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize a Component with an empty parameters dictionary.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        self._parameters = {}

    def __call__(self, x: Vector) -> Vector:
        """
        Make the component callable, forwarding to the forward method.

        Args:
            x: Input vector

        Returns:
            Vector: Output of the forward pass
        """
        return self.forward(x)

    @property
    def parameters(self) -> dict:
        """
        Get the parameters of this component.

        Returns:
            dict: Dictionary of parameters
        """
        return self._parameters

    @abstractmethod
    def forward(self, x: Vector) -> Vector:
        """
        Perform the forward pass of the component.

        Args:
            x: Input vector

        Returns:
            Vector: Output of the forward pass
        """
        raise NotImplementedError
