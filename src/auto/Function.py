from abc import ABC, abstractmethod
import typing
from typing import overload

if typing.TYPE_CHECKING:
    # Avoid circular depdency, since Vertex needs to implement Functions
    from src.auto.Vertex import Vertex


class Function(ABC):
    """
    Abstract base class for differentiable functions in the autodiff system.

    This class defines the interface for all differentiable functions and provides
    the mechanism for forward and backward propagation in the computational graph.
    """

    @classmethod
    def __call__(cls, *args: "Vertex") -> "Vertex":
        """
        Call the function on the given arguments.

        This method:
        1. Performs the forward computation
        2. Sets up the computational graph for backpropagation
        3. Returns the resulting Vertex

        Args:
            *args: Input Vertex objects

        Returns:
            Vertex: Result of the function application
        """
        z = cls.forward(*args)

        # Add parents and backwards function for backprop
        z._parents = args
        z._backward = cls.backward

        return z

    @staticmethod
    @abstractmethod
    def forward(*args: "Vertex") -> "Vertex":
        """
        Perform the forward computation of the function.

        Args:
            *args: Input Vertex objects

        Returns:
            Vertex: Result of the function application
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def backward(*args: "Vertex") -> tuple[float, ...]:
        """
        Compute the gradients of the function with respect to its inputs.

        Args:
            *args: Input Vertex objects

        Returns:
            tuple[float, ...]: Gradients with respect to each input
        """
        raise NotImplementedError
