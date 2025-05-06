from abc import ABC, abstractmethod
import typing

if typing.TYPE_CHECKING:
    from src.primitives.Vertex import Vertex


class Function(ABC):
    @classmethod
    def __call__(cls, *args) -> "Vertex":
        z = cls.forward(*args)

        # Add paranets and backwards function for backprop
        z._parents = args
        z._backward = cls.backward

        return z

    @staticmethod
    @abstractmethod
    def forward(*args) -> "Vertex":
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def backward(*args) -> tuple[float, ...]:
        raise NotImplementedError
