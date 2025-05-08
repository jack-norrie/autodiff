from abc import abstractmethod

from src.nn import Vector


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
