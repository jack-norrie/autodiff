from abc import abstractmethod
import typing

from copy import deepcopy

from src.primatives import Vertex
from typing import Any


class Optimizer:
    def __init__(self, parameters: dict, nu: float = 0.01, *args, **kwargs) -> None:
        self.parameters = parameters

        self.opt_parameters = self.init_opt_parameters()

        self.nu = nu

    @abstractmethod
    def init_opt_parameter(self) -> dict:
        raise NotImplementedError

    def init_opt_parameters(self):
        def dfs(parameters: dict | list | Vertex) -> dict | list:
            if isinstance(parameters, dict):
                opt_parameters = {}
                opt_parameters = typing.cast(dict, opt_parameters)

                for k, v in parameters.items():
                    if isinstance(v, Vertex):
                        opt_parameters[k] = self.init_opt_parameter()
                    else:
                        opt_parameters[k] = dfs(v)

            elif isinstance(parameters, list):
                opt_parameters = [None for _ in range(len(parameters))]
                opt_parameters = typing.cast(list, opt_parameters)

                for i, v in enumerate(parameters):
                    if isinstance(v, Vertex):
                        opt_parameters[i] = self.init_opt_parameter()
                    else:
                        opt_parameters[i] = dfs(v)

            else:
                raise ValueError(
                    "Parmas must have a matching nested structure (dicts and lists) of Vertex instanceces"
                )

            return opt_parameters

        return dfs(self.parameters)

    @abstractmethod
    def update(self, parameter: Vertex, opt_parameter) -> None:
        raise NotImplementedError

    def step(self):
        def opt(parameters: dict | list | Vertex, opt_parameters: dict | list) -> None:
            if isinstance(parameters, dict) and isinstance(opt_parameters, dict):
                for k in parameters:
                    opt(parameters[k], opt_parameters[k])
            elif isinstance(parameters, list) and isinstance(opt_parameters, list):
                for p, q in zip(parameters, opt_parameters):
                    opt(p, q)
            elif isinstance(parameters, Vertex):
                self.update(parameters, opt_parameters)
            else:
                raise ValueError(
                    f"Parmas/Optimizer-Params must have a matching nested structure (dicts and lists) of Vertex instanceces - got {type(parameters)} - {type(opt_parameters)}"
                )

        opt(self.parameters, self.opt_parameters)


class SGD(Optimizer):
    def init_opt_parameter(self) -> dict:
        return {}

    def update(self, parameter: Vertex, opt_parameter: None) -> None:
        parameter.value -= self.nu * parameter.grad


class MomentumSGD(Optimizer):
    def __init__(
        self, parameters: dict, nu: float = 0.01, momentum: float = 0.9
    ) -> None:
        super().__init__(parameters, nu)
        self.momentum = momentum

    def init_opt_parameter(self) -> dict:
        return {"momentum": 0.0}

    def update(self, parameter: Vertex, opt_parameter: dict) -> None:
        delta = self.momentum * opt_parameter["momentum"] - self.nu * parameter.grad
        parameter.value += delta
        opt_parameter["momentum"] = delta
