import math
import typing
from abc import abstractmethod
from collections.abc import Sequence

from src.auto import Vertex


class Optimizer:
    def __init__(self, parameters: dict, nu: float = 0.01, *args, **kwargs) -> None:
        self.parameters = parameters
        self.opt_parameters = self.init_opt_parameters()

        self.nu = nu
        self.t = 1

    @abstractmethod
    def init_opt_parameter(self) -> dict:
        raise NotImplementedError

    def init_opt_parameters(self):
        def dfs(parameters: dict | Sequence | Vertex) -> dict | list:
            if isinstance(parameters, dict):
                opt_parameters = {}
                opt_parameters = typing.cast(dict, opt_parameters)

                for k, v in parameters.items():
                    if isinstance(v, Vertex):
                        opt_parameters[k] = self.init_opt_parameter()
                    else:
                        opt_parameters[k] = dfs(v)

            elif isinstance(parameters, Sequence):
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
            elif isinstance(parameters, Sequence) and isinstance(
                opt_parameters, Sequence
            ):
                for p, q in zip(parameters, opt_parameters):
                    opt(p, q)
            elif isinstance(parameters, Vertex):
                self.update(parameters, opt_parameters)
            else:
                raise ValueError(
                    f"Parmas/Optimizer-Params must have a matching nested structure (dicts and lists) of Vertex instanceces - got {type(parameters)} - {type(opt_parameters)}"
                )

        opt(self.parameters, self.opt_parameters)
        self.t += 1


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


class AdaGrad(Optimizer):
    def init_opt_parameter(self) -> dict:
        return {"scale": 0.0}

    def update(self, parameter: Vertex, opt_parameter: dict) -> None:
        g = parameter.grad
        opt_parameter["scale"] += g**2
        delta = -self.nu * g / (math.sqrt(opt_parameter["scale"]) + 1e-8)
        parameter.value += delta


class RMSProp(Optimizer):
    def __init__(self, parameters: dict, nu: float = 0.01, beta: float = 0.9) -> None:
        super().__init__(parameters, nu)
        self.beta = beta

    def init_opt_parameter(self) -> dict:
        return {"scale": 0.0}

    def update(self, parameter: Vertex, opt_parameter: dict) -> None:
        g = parameter.grad
        opt_parameter["scale"] = (
            self.beta * opt_parameter["scale"] + (1 - self.beta) * g**2
        )
        delta = -self.nu * g / (math.sqrt(opt_parameter["scale"]) + 1e-8)
        parameter.value += delta


class Adam(Optimizer):
    def __init__(
        self,
        parameters: dict,
        nu: float = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
    ) -> None:
        super().__init__(parameters, nu)
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def init_opt_parameter(self) -> dict:
        return {"moment_1": 0.0, "moment_2": 0.0}

    def update(self, parameter: Vertex, opt_parameter: dict) -> None:
        g = parameter.grad

        opt_parameter["moment_1"] = (
            self.beta_1 * opt_parameter["moment_1"] + (1 - self.beta_1) * g
        )
        opt_parameter["moment_2"] = (
            self.beta_2 * opt_parameter["moment_2"] + (1 - self.beta_2) * g**2
        )

        m1 = opt_parameter["moment_1"] / (1 - self.beta_1**self.t)
        m2 = opt_parameter["moment_2"] / (1 - self.beta_2**self.t)

        delta = -self.nu * m1 / (math.sqrt(m2) + 1e-8)

        parameter.value += delta
