import math
import random
from abc import ABC, abstractmethod

from src.nn import Matrix


class WeightInitialiser(ABC):
    @abstractmethod
    def __call__(self, data: Matrix, seed: int = 42) -> None:
        raise NotImplementedError


class VarianceInitialiser(WeightInitialiser):
    @abstractmethod
    def _calculate_sigma(self, f_in: int, f_out: int) -> float:
        raise NotImplementedError

    def __next__(self):
        return random.gauss(0, self._sigma)

    def __call__(self, weight: Matrix, seed: int = 42) -> None:
        random.seed(seed)

        f_out, f_in = weight.shape
        self._sigma = self._calculate_sigma(f_in, f_out)

        for r in range(f_out):
            for c in range(f_in):
                weight[r][c].value = next(self)


class LeCun(VarianceInitialiser):
    def _calculate_sigma(self, f_in: int, f_out: int) -> float:
        return math.sqrt(1 / f_in)


class Xavier(VarianceInitialiser):
    def _calculate_sigma(self, f_in: int, f_out: int) -> float:
        return math.sqrt(2 / (f_in + f_out))


class He(VarianceInitialiser):
    def _calculate_sigma(self, f_in: int, f_out: int) -> float:
        return math.sqrt(2 / f_in)
