from abc import abstractmethod
import numpy as np
from numpy.typing import NDArray
from nn.common import Differentiable


class Activation(Differentiable):

    @abstractmethod
    def __call__(self, input_tensor: NDArray) -> NDArray: ...


class ReLU(Activation):
    def __call__(self, input_tensor: NDArray) -> NDArray:
        return np.maximum(input_tensor, 0)

    def gradient(self, input_tensor: NDArray) -> NDArray:
        _result = input_tensor.copy()
        _result[input_tensor > 0] = 1
        _result[input_tensor <= 0] = 0
        return _result


class Sigmoid(Activation):
    def __call__(self, input_tensor: NDArray) -> NDArray:
        x = np.asarray(input_tensor, dtype=float)
        result = np.empty_like(x)
        positive = x >= 0
        result[positive] = 1 / (1 + np.exp(-x[positive]))
        exp_x = np.exp(x[~positive])
        result[~positive] = exp_x / (1 + exp_x)
        return result

    def gradient(self, input_tensor: NDArray) -> NDArray:
        return self(input_tensor) * (1 - self(input_tensor))


class Linear(Activation):
    def __call__(self, input_tensor: NDArray) -> NDArray:
        return input_tensor

    def gradient(self, input_tensor: NDArray) -> NDArray:
        return np.ones_like(input_tensor)
