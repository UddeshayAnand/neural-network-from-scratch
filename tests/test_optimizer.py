import unittest
import numpy as np
from nn.layer import Dense
from nn.optimizer import SGD, Adam, RMSprop


class OptimizerTest(unittest.TestCase):
    def setUp(self):
        self.layer = Dense(units=10)
        self.layer(input_tensor=np.zeros((10, 1)))
        self.grad_weights = np.random.randn(*self.layer.weights.shape)
        self.grad_bias = np.random.randn(*self.layer.bias.shape)

    def test_SGD_update_weights(self):
        learning_rate = 0.01
        optimizer = SGD(learning_rate=learning_rate)
        original_weights = self.layer.weights.copy()

        optimizer.update_weights(self.layer, self.grad_weights)

        expected_weights = original_weights - learning_rate * self.grad_weights
        np.testing.assert_allclose(self.layer.weights, expected_weights, atol=1e-12)

    def test_SGD_update_bias(self):
        learning_rate = 0.01
        optimizer = SGD(learning_rate=learning_rate)
        original_bias = self.layer.bias.copy()

        optimizer.update_bias(self.layer, self.grad_bias)

        expected_bias = original_bias - learning_rate * self.grad_bias
        np.testing.assert_allclose(self.layer.bias, expected_bias, atol=1e-12)

    def test_Adam_update_weights(self):
        learning_rate = 0.01
        optimizer = Adam(learning_rate=learning_rate)
        original_weights = self.layer.weights.copy()

        optimizer.update_weights(self.layer, self.grad_weights)

        expected_weights = original_weights - learning_rate * self.grad_weights / (np.abs(self.grad_weights) + 1e-8)
        np.testing.assert_allclose(self.layer.weights, expected_weights, atol=1e-12)

    def test_Adam_update_bias(self):
        learning_rate = 0.01
        optimizer = Adam(learning_rate=learning_rate)
        original_bias = self.layer.bias.copy()

        optimizer.update_bias(self.layer, self.grad_bias)

        expected_bias = original_bias - learning_rate * self.grad_bias / (np.abs(self.grad_bias) + 1e-8)
        np.testing.assert_allclose(self.layer.bias, expected_bias, atol=1e-12)

    def test_RMSprop_update_weights(self):
        learning_rate = 0.01
        optimizer = RMSprop(learning_rate=learning_rate)
        original_weights = self.layer.weights.copy()

        optimizer.update_weights(self.layer, self.grad_weights)

        expected_weights = original_weights - learning_rate * self.grad_weights / (np.sqrt(.1 * self.grad_weights**2) + 1e-8)
        np.testing.assert_allclose(self.layer.weights, expected_weights, atol=1e-12)

    def test_RMSprop_update_bias(self):
        learning_rate = 0.01
        optimizer = RMSprop(learning_rate=learning_rate)
        original_bias = self.layer.bias.copy()

        optimizer.update_bias(self.layer, self.grad_bias)

        expected_bias = original_bias - learning_rate * self.grad_bias / (np.sqrt(.1 * self.grad_bias**2) + 1e-8)
        np.testing.assert_allclose(self.layer.bias, expected_bias, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
