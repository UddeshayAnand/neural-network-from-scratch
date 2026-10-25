"""Offline AND-gate demonstration. Run: python and_gate.py."""
import numpy as np

from nn.activation import Sigmoid
from nn.layer import Dense
from nn.loss import BinaryCrossEntropy
from nn.model import NeuralNetwork
from nn.optimizer import SGD


def main():
    np.random.seed(42)

    # Columns are examples; rows are features or output units.
    x = np.array([[0., 0., 1., 1.],
                  [0., 1., 0., 1.]])
    y = np.array([[0., 0., 0., 1.]])

    model = NeuralNetwork(
        layers=((Dense(1), Sigmoid()),),
        loss=BinaryCrossEntropy(),
        optimizer=SGD(learning_rate=0.5),
    )

    print(f"Initial loss: {model.evaluate(x, y):.4f}")
    model.fit(x, y, epochs=1000)
    print(f"Final loss: {model.evaluate(x, y):.4f}")
    print("Predictions:", (model.predict(x) >= 0.5).astype(int))


if __name__ == "__main__":
    main()
