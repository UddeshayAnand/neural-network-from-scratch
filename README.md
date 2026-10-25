# Neural Network from Scratch

A small neural-network library built with **Python and NumPy**, with explicit forward propagation, backpropagation, and parameter updates. The project makes the mechanics of a dense neural network readable without a deep-learning framework or automatic differentiation.

It includes binary classification and regression examples, three optimizers, four losses, and a CSV training logger.

## What is implemented

| Component | Available implementations |
| --- | --- |
| Layer | `Dense`, with lazy He-style random initialization and zero biases |
| Activations | `ReLU`, `Sigmoid`, `Linear` |
| Losses | `BinaryCrossEntropy`, `MeanSquaredError`, `MeanAbsoluteError`, `HuberLoss` |
| Optimizers | `SGD`, `Adam`, `RMSprop` |
| Training | Full-batch gradient updates, configurable epochs and logging interval |
| Regularization | L2 contribution to weight gradients; excluded from reported loss |
| Callback | `CSVLogger` for epoch and loss records |
| Examples | MNIST digits **0 vs. 1** and Boston Housing regression |

NumPy is the only runtime dependency. Dataset downloading, concurrency, and logging use the Python standard library.

## Setup

Run commands from this project's root directory. The core library and existing tests were checked with **Python 3.12.5 and NumPy 1.26.4**. The default requirements allow compatible NumPy updates. To match the validated NumPy version on Python 3.12, install `requirements.txt` instead:

```bash
python -m pip install -r requirements.txt
```

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the project requirements:

```bash
python -m pip install -r requirements.txt
```

For just the core library, tests, and offline example, `python -m pip install numpy` is sufficient. Optional development tooling is listed separately in `requirements-dev.txt`.

## Quick start: learn an AND gate

This example needs no dataset download. Run the included script:

```bash
python and_gate.py
```

Its implementation is shown below:

```python
import numpy as np

from nn.activation import Sigmoid
from nn.layer import Dense
from nn.loss import BinaryCrossEntropy
from nn.model import NeuralNetwork
from nn.optimizer import SGD

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
```

Observed output in the reviewed environment:

```text
Initial loss: 0.7054
Final loss: 0.0346
Predictions: [[0 0 0 1]]
```

This verifies one simple training path. The automated suite separately checks all implemented losses and optimizers.

## Array shapes and API

The library uses **features first, examples second**:

| Array | Shape |
| --- | --- |
| Input `x` | `(number_of_features, number_of_examples)` |
| Target `y` | `(number_of_outputs, number_of_examples)` |
| Dense weights | `(units, input_features)` |
| Dense bias | `(units, 1)` |
| Predictions | `(number_of_outputs, number_of_examples)` |

Use two-dimensional arrays, including for one example or one output. Layers initialize their parameters on the first forward pass.

Each entry in `layers` is a `(layer, activation)` pair:

```python
from nn.activation import ReLU, Sigmoid

layers = (
    (Dense(64), ReLU()),
    (Dense(64), ReLU()),
    (Dense(1), Sigmoid()),
)
```

Pass `layers=layers`, a loss instance, and an optimizer instance to `NeuralNetwork`. For scalar regression, use a final `Linear()` activation and a regression loss, using the same mean-loss derivative convention as classification.

- `model(x)` performs a forward pass.
- `model.fit(x, y, epochs=..., verbose=True, log_interval=...)` trains on the entire dataset once per epoch and returns `None`.
- `model.predict(x)` returns raw outputs. Apply a threshold yourself for binary classification.
- `model.evaluate(x, y)` returns the configured data loss. It does not calculate accuracy or include the L2 penalty.

`model.objective(x, y)` returns data loss plus `regularization_factor * sum(W**2) / (2 * number_of_examples)`. Biases are not regularized. `model.learning_rate` reads or changes the optimizer rate.

### Record training loss

```python
from nn.callback import CSVLogger

# The parent directory must already exist. Existing files raise an error
# unless overwrite=True is explicitly passed.
logger = CSVLogger("training_loss.csv")
model.fit(x, y, epochs=1000, callbacks=(logger,))
```

The CSV header is `Epoch,Loss`. Logged loss is measured before that epoch's parameter update. There is no history object or validation callback built in.

## Dataset examples

### MNIST: binary digit classification

```bash
python mnist.py
```

The script downloads four compressed IDX files from the `fgnt/mnist` GitHub mirror into `data/`, keeps digits 0 and 1, flattens the 28 x 28 images into 784 features, and scales pixels by 255.

The network is `784 -> 64 -> 64 -> 1`, with ReLU hidden activations and a sigmoid output. It trains for 20 full-batch epochs using binary cross-entropy, Adam with learning rate `0.01`, and L2 factor `2.0`. The script prints held-out loss, a few predictions, and binary accuracy.

**Verified result:** test binary cross-entropy **0.0029** and accuracy **0.9995 (99.95%)** after 20 epochs with seed 42. This applies only to digits 0 and 1. Downloads use platform-independent URLs, timeouts, atomic writes, and archive recovery.

### Boston Housing: regression

```bash
python boston.py
```

The script downloads `housing.data` from the UCI URL specified in `boston.py`, uses 13 input features and one target, and creates a 70% training / 30% test split with seed 42.

The network is `13 -> 13 -> 128 -> 64 -> 1`, with ReLU hidden activations and a linear output. It trains for **24,000 full-batch epochs**, using MAE, RMSprop with learning rate `0.0003`, and L2 factor `0.001`.

**Verified result:** test normalized MAE **0.0616**, original-target MAE **3.0793**, and **R-squared 0.7514** after 24,000 epochs with seed 42. Scaling is fitted on the training partition only. The loader can return the target scale with `return_target_scale=True`, allowing predictions to be restored to original units.

Both scripts require internet access for their initial downloads. Both dataset URLs were downloaded successfully during validation.

## How training works

For each full-batch epoch:

1. Each dense layer computes `Z = W @ A_previous + b`, then applies its activation.
2. The configured loss compares predictions with targets.
3. Backpropagation applies the chain rule in reverse layer order and stores weight and bias gradients.
4. The optimizer updates each layer's parameters.
5. Callbacks receive the epoch number and the pre-update loss.

The input gradient is propagated using the layer's **weights**. All derivatives are written explicitly in `nn/`; there is no automatic differentiation engine.

## Project layout

```text
.
|-- nn/
|   |-- activation.py       # ReLU, sigmoid, linear and their derivatives
|   |-- callback.py         # Callback interface and CSV logger
|   |-- common.py           # Differentiable interface
|   |-- data.py             # Shared atomic downloader
|   |-- layer.py            # Dense layer and parameter storage
|   |-- loss.py             # Four losses and their derivatives
|   |-- model.py            # Forward pass, training, backpropagation
|   `-- optimizer.py        # SGD, Adam, RMSprop
|-- tests/                  # 24 unittest tests
|-- and_gate.py             # Offline quick start
|-- mnist.py
|-- boston.py
|-- requirements.txt
`-- README.md
```

## Limitations and future work

Mini-batches, softmax / categorical cross-entropy, dropout, model save/load, early stopping, and a learning-rate scheduler are not implemented. This remains an educational dense-network implementation. Rerun the dataset examples when changing preprocessing, architecture, or training settings.
