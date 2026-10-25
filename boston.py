import os

import numpy as np
from numpy.typing import NDArray
import logging
from nn.data import download_file

logger = logging.getLogger(__name__)

from nn.activation import ReLU, Linear
from nn.layer import Dense
from nn.loss import MeanAbsoluteError
from nn.model import NeuralNetwork
from nn.optimizer import RMSprop


def _download_url(url: str, data_dir: str) -> str:
    return download_file(url, data_dir)


def _read_file(file_path: str) -> NDArray:
    """Read a file and return its content as a numpy array"""

    logger.info(f"Reading file at path {file_path}")

    with open(file_path, "r") as f:
        data = f.read()

    return np.array(
        [list(map(float, line.split())) for line in data.split("\n") if line.strip()]
    )


def _decode_data(data_path: str) -> NDArray:
    """Read data and return decoded numpy array"""

    logger.info(f"Decoding data at {data_path}")

    return _read_file(data_path)


def _orignal_data(x: NDArray) -> NDArray:
    """Return the orignal data as it is"""
    return x


def _normalize_data(x: NDArray) -> NDArray:
    """Normalize data by dividing by max value"""

    logger.info("Normalizing data")

    scale = np.max(np.abs(x), axis=0)
    return x / np.where(scale == 0, 1, scale)


def train_test_split(x, y, test_size=0.3, random_state=None):
    """Split the data into train and test sets"""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    rng = np.random.default_rng(random_state)

    n_samples = x.shape[0]
    test_size = int(test_size * n_samples)
    if test_size == 0 or test_size == n_samples:
        raise ValueError("Split must contain training and test examples")
    shuffled_indices = rng.permutation(n_samples)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    x_train = x[train_indices].T
    y_train = y[train_indices].reshape(1, -1)
    x_test = x[test_indices].T
    y_test = y[test_indices].reshape(1, -1)

    return x_train, y_train, x_test, y_test


def _download_dataset(data_dir):
    """Download dataset to data_dir"""

    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    )

    os.makedirs(data_dir, exist_ok=True)

    _download_url(url, data_dir)


def boston_load(preprocess_fn=_orignal_data, test_size=0.3, return_target_scale=False):
    """Load the Boston Housing dataset and return a train-test split"""

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    if not os.path.isdir(data_dir) or not os.path.isfile(
        os.path.join(data_dir, "housing.data")
    ):
        _download_dataset(data_dir)

    data_path = os.path.join(data_dir, "housing.data")

    x = _decode_data(data_path)
    y = x[:, -1]  # The last column in the dataset is the target variable

    x = x[:, :-1]  # Remove the target variable from the input features

    split = train_test_split(x, y, test_size=test_size, random_state=42)
    x_train, y_train, x_test, y_test = split
    target_scale = 1.0
    if preprocess_fn is _normalize_data:
        # Fit scaling only to the training partition, then reuse it on test data.
        feature_scale = np.max(np.abs(x_train), axis=1, keepdims=True)
        feature_scale = np.where(feature_scale == 0, 1, feature_scale)
        target_scale = float(np.max(np.abs(y_train))) or 1.0
        split = (x_train / feature_scale, y_train / target_scale,
                 x_test / feature_scale, y_test / target_scale)
    elif preprocess_fn not in (None, _orignal_data):
        raise ValueError("Use _normalize_data or no preprocessing to avoid inconsistent transforms")
    if return_target_scale:
        return (*split, target_scale)
    return split



def r2_score(preds: NDArray, y_test: NDArray) -> float:
    if np.shape(preds) != np.shape(y_test) or np.size(y_test) < 2:
        raise ValueError("R-squared requires matching arrays with at least two values")
    mean_y = np.mean(y_test)
    ss_total = np.sum((y_test - mean_y) ** 2)
    ss_residual = np.sum((y_test - preds) ** 2)
    if ss_total == 0:
        return 1.0 if ss_residual == 0 else 0.0
    r2 = 1 - (ss_residual / ss_total)
    return r2


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    np.random.seed(42)
    logger.info("Creating dataset")

    x_train, y_train, x_test, y_test, target_scale = boston_load(
        preprocess_fn=_normalize_data, return_target_scale=True
    )

    logger.info("Creating model")

    model = NeuralNetwork(
        layers=(
            (Dense(13), ReLU()),
            (Dense(128), ReLU()),
            (Dense(64), ReLU()),
            (Dense(1), Linear()),
        ),
        loss=MeanAbsoluteError(),
        optimizer=RMSprop(learning_rate=0.0003),
        regularization_factor=0.001,
    )

    logger.info("Training model")

    model.fit(x_train, y_train, epochs=24000, verbose=True, log_interval=2000)

    logger.info("Evaluating trained model")

    loss = model.evaluate(x_test, y_test)

    logger.info(f"Test loss (normalized MAE): {np.squeeze(loss):.4f}")

    preds = model.predict(x_test) * target_scale
    y_test = y_test * target_scale
    logger.info(f"Test MAE (original target units): {np.mean(np.abs(preds - y_test)):.4f}")

    logger.info(f"First 5 predictions: {preds[:, :5]}")
    logger.info(f"First 5 labels     : {y_test[:, :5]}")

    acc = r2_score(preds, y_test)

    logger.info(f"Test R-squared     : {acc:.4f}")


if __name__ == "__main__":
    main()
