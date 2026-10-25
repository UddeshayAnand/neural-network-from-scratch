"""Binary MNIST classification with manual backpropagation."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import logging
import os
import shutil
import struct
import tempfile
from typing import Tuple

import numpy as np

from nn.activation import ReLU, Sigmoid
from nn.data import download_file
from nn.layer import Dense
from nn.loss import BinaryCrossEntropy
from nn.model import NeuralNetwork
from nn.optimizer import Adam

logger = logging.getLogger(__name__)


def _unzip_file(source_path: str):
    target_path = os.path.splitext(source_path)[0]
    temporary = None
    try:
        with gzip.open(source_path, "rb") as source:
            with tempfile.NamedTemporaryFile(dir=os.path.dirname(source_path), delete=False) as output:
                temporary = output.name
                shutil.copyfileobj(source, output)
        if os.path.getsize(temporary) == 0:
            raise OSError("Empty MNIST archive")
        os.replace(temporary, target_path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.remove(temporary)


def _download_url(url: str, data_dir: str) -> str:
    file_path = download_file(url, data_dir)
    # Validate cached archives too, recovering interrupted extraction and corruption.
    try:
        _unzip_file(file_path)
    except (OSError, EOFError):
        file_path = download_file(url, data_dir, force=True)
        _unzip_file(file_path)
    return file_path


def _read_file(
    file_path: str, buffer_size: int, discard_buffer_size: int
) -> np.ndarray:
    """Read a file and return its content as a numpy array"""

    logger.info(f"Reading file at path {file_path}")

    with open(file_path, "rb") as f:
        _ = f.read(discard_buffer_size)
        buffer = f.read(buffer_size)
    return np.frombuffer(buffer, dtype=np.uint8)


def _slice_data(
    images: np.ndarray, labels: np.ndarray, image_size: int
) -> Tuple[np.ndarray]:
    logger.info("Slicing data")

    zero_indices, *_ = np.where(np.squeeze(labels) == 0.0)
    one_indices, *_ = np.where(np.squeeze(labels) == 1.0)

    x = np.concatenate([images[zero_indices], images[one_indices]], axis=0)
    y = np.concatenate([labels[zero_indices], labels[one_indices]], axis=0)

    indices = np.random.permutation(y.shape[0])
    x = x[indices]
    y = y[indices]

    num_examples = x.shape[0]

    x = np.reshape(x, (num_examples, image_size * image_size))
    x = np.transpose(x) / 255.0
    y = np.transpose(y)
    return x, y


def _decode_data(
    image_path: str, label_path: str, num_images: int, image_size: int
) -> Tuple[np.ndarray]:
    """Read images and labels and return decoded numpy arrays"""

    logger.info(f"Decoding images at {image_path} and labels at {label_path}")

    with open(image_path, "rb") as source:
        header = source.read(16)
    with open(label_path, "rb") as source:
        label_header = source.read(8)
    if len(header) != 16 or struct.unpack(">IIII", header) != (2051, num_images, image_size, image_size):
        raise ValueError("Invalid MNIST image header")
    if len(label_header) != 8 or struct.unpack(">II", label_header) != (2049, num_images):
        raise ValueError("Invalid MNIST label header")
    buffer_size = image_size * image_size * num_images
    image_data = _read_file(image_path, buffer_size=buffer_size, discard_buffer_size=16)
    label_data = _read_file(label_path, buffer_size=num_images, discard_buffer_size=8)
    if image_data.size != buffer_size or label_data.size != num_images:
        raise ValueError("Truncated MNIST data")
    images = image_data.astype(np.float32).reshape(num_images, image_size, image_size)
    labels = label_data.reshape(num_images, 1)

    return images, labels


def _decode_and_slice_data(
    image_path: str, label_path: str, num_images: int, image_size: int
) -> Tuple[np.ndarray]:
    images, labels = _decode_data(image_path, label_path, num_images, image_size)
    return _slice_data(images, labels, image_size)


def _download_dataset(data_dir):
    """Download dataset to data_dir"""

    base_url = "https://raw.githubusercontent.com/fgnt/mnist/master"
    files = (
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    )
    urls = [base_url + "/" + file for file in files]

    os.makedirs(data_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda url: _download_url(url, data_dir), urls))


def _create_dataset(data_dir: str) -> Tuple[np.ndarray]:
    num_test_images = 10000
    num_train_images = 60000
    image_size = 28

    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte")
    test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte")
    test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte")

    x_train, y_train = _decode_and_slice_data(
        train_images_path, train_labels_path, num_train_images, image_size
    )
    x_test, y_test = _decode_and_slice_data(
        test_images_path, test_labels_path, num_test_images, image_size
    )

    return x_train, y_train, x_test, y_test


def _download(data_dir: str):
    _download_dataset(data_dir)


def mnist_load(data_dir: str) -> Tuple[np.ndarray]:
    _download(data_dir)
    return _create_dataset(data_dir)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    np.random.seed(42)
    logger.info("Fetching dataset")

    x_train, y_train, x_test, y_test = mnist_load("data")

    logger.info("Creating model")

    model = NeuralNetwork(
        layers=(
            (Dense(64), ReLU()),
            (Dense(64), ReLU()),
            (Dense(1), Sigmoid()),
        ),
        loss=BinaryCrossEntropy(),
        optimizer=Adam(learning_rate=0.01),
        regularization_factor=2.0,
    )

    logger.info("Training model")

    model.fit(x_train, y_train, epochs=20, verbose=True)

    logger.info("Evaluating trained model")

    loss = model.evaluate(x_test, y_test)

    logger.info(f"Validation loss: {np.squeeze(loss):.4f}")

    preds = model.predict(x_test)
    preds = (preds >= 0.5).astype(int)

    logger.info(f"First 5 predictions: {preds[:, :5]}")
    logger.info(f"First 5 labels     : {y_test[:, :5]}")

    acc = np.squeeze(np.mean(preds == y_test))

    logger.info(f"Test set accuracy  : {acc:.4f}")


if __name__ == "__main__":
    main()
