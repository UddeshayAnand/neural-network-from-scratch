import gzip
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np
import boston
import mnist
from nn.activation import Linear, ReLU, Sigmoid
from nn.data import download_file
from nn.layer import Dense
from nn.loss import BinaryCrossEntropy, MeanSquaredError, MeanAbsoluteError, HuberLoss
from nn.model import NeuralNetwork
from nn.optimizer import SGD, Adam, RMSprop


class CorrectnessTest(unittest.TestCase):
    def test_multilayer_gradients(self):
        x = np.array([[.2, .5, .9], [.4, .1, .3]])
        y = np.array([[.1, .2, .4], [.7, .8, .6]])
        for loss in (BinaryCrossEntropy(), MeanSquaredError(), MeanAbsoluteError(), HuberLoss(.2)):
            for regularization in (0., .3):
                with self.subTest(loss=type(loss).__name__, l2=regularization):
                    np.random.seed(4)
                    model = NeuralNetwork(((Dense(3), Sigmoid()), (Dense(2), Sigmoid())),
                                          loss, SGD(.01), regularization)
                    model(x)
                    model.backward_step(y)
                    gradients = [(layer.grad_weights.copy(), layer.grad_bias.copy())
                                 for layer, _ in model._layers]
                    for (layer, _), (dw, db) in zip(model._layers, gradients):
                        for parameters, gradient in ((layer.weights, dw), (layer.bias, db)):
                            for index in np.ndindex(parameters.shape):
                                original = parameters[index]
                                parameters[index] = original + 1e-6
                                plus = model.objective(x, y)
                                parameters[index] = original - 1e-6
                                minus = model.objective(x, y)
                                parameters[index] = original
                                self.assertAlmostEqual(gradient[index], (plus-minus)/2e-6, places=7)

    def test_adam_reference_multiple_layers_and_steps(self):
        optimizer = Adam(.01)
        layers = [Dense(2), Dense(2)]
        expected = {}
        for layer in layers:
            layer(np.ones((2, 1)))
            layer.weights[:] = 0
            layer.bias[:] = 0
        for step in range(1, 4):
            for number, layer in enumerate(layers):
                optimizer.ln = number
                for kind, parameters, update in (("w", layer.weights, optimizer.update_weights),
                                                 ("b", layer.bias, optimizer.update_bias)):
                    key = (number, kind)
                    value, m, v = expected.get(key, (np.zeros_like(parameters), 0., 0.))
                    grad = np.full_like(parameters, (-1)**step * (step + number))
                    m = .9*m + .1*grad
                    v = .999*v + .001*grad**2
                    value = value - .01*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
                    update(layer, grad)
                    np.testing.assert_allclose(parameters, value, atol=1e-12, rtol=1e-12)
                    expected[key] = (value, m, v)

    def test_rmsprop_reference(self):
        optimizer = RMSprop(.01)
        for number in range(2):
            layer = Dense(1)
            layer(np.ones((1, 1)))
            layer.weights[:] = 0
            layer.bias[:] = 0
            optimizer.ln = number
            value, cache = 0., 0.
            for gradient in (2., -3., 1.):
                cache = .9*cache + .1*gradient**2
                value -= .01*gradient/(np.sqrt(cache)+1e-8)
                optimizer.update_weights(layer, np.array([[gradient]]))
                optimizer.update_bias(layer, np.array([[gradient]]))
                self.assertAlmostEqual(layer.weights.item(), value, places=12)
                self.assertAlmostEqual(layer.bias.item(), value, places=12)

    def test_current_batch_and_learning_rate(self):
        model = NeuralNetwork(((Dense(1), Linear()),), MeanSquaredError(), SGD(.01))
        model.predict(np.ones((1, 2)))
        model.fit(np.ones((1, 5)), np.zeros((1, 5)), epochs=0)
        self.assertEqual(model._num_examples, 5)
        model._layers[0][0].weights[:] = 1
        model._layers[0][0].bias[:] = 0
        self.assertEqual(model.learning_rate, .01)
        model.learning_rate = .1
        model.fit(np.ones((1, 5)), np.zeros((1, 5)), epochs=1)
        self.assertAlmostEqual(model._layers[0][0].weights.item(), .8)
        self.assertAlmostEqual(model._layers[0][0].bias.item(), -.2)

    def test_convergence(self):
        x = np.array([[0., 0., 1., 1.], [0., 1., 0., 1.]])
        y = np.array([[0., 0., 0., 1.]])
        for optimizer in (SGD(.5), Adam(.05), RMSprop(.05)):
            np.random.seed(42)
            model = NeuralNetwork(((Dense(1), Sigmoid()),), BinaryCrossEntropy(), optimizer)
            model.fit(x, y, epochs=1000)
            np.testing.assert_array_equal(model.predict(x) >= .5, y)
            self.assertLess(model.evaluate(x, y), .05)
        for loss in (MeanSquaredError(), MeanAbsoluteError(), HuberLoss()):
            np.random.seed(42)
            x = np.array([[-1., -.5, 0., .5, 1.]])
            y = 2*x + .3
            model = NeuralNetwork(((Dense(1), Linear()),), loss, SGD(.05))
            model.fit(x, y, epochs=1000)
            self.assertLess(np.mean(np.abs(model.predict(x)-y)), .06)

    def test_validation_and_stable_sigmoid(self):
        with np.errstate(over="raise", invalid="raise"):
            np.testing.assert_allclose(Sigmoid()(np.array([-1000., 0., 1000.])), [0., .5, 1.])
        model = NeuralNetwork(((Dense(1), Linear()),), MeanSquaredError(), SGD(.1))
        for x, y in ((np.ones(3), np.ones((1, 3))),
                     (np.ones((1, 3)), np.ones(3)),
                     (np.empty((1, 0)), np.empty((1, 0)))):
            with self.assertRaises(ValueError):
                model.fit(x, y, 1)
        with self.assertRaises(ValueError):
            model.fit(np.ones((1, 3)), np.ones((1, 3)), 1, log_interval=0)
        with self.assertRaises(ValueError):
            Dense(0)


class DatasetTest(unittest.TestCase):
    def response(self, payload, length=None):
        response = io.BytesIO(payload)
        response.headers = {"Content-Length": str(len(payload) if length is None else length)}
        return response

    def test_atomic_download_and_cache(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            with patch("nn.data.urlopen", return_value=self.response(b"data")) as request:
                target = download_file("https://example.com/data.txt", directory)
                self.assertEqual(Path(target).read_bytes(), b"data")
                self.assertEqual(download_file("https://example.com/data.txt", directory), target)
                request.assert_called_once_with("https://example.com/data.txt", timeout=30)
            with patch("nn.data.urlopen", return_value=self.response(b"bad", 100)):
                with self.assertRaises(OSError):
                    download_file("https://example.com/data.txt", directory, force=True)
            self.assertEqual(Path(target).read_bytes(), b"data")
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_http_error(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            error = HTTPError("https://example.com/file", 404, "missing", {}, None)
            with patch("nn.data.urlopen", side_effect=error):
                with self.assertRaises(HTTPError):
                    download_file("https://example.com/file", directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_archive_recovery(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            archive = Path(directory)/"images.gz"
            archive.write_bytes(gzip.compress(b"images"))
            with patch("nn.data.urlopen") as request:
                mnist._download_url("https://example.com/images.gz", directory)
                request.assert_not_called()
            self.assertEqual((Path(directory)/"images").read_bytes(), b"images")
            archive.write_bytes(b"corrupt")
            with patch("nn.data.urlopen", return_value=self.response(gzip.compress(b"recovered"))):
                mnist._download_url("https://example.com/images.gz", directory)
            self.assertEqual((Path(directory)/"images").read_bytes(), b"recovered")

    def test_mnist_idx_decoding(self):
        import struct
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            images = Path(directory) / "images"
            labels = Path(directory) / "labels"
            images.write_bytes(struct.pack(">IIII", 2051, 2, 2, 2) + bytes(range(8)))
            labels.write_bytes(struct.pack(">II", 2049, 2) + bytes([0, 1]))
            x, y = mnist._decode_and_slice_data(str(images), str(labels), 2, 2)
            self.assertEqual(x.shape, (4, 2))
            self.assertEqual(y.shape, (1, 2))
            self.assertAlmostEqual(x.max(), 7 / 255)
            labels.write_bytes(struct.pack(">II", 2049, 2) + bytes([0]))
            with self.assertRaisesRegex(ValueError, "Truncated"):
                mnist._decode_data(str(images), str(labels), 2, 2)

    def test_mnist_urls(self):
        urls = []
        def capture(url, directory):
            urls.append(url)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            with patch("mnist._download_url", side_effect=capture):
                mnist._download_dataset(directory)
        self.assertEqual(len(urls), 4)
        self.assertTrue(all("\\" not in url and "/master/" in url for url in urls))

    def test_housing_scaling_uses_training_only(self):
        data = np.column_stack((np.arange(1., 11.), np.zeros(10), np.arange(1., 11.)*2))
        split = boston.train_test_split(data[:, :-1], data[:, -1], random_state=42)
        # Put a large value exclusively in a held-out example.
        heldout_value = split[2][0, 0]
        data[data[:, 0] == heldout_value, 0] = 10000
        raw = boston.train_test_split(data[:, :-1], data[:, -1], random_state=42)
        with patch("boston.os.path.isfile", return_value=True), patch("boston.os.path.isdir", return_value=True), patch("boston._decode_data", return_value=data):
            x_train, y_train, x_test, y_test, scale = boston.boston_load(
                boston._normalize_data, return_target_scale=True)
        self.assertAlmostEqual(x_train[0].max(), 1.)
        self.assertGreater(x_test[0].max(), 100.)
        np.testing.assert_array_equal(x_train[1], 0)
        np.testing.assert_allclose(y_test * scale, raw[3])
        np.testing.assert_allclose(y_train * scale, raw[1])
        self.assertEqual(boston.r2_score(np.ones((1, 3)), np.ones((1, 3))), 1.)
        self.assertEqual(boston.r2_score(np.zeros((1, 3)), np.ones((1, 3))), 0.)
