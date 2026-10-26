import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from src.ann import NeuralNetwork
from src.data import load_data, standardize


def test_learns_xor():
    # XOR cannot be solved without a hidden layer
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 25, dtype=float)
    y = np.array([0, 1, 1, 0] * 25)
    model = NeuralNetwork(hidden_layers=(8,), learning_rate=0.5, seed=1)
    model.fit(X, y, epochs=300, batch_size=16)
    assert model.score(X, y) == 1.0


def test_loss_goes_down():
    X, y, _ = load_data()
    X, _ = standardize(X, X)
    model = NeuralNetwork(seed=0).fit(X, y, epochs=30)
    assert model.loss_history[-1] < model.loss_history[0]


def test_probabilities_between_0_and_1():
    X, y, _ = load_data()
    X, _ = standardize(X, X)
    model = NeuralNetwork(seed=0).fit(X, y, epochs=5)
    p = model.predict_proba(X)
    assert p.shape == (len(X),)
    assert np.all((p >= 0) & (p <= 1))


def test_gradients_match_numerical_estimate():
    # Backpropagation is correct if it agrees with a slow finite-difference estimate
    rng = np.random.RandomState(0)
    X = rng.randn(6, 3)
    y = np.array([[0], [1], [0], [1], [1], [0]], dtype=float)
    model = NeuralNetwork(hidden_layers=(4,), seed=0)
    model._init_weights(3)
    grads_W, _ = model._backward(model._forward(X), y)
    eps = 1e-6
    W = model.weights[0]
    for (i, j) in [(0, 0), (1, 2), (2, 3)]:
        old = W[i, j]
        W[i, j] = old + eps
        up = model._loss(model._forward(X)[-1], y)
        W[i, j] = old - eps
        down = model._loss(model._forward(X)[-1], y)
        W[i, j] = old
        assert abs((up - down) / (2 * eps) - grads_W[0][i, j]) < 1e-5


def test_good_accuracy_on_breast_cancer():
    X, y, _ = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_te = standardize(X_tr, X_te)
    model = NeuralNetwork(seed=0).fit(X_tr, y_tr, epochs=100)
    assert model.score(X_te, y_te) > 0.93


def test_invalid_inputs():
    with pytest.raises(ValueError):
        NeuralNetwork(hidden_layers=())
    with pytest.raises(ValueError):
        NeuralNetwork(learning_rate=0)
    with pytest.raises(ValueError):
        NeuralNetwork().fit([[1.0], [2.0]], [0])
    with pytest.raises(ValueError):
        NeuralNetwork().fit([[1.0], [2.0]], [0, 2])
    with pytest.raises(RuntimeError):
        NeuralNetwork().predict([[1.0]])
    model = NeuralNetwork().fit([[1.0, 2.0], [3.0, 4.0]], [0, 1], epochs=1)
    with pytest.raises(ValueError):
        model.predict([[1.0]])
