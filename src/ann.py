"""A feed-forward neural network (ANN) for binary classification, from scratch with NumPy.

Structure: input -> hidden layer(s) with ReLU -> one output neuron with sigmoid.
Training: mini-batch gradient descent, with gradients from backpropagation.
Loss: binary cross-entropy."""
import numpy as np


def _sigmoid(z):
    # clip so exp() never overflows
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class NeuralNetwork:
    def __init__(self, hidden_layers=(16, 8), learning_rate=0.05, seed=0):
        if len(hidden_layers) == 0 or any(h < 1 for h in hidden_layers):
            raise ValueError("hidden_layers must be a non-empty list of positive sizes")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.hidden_layers = tuple(hidden_layers)
        self.learning_rate = learning_rate
        self.seed = seed

    # ------------------------------------------------------------ setup
    def _init_weights(self, n_inputs):
        rng = np.random.RandomState(self.seed)
        sizes = [n_inputs, *self.hidden_layers, 1]
        self.weights, self.biases = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            # He initialisation works well with ReLU
            self.weights.append(rng.randn(n_in, n_out) * np.sqrt(2.0 / n_in))
            self.biases.append(np.zeros((1, n_out)))

    # ------------------------------------------------------------ forward pass
    def _forward(self, X):
        """Return the activations of every layer (the first is the input itself)."""
        activations = [X]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ W + b
            last = i == len(self.weights) - 1
            activations.append(_sigmoid(z) if last else np.maximum(0.0, z))
        return activations

    # ------------------------------------------------------------ backpropagation
    def _backward(self, activations, y):
        """Gradients of the loss with respect to every weight and bias."""
        n = len(y)
        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.weights)
        # For sigmoid + cross-entropy the output error simplifies to (prediction - truth)
        delta = (activations[-1] - y) / n
        for i in reversed(range(len(self.weights))):
            grads_W[i] = activations[i].T @ delta
            grads_b[i] = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.weights[i].T) * (activations[i] > 0)  # ReLU derivative
        return grads_W, grads_b

    @staticmethod
    def _loss(p, y):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    # ------------------------------------------------------------ public API
    def fit(self, X, y, epochs=200, batch_size=32, X_val=None, y_val=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (rows are examples)")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        if len(X) == 0:
            raise ValueError("there is no training data")
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("y must contain only 0 and 1")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be at least 1")

        self._init_weights(X.shape[1])
        rng = np.random.RandomState(self.seed)
        self.loss_history, self.val_loss_history = [], []
        self.train_acc_history, self.val_acc_history = [], []

        for _ in range(epochs):
            order = rng.permutation(len(X))  # shuffle every epoch
            for start in range(0, len(X), batch_size):
                idx = order[start:start + batch_size]
                acts = self._forward(X[idx])
                grads_W, grads_b = self._backward(acts, y[idx])
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * grads_W[i]
                    self.biases[i] -= self.learning_rate * grads_b[i]

            self.loss_history.append(self._loss(self._forward(X)[-1], y))
            self.train_acc_history.append(self.score(X, y.ravel()))
            if X_val is not None:
                yv = np.asarray(y_val, dtype=float).reshape(-1, 1)
                self.val_loss_history.append(self._loss(self._forward(np.asarray(X_val, float))[-1], yv))
                self.val_acc_history.append(self.score(X_val, yv.ravel()))
        return self

    def predict_proba(self, X):
        """Probability that each row belongs to class 1."""
        if not hasattr(self, "weights"):
            raise RuntimeError("call fit() before predicting")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.weights[0].shape[0]:
            raise ValueError(f"X must have {self.weights[0].shape[0]} features per row")
        return self._forward(X)[-1].ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        """Accuracy: the fraction of correct predictions."""
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))
