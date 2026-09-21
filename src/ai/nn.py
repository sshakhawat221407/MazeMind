"""A small feed-forward neural network written from first principles.

No autograd, no ML framework: every layer implements its own forward pass and
its own analytic gradient, the optimiser is a hand-rolled Adam, and the loss
functions return ``(value, dL/dprediction)`` explicitly. The point of the
project is to *apply* a learning technique rather than call one, so the
chain rule is spelled out rather than delegated.

Architecture used for the heuristic::

    16 features -> Dense(64) -> ReLU -> Dense(64) -> ReLU -> Dense(1) -> Softplus

Softplus on the output is a deliberate choice, not decoration: a heuristic must
never be negative, and ``log(1 + e^x)`` is smooth everywhere (unlike ReLU, whose
zero gradient on the negative side strands dead output units and would freeze
the prediction at exactly 0 for whole regions of the maze).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
class Layer:
    """Base class. ``backward`` receives dL/d(output) and returns dL/d(input)."""

    def forward(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def params(self) -> list[np.ndarray]:
        return []

    def grads(self) -> list[np.ndarray]:
        return []


class Dense(Layer):
    """Fully connected layer: ``y = x @ W + b``."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator) -> None:
        # He initialisation: variance 2/n_in keeps activation scale stable
        # through ReLU stacks, which halve the variance at every layer.
        self.W = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self._x is not None, "backward called before forward"
        # y = xW + b  =>  dL/dW = x^T (dL/dy), dL/db = sum(dL/dy), dL/dx = (dL/dy) W^T
        self.dW = self._x.T @ grad
        self.db = grad.sum(axis=0)
        return grad @ self.W.T

    def params(self) -> list[np.ndarray]:
        return [self.W, self.b]

    def grads(self) -> list[np.ndarray]:
        return [self.dW, self.db]


class ReLU(Layer):
    """``y = max(0, x)``; gradient passes through only where x was positive."""

    def __init__(self) -> None:
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._mask


class Softplus(Layer):
    """``y = log(1 + e^x)``, derivative ``sigma(x)``. Guarantees y > 0."""

    def __init__(self) -> None:
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        # logaddexp(0, x) is the overflow-safe form of log(1 + e^x).
        return np.logaddexp(0.0, x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self._x is not None
        # Stable logistic sigmoid, branch-free on the sign of x.
        return grad * (1.0 / (1.0 + np.exp(-np.clip(self._x, -60, 60))))


class Shift(Layer):
    """``y = x + c`` for a fixed constant. Gradient passes straight through.

    Stacked after Softplus this pins the network's output to ``(c, inf)``. With
    ``c = 1`` the model can only ever predict a detour ratio of at least 1,
    which is precisely the statement "never claim a route is shorter than the
    straight-line Manhattan bound" - so the learned heuristic is structurally
    incapable of being less informed than Manhattan.
    """

    def __init__(self, c: float = 1.0) -> None:
        self.c = float(c)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x + self.c

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad


# ---------------------------------------------------------------------------
# Losses  ->  (scalar loss, gradient w.r.t. predictions)
# ---------------------------------------------------------------------------
def mse_loss(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    diff = pred - target
    loss = float(np.mean(diff ** 2))
    return loss, (2.0 * diff) / pred.shape[0]


def pinball_loss(
    pred: np.ndarray, target: np.ndarray, tau: float
) -> tuple[float, np.ndarray]:
    """Quantile ("pinball") loss - the reason this heuristic stays usable in A*.

    With ``e = target - pred``::

        L = max(tau * e, (tau - 1) * e)

    Underestimating costs ``tau`` per unit; overestimating costs ``1 - tau``.
    At ``tau = 0.12`` an overestimate is punished about seven times harder than
    an underestimate, so the network converges onto roughly the 12th percentile
    of the true distance. A heuristic that *underestimates* is admissible and
    keeps A* optimal; one that overestimates silently breaks that guarantee.
    Ordinary MSE would sit on the mean and overestimate half the time.
    """
    e = target - pred
    loss = float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))
    # d/dpred:  -tau where e >= 0, (1 - tau) where e < 0.
    grad = np.where(e >= 0, -tau, 1.0 - tau) / pred.shape[0]
    return loss, grad


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------
class Adam:
    """Adam with bias correction, operating in-place on a list of arrays."""

    def __init__(
        self,
        params: Sequence[np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads: Sequence[np.ndarray]) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        # Bias-correction terms: without these the first steps are far too
        # small, because m and v start at zero.
        c1 = 1.0 - b1 ** self.t
        c2 = 1.0 - b2 ** self.t
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * (g * g)
            m_hat = self.m[i] / c1
            v_hat = self.v[i] / c2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class MLP:
    """Multilayer perceptron with input standardisation baked in."""

    def __init__(
        self,
        n_in: int,
        hidden: Iterable[int] = (64, 64),
        seed: int = 0,
        output_offset: float = 1.0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.n_in = n_in
        self.hidden = tuple(hidden)
        self.output_offset = float(output_offset)
        self.layers: list[Layer] = []
        prev = n_in
        for size in self.hidden:
            self.layers.append(Dense(prev, size, rng))
            self.layers.append(ReLU())
            prev = size
        self.layers.append(Dense(prev, 1, rng))
        self.layers.append(Softplus())
        if self.output_offset:
            self.layers.append(Shift(self.output_offset))
        # Feature standardisation, filled in by ``fit_normaliser``.
        self.mu = np.zeros(n_in)
        self.sigma = np.ones(n_in)
        # Admissibility calibration, fitted on held-out data in train.py.
        # Applied as  ratio_safe = 1 + (ratio_pred - 1) * alpha,  which shrinks
        # the prediction back toward plain Manhattan. alpha = 1 leaves the model
        # untouched; alpha = 0 collapses it to Manhattan. Because the shrink is
        # anchored at 1 rather than at 0, no value of alpha can ever make the
        # heuristic weaker than the Manhattan baseline.
        self.alpha = 1.0

    # -- normalisation -----------------------------------------------------
    def fit_normaliser(self, x: np.ndarray) -> None:
        self.mu = x.mean(axis=0)
        # Guard against constant features (sigma == 0 would divide by zero).
        self.sigma = np.where(x.std(axis=0) < 1e-8, 1.0, x.std(axis=0))

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sigma

    # -- forward / backward ------------------------------------------------
    def forward(self, x: np.ndarray, standardise: bool = True) -> np.ndarray:
        out = self._standardise(x) if standardise else x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self) -> list[np.ndarray]:
        return [p for layer in self.layers for p in layer.params()]

    def gradients(self) -> list[np.ndarray]:
        return [g for layer in self.layers for g in layer.grads()]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Inference: returns a flat array of non-negative predictions."""
        if x.ndim == 1:
            x = x[None, :]
        return self.forward(x).ravel()

    # -- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        blob: dict[str, np.ndarray] = {
            "mu": self.mu,
            "sigma": self.sigma,
            "hidden": np.array(self.hidden, dtype=np.int64),
            "n_in": np.array([self.n_in], dtype=np.int64),
            "alpha": np.array([self.alpha]),
            "output_offset": np.array([self.output_offset]),
        }
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                blob[f"W{idx}"] = layer.W
                blob[f"b{idx}"] = layer.b
                idx += 1
        np.savez_compressed(path, **blob)

    @classmethod
    def load(cls, path: str) -> "MLP":
        data = np.load(path)
        model = cls(
            n_in=int(data["n_in"][0]),
            hidden=tuple(int(v) for v in data["hidden"]),
            output_offset=float(data["output_offset"][0]),
        )
        model.mu = data["mu"]
        model.sigma = data["sigma"]
        model.alpha = float(data["alpha"][0])
        idx = 0
        for layer in model.layers:
            if isinstance(layer, Dense):
                layer.W = data[f"W{idx}"]
                layer.b = data[f"b{idx}"]
                idx += 1
        return model


# ---------------------------------------------------------------------------
# Gradient check - the honest way to prove the backprop above is correct
# ---------------------------------------------------------------------------
def gradient_check(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    tau: float = 0.5,
    eps: float = 1e-5,
) -> float:
    """Compare analytic gradients against central finite differences.

    Returns the largest relative error across a sample of weights. Anything
    below ~1e-5 means the hand-derived chain rule matches numerical reality.
    Uses ``tau = 0.5`` because the pinball loss is kinked at ``e = 0`` and a
    symmetric quantile keeps the perturbation away from that kink.
    """
    def loss_of(params_snapshot: None = None) -> float:
        pred = model.forward(x)
        return pinball_loss(pred, y, tau)[0]

    pred = model.forward(x)
    _, dpred = pinball_loss(pred, y, tau)
    model.backward(dpred)

    analytic = model.gradients()
    params = model.parameters()

    rng = np.random.default_rng(0)
    worst = 0.0
    for p, g in zip(params, analytic):
        flat_p = p.ravel()
        flat_g = g.ravel()
        picks = rng.choice(flat_p.size, size=min(12, flat_p.size), replace=False)
        for i in picks:
            original = flat_p[i]
            flat_p[i] = original + eps
            plus = loss_of()
            flat_p[i] = original - eps
            minus = loss_of()
            flat_p[i] = original
            numeric = (plus - minus) / (2 * eps)
            denom = max(1e-8, abs(numeric) + abs(flat_g[i]))
            worst = max(worst, abs(numeric - flat_g[i]) / denom)
    return worst
