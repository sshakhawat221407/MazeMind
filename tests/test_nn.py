"""The from-scratch neural network: gradients, shapes, and persistence."""

import numpy as np
import pytest

from src.ai.features import feature_grid, manhattan_grid
from src.ai.nn import (
    MLP,
    Adam,
    Dense,
    ReLU,
    Shift,
    Softplus,
    gradient_check,
    mse_loss,
    pinball_loss,
)
from src.config import N_FEATURES
from src.core.generator import generate


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(48, N_FEATURES))
    y = 1.0 + np.abs(rng.normal(size=(48, 1)))
    return x, y


# ---------------------------------------------------------------------------
# Gradients - the property that matters most
# ---------------------------------------------------------------------------
def test_analytic_gradients_match_finite_differences(sample):
    x, y = sample
    model = MLP(N_FEATURES, (12, 8), seed=1)
    model.fit_normaliser(x)
    error = gradient_check(model, x[:16], y[:16])
    assert error < 1e-5, f"backprop disagrees with numeric gradient: {error:.2e}"


def test_dense_backward_shapes():
    rng = np.random.default_rng(2)
    layer = Dense(5, 3, rng)
    x = rng.normal(size=(7, 5))
    out = layer.forward(x)
    assert out.shape == (7, 3)
    grad = layer.backward(np.ones((7, 3)))
    assert grad.shape == (7, 5)
    assert layer.dW.shape == (5, 3)
    assert layer.db.shape == (3,)


def test_relu_blocks_negative_gradient():
    layer = ReLU()
    x = np.array([[-2.0, 3.0]])
    assert np.array_equal(layer.forward(x), [[0.0, 3.0]])
    assert np.array_equal(layer.backward(np.ones((1, 2))), [[0.0, 1.0]])


def test_softplus_is_strictly_positive_and_stable():
    layer = Softplus()
    x = np.array([[-800.0, 0.0, 800.0]])
    out = layer.forward(x)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0)
    assert out[0, 1] == pytest.approx(np.log(2))
    # Large positive input must not overflow; softplus(x) -> x.
    assert out[0, 2] == pytest.approx(800.0)


def test_shift_passes_gradient_through():
    layer = Shift(1.0)
    assert layer.forward(np.zeros((2, 1)))[0, 0] == 1.0
    assert layer.backward(np.full((2, 1), 3.0))[0, 0] == 3.0


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def test_pinball_penalises_overestimates_harder():
    over = pinball_loss(np.array([[2.0]]), np.array([[1.0]]), 0.25)[0]
    under = pinball_loss(np.array([[0.0]]), np.array([[1.0]]), 0.25)[0]
    assert over > under
    assert over / under == pytest.approx(0.75 / 0.25)


def test_pinball_is_zero_at_a_perfect_prediction():
    loss, grad = pinball_loss(np.array([[1.5]]), np.array([[1.5]]), 0.3)
    assert loss == pytest.approx(0.0)
    assert grad.shape == (1, 1)


def test_mse_gradient_direction():
    loss, grad = mse_loss(np.array([[3.0]]), np.array([[1.0]]))
    assert loss == pytest.approx(4.0)
    assert grad[0, 0] > 0  # overestimating => push the prediction down


# ---------------------------------------------------------------------------
# Model behaviour
# ---------------------------------------------------------------------------
def test_output_offset_floors_predictions_at_one(sample):
    x, _ = sample
    model = MLP(N_FEATURES, (8,), seed=3, output_offset=1.0)
    model.fit_normaliser(x)
    assert np.all(model.predict(x) >= 1.0)


def test_normaliser_handles_constant_features():
    x = np.ones((10, N_FEATURES))  # zero variance everywhere
    model = MLP(N_FEATURES, (4,), seed=0)
    model.fit_normaliser(x)
    assert np.all(np.isfinite(model._standardise(x)))


def test_training_reduces_the_loss(sample):
    x, y = sample
    model = MLP(N_FEATURES, (16, 16), seed=4)
    model.fit_normaliser(x)
    opt = Adam(model.parameters(), lr=5e-3)
    first = pinball_loss(model.forward(x), y, 0.25)[0]
    for _ in range(150):
        pred = model.forward(x)
        _, dpred = pinball_loss(pred, y, 0.25)
        model.backward(dpred)
        opt.step(model.gradients())
    last = pinball_loss(model.forward(x), y, 0.25)[0]
    assert last < first


def test_adam_bias_correction_moves_on_the_first_step():
    param = np.zeros(3)
    opt = Adam([param], lr=0.1)
    opt.step([np.ones(3)])
    # Without bias correction the first step would be far smaller than lr.
    assert np.allclose(param, -0.1, atol=1e-6)


def test_save_and_load_roundtrip(tmp_path, sample):
    x, _ = sample
    model = MLP(N_FEATURES, (12, 6), seed=5)
    model.fit_normaliser(x)
    model.alpha = 0.42
    before = model.predict(x)

    path = str(tmp_path / "weights.npz")
    model.save(path)
    restored = MLP.load(path)

    assert restored.alpha == pytest.approx(0.42)
    assert restored.hidden == (12, 6)
    assert np.allclose(restored.predict(x), before)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def test_feature_grid_shape_and_finiteness():
    maze = generate(21, 15, seed=9)
    grid = feature_grid(maze, (20, 14))
    assert grid.shape == (maze.area, N_FEATURES)
    assert np.all(np.isfinite(grid))


def test_features_for_cell_matches_the_grid():
    maze = generate(15, 11, seed=10)
    goal = (14, 10)
    grid = feature_grid(maze, goal)
    from src.ai.features import features_for_cell

    single = features_for_cell(maze, (3, 4), goal)
    assert np.allclose(single, grid[4 * maze.width + 3])


def test_manhattan_grid_is_zero_at_the_goal():
    maze = generate(11, 9, seed=11)
    goal = (5, 4)
    grid = manhattan_grid(maze, goal)
    assert grid[goal[1], goal[0]] == 0
    assert grid[0, 0] == 5 + 4
