"""Unit tests for `KingLoss` and `StudentTLoss`."""

import numpy as np
import pytest
import scipy.stats
import torch

from graphnet.training.loss_functions import KingLoss, StudentTLoss


def _directions(angles: np.ndarray) -> torch.Tensor:
    """Return unit vectors at `angles` (radians) from the z axis."""
    return torch.tensor(
        np.stack(
            [np.sin(angles), np.zeros_like(angles), np.cos(angles)], axis=1
        ),
        dtype=torch.float64,
    )


def _king(k: float, gamma: float, angles: np.ndarray) -> torch.Tensor:
    """King loss of predictions along z for targets at `angles`."""
    n = len(angles)
    prediction = torch.zeros(n, 5, dtype=torch.float64)
    prediction[:, 2] = 1.0
    prediction[:, 3] = k
    prediction[:, 4] = gamma
    return KingLoss()(prediction, _directions(angles), return_elements=True)


@pytest.mark.parametrize("k", [0.3, 3.0, 50.0])
@pytest.mark.parametrize("gamma", [1.1, 2.0, 5.0, 40.0])
def test_king_is_normalized(k: float, gamma: float) -> None:
    """The density integrates to one over the sphere."""
    # integrate over u = 1 - cos(angle) in [0, 2], dense near 0
    u = np.concatenate([[0.0], np.geomspace(1e-12, 2.0, 200001)])
    density = np.exp(-_king(k, gamma, np.arccos(1 - u)).double().numpy())
    integral = 2 * np.pi * np.trapz(density, u)
    assert integral == pytest.approx(1.0, rel=2e-3)


def test_king_with_gamma_2_is_the_spherical_cauchy() -> None:
    """For gamma = 2 the loss is the spherical Cauchy NLL."""
    angles = np.radians([0.0, 0.01, 0.3, 5.0, 40.0, 179.0])
    for k in (0.5, 7.0, 400.0):
        rho = k / (1 + k)
        dot = np.cos(angles)
        # f = (1 - rho^2)^2 / (4 pi (1 + rho^2 - 2 rho dot)^2) in 3D
        expected = -np.log(
            (1 - rho**2) ** 2 / (4 * np.pi * (1 + rho**2 - 2 * rho * dot) ** 2)
        )
        assert _king(k, 2.0, angles).numpy() == pytest.approx(
            expected, rel=1e-6, abs=1e-6
        )


def test_king_gradients_are_finite_at_extremes() -> None:
    """Value and gradients stay finite for extreme k, gamma and angles."""
    angles = torch.tensor([0.0, 1e-5, 1e-3, 1.0, 3.1])
    target = torch.stack(
        [torch.sin(angles), torch.zeros_like(angles), torch.cos(angles)], 1
    )
    for k in (0.0, 1e-6, 1e6):
        for gamma in (1.0 + 1e-6, 1.1, 1e3):
            params = torch.tensor([k, gamma]).repeat(len(angles), 1)
            params.requires_grad_(True)
            prediction = torch.cat(
                [
                    torch.tensor([[0.0, 0.0, 1.0]]).repeat(len(angles), 1),
                    params,
                ],
                dim=1,
            )
            loss = KingLoss()(prediction, target)
            loss.backward()
            assert torch.isfinite(loss)
            assert torch.isfinite(params.grad).all()


@pytest.mark.parametrize("nu", [0.7, 1.0, 3.0, 30.0])
def test_student_t_matches_scipy(nu: float) -> None:
    """The loss is the negative Student-t log-density."""
    target = torch.tensor([[0.0], [0.3], [-2.0], [10.0]], dtype=torch.float64)
    value, scale = 0.1, 0.5
    prediction = torch.tensor([[value, scale, nu]] * 4, dtype=torch.float64)
    loss = StudentTLoss()(prediction, target, return_elements=True)
    expected = -scipy.stats.t.logpdf(
        target[:, 0].numpy(), nu, loc=value, scale=scale
    )
    assert loss.numpy() == pytest.approx(expected, rel=1e-6)
