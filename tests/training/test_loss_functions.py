"""Unit tests for LossFunction classes."""

import numpy as np
from typing import Any

import pytest
import torch
import warnings
from torch import Tensor
from torch.autograd import grad

from graphnet.training.loss_functions import (
    BinaryCrossEntropyLoss,
    CauchyLoss,
    EnsembleLoss,
    FocalBinaryCrossEntropyLoss,
    LogCoshLoss,
    LossFunction,
    MAELoss,
    MSELoss,
    NegCosLoss,
    RMSELoss,
    spCauchyLoss,
    VonMisesFisherLoss,
    VonMisesFisher3DLoss,
)
from graphnet.utilities.maths import eps_like


# Utility method(s)
def _compute_elementwise_gradient(outputs: Tensor, inputs: Tensor) -> Tensor:
    """Compute gradient of each element in `outptus` wrt. `inputs`.

    It is assumed that each element in `inputs` only affects the corresponding
    element in `outputs`. This should be the result of any vectorised
    calculation (as used in tests).
    """
    # Check(s)
    assert inputs.dim() == 1
    assert outputs.dim() == 1
    assert inputs.size(dim=0) == outputs.size(dim=0)

    # Compute list of elementwise gradients
    nb_elements = inputs.size(dim=0)
    elementwise_gradients = torch.stack(
        [
            grad(
                outputs=outputs[ix],
                inputs=inputs,
                retain_graph=True,
            )[
                0
            ][ix]
            for ix in range(nb_elements)
        ]
    )
    return elementwise_gradients


# Unit test(s)
def test_log_cosh(dtype: torch.dtype = torch.float32) -> None:
    """Test agreement of the two ways to calculate this loss."""
    # Prepare test data
    x = torch.tensor([-100, -10, -1, 0, 1, 10, 100], dtype=dtype).unsqueeze(
        1
    )  # Shape [N, 1]
    y = 0.0 * x.clone()  # Shape [N,1]
    # Calculate losses using loss function, and manually
    log_cosh_loss = LogCoshLoss()
    losses = log_cosh_loss(x, y, return_elements=True)
    losses_reference = torch.log(torch.cosh(x - y))

    # (1) Loss functions should not return  `inf` losses, even for large
    #     differences between prediction and target. This is not necessarily
    #     true for the directly calculated loss (reference) where cosh(x)
    #     may go to `inf` for x >~ 100.
    assert torch.all(torch.isfinite(losses))

    # (2) For the inputs where the reference loss _is_ valid, the two
    #     calculations should agree exactly.
    reference_is_valid = torch.isfinite(losses_reference)
    assert torch.allclose(
        losses_reference[reference_is_valid], losses[reference_is_valid]
    )


def test_von_mises_fisher_exact_m3(dtype: torch.dtype = torch.float64) -> None:
    """Test implementaion of exact von-Mises Fisher loss with m=3.

    See
    https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
    for exact, simplified reference.
    """
    # Define test parameters
    m = 3
    k = torch.tensor(
        data=[0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
        requires_grad=True,
        dtype=dtype,
    )

    # Compute values
    res_reference = (
        torch.log(k) - k - torch.log(2 * np.pi * (1 - torch.exp(-2 * k)))
    )
    res_exact = VonMisesFisherLoss.log_cmk_exact(m, k)

    # Compute gradients
    grads_reference = _compute_elementwise_gradient(res_reference, k)
    grads_exact = _compute_elementwise_gradient(res_exact, k)

    # Test that values agree
    assert torch.allclose(res_exact, res_reference)

    # Test that gradients agree
    assert torch.allclose(grads_reference, grads_exact)


@pytest.mark.parametrize("m", [2, 3])
def test_von_mises_fisher_approximation(
    m: int, dtype: torch.dtype = torch.float64
) -> None:
    """Test approximate calculation of $log C_{m}(k)$.

    See [1812.04616], Section 8.2 for approximation.
    """
    # Check(s)
    assert isinstance(m, int)
    assert m > 1

    # Define test parameters
    k = torch.tensor(
        data=[0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
        requires_grad=True,
        dtype=dtype,
    )

    # Compute values
    res_approx = VonMisesFisherLoss.log_cmk_approx(m, k)
    res_exact = VonMisesFisherLoss.log_cmk_exact(m, k)

    C = (
        res_exact[0] - res_approx[0]
    )  # Normalisation constant from integrating gradient
    res_approx += C - eps_like(C)

    # Compute gradients
    grads_approx = _compute_elementwise_gradient(res_approx, k)
    grads_exact = _compute_elementwise_gradient(res_exact, k)

    # Test inequality in [1812.04616] Sec. 8.2
    assert torch.all(res_exact >= res_approx), (m, res_exact, res_approx)

    # Test value approximation
    assert torch.allclose(res_approx, res_exact, rtol=1e0, atol=1e-01)

    # Test gradient approximation
    assert torch.allclose(grads_approx, grads_exact, rtol=1e0)


@pytest.mark.parametrize("m", [2, 3])
def test_von_mises_fisher_approximation_large_kappa(
    m: int, dtype: torch.dtype = torch.float64
) -> None:
    """Test approximate calculation of $log C_{m}(k)$ for large kappa values.

    See [1812.04616], Section 8.2 for approximation.
    """
    # Check(s)
    assert isinstance(m, int)
    assert m > 1

    # Define test parameters
    k = torch.tensor(
        data=[100.0, 200.0, 300.0, 500.0, 1000.0],
        requires_grad=True,
        dtype=dtype,
    )

    # Compute values
    res_approx = VonMisesFisherLoss.log_cmk_approx(m, k)
    res_exact = VonMisesFisherLoss.log_cmk_exact(m, k)

    C = res_exact[0] - res_approx[0]  # Normalisation constant
    res_approx += C

    # Compute gradients
    grads_approx = _compute_elementwise_gradient(res_approx, k)
    grads_exact = _compute_elementwise_gradient(res_exact, k)

    exact_is_valid = torch.isfinite(res_exact)

    # Test value approximation
    assert torch.allclose(
        res_approx[exact_is_valid], res_exact[exact_is_valid], rtol=1e-2
    )

    # Test gradient approximation
    assert torch.allclose(
        grads_approx[exact_is_valid], grads_exact[exact_is_valid], rtol=1e-2
    )


def test_logcmk_backward_zero_handling(
    dtype: torch.dtype = torch.float64,
) -> None:
    """Test LogCMK backward pass handles arrays with zero values correctly.

    This test ensures that the LogCMK.backward method correctly handles cases
    where the kappa tensor contains zero values without raising division by zero
    errors or warnings. The implementation uses boolean masking to conditionally
    apply different formulas for small (including zero) and large kappa values,
    avoiding double evaluation that would cause RuntimeWarnings.

    Args:
        dtype: PyTorch data type for the test tensors.
    """
    # Test parameters
    m = 3  # Dimension for which we have the exact formula

    # Create kappa tensor with zeros and other values, including edge cases
    kappa_values = [0.0, 1e-7, 1e-6, 1e-5, 0.1, 1.0, 10.0]
    kappa = torch.tensor(kappa_values, dtype=dtype, requires_grad=True)

    # Forward pass using VonMisesFisherLoss.log_cmk_exact which internally uses LogCMK
    result = VonMisesFisherLoss.log_cmk_exact(m, kappa)

    # Test that backward pass doesn't raise any errors or warnings
    # Capture warnings to ensure no RuntimeWarnings are generated
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        try:
            grads = torch.autograd.grad(
                outputs=result.sum(),
                inputs=kappa,
                grad_outputs=None,
                retain_graph=False,
                create_graph=False,
            )[0]
            backward_success = True
            error_msg = ""
        except (ZeroDivisionError, RuntimeWarning) as e:
            backward_success = False
            error_msg = str(e)

    # Verify no errors occurred
    assert backward_success, f"Backward pass failed with error: {error_msg}"

    # Verify no RuntimeWarnings were generated
    runtime_warnings = [
        w for w in caught_warnings if issubclass(w.category, RuntimeWarning)
    ]
    assert (
        len(runtime_warnings) == 0
    ), f"RuntimeWarnings were generated: {[str(w.message) for w in runtime_warnings]}"

    # Verify gradients are finite
    assert torch.all(
        torch.isfinite(grads)
    ), "Gradients should be finite for all kappa values"

    # Test specific values for correctness
    # For kappa=0, the gradient should be -kappa/3 = 0
    zero_idx = 0  # Index where kappa=0
    assert torch.isclose(
        grads[zero_idx], torch.tensor(0.0, dtype=dtype)
    ), f"Gradient at kappa=0 should be 0, got {grads[zero_idx]}"

    # For very small kappa (1e-7), should use -kappa/3 approximation
    small_kappa_idx = 1  # Index where kappa=1e-7
    expected_small_grad = -kappa_values[small_kappa_idx] / 3
    assert torch.isclose(
        grads[small_kappa_idx],
        torch.tensor(expected_small_grad, dtype=dtype),
        atol=1e-10,
    ), "Gradient for small kappa should use -kappa/3 approximation"

    # Test with array containing multiple zeros
    kappa_multi_zero = torch.tensor(
        [0.0, 0.0, 1.0, 0.0, 10.0], dtype=dtype, requires_grad=True
    )
    result_multi = VonMisesFisherLoss.log_cmk_exact(m, kappa_multi_zero)

    with warnings.catch_warnings(record=True) as caught_warnings_multi:
        warnings.simplefilter("always")
        try:
            grads_multi = torch.autograd.grad(
                outputs=result_multi.sum(),
                inputs=kappa_multi_zero,
                grad_outputs=None,
            )[0]
            multi_zero_success = True
        except (ZeroDivisionError, RuntimeWarning):
            multi_zero_success = False

    assert multi_zero_success, "Should handle arrays with multiple zero values"
    assert torch.all(
        torch.isfinite(grads_multi)
    ), "All gradients should be finite with multiple zeros"

    # Verify no RuntimeWarnings for multiple zeros case
    runtime_warnings_multi = [
        w
        for w in caught_warnings_multi
        if issubclass(w.category, RuntimeWarning)
    ]
    assert (
        len(runtime_warnings_multi) == 0
    ), f"RuntimeWarnings were generated with multiple zeros: {[str(w.message) for w in runtime_warnings_multi]}"

    # Verify that zero elements have zero gradients
    zero_mask = kappa_multi_zero == 0.0
    assert torch.all(
        grads_multi[zero_mask] == 0.0
    ), "Zero kappa values should have zero gradients"


def test_vmf3d_log_cmk_closed_form(dtype: torch.dtype = torch.float64) -> None:
    """Test the vMF-3D normalizer against the m=3 closed form.

    `log C_3(k) = _log_cmk_scaled(k, eps) - k` is checked in value against
    the reference log(k) - k - log(2 pi (1 - exp(-2k))) and in gradient
    against d/dk log C_3(k) = 1/k - coth(k).
    """
    k = torch.tensor(
        data=[0.1, 0.5, 1.0, 3.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
        requires_grad=True,
        dtype=dtype,
    )

    res = VonMisesFisher3DLoss._log_cmk_scaled(k, eps=1e-8) - k
    res_reference = (
        torch.log(k) - k - torch.log(2 * np.pi * (1 - torch.exp(-2 * k)))
    )
    assert torch.allclose(res, res_reference, atol=1e-6)

    grads = _compute_elementwise_gradient(res, k)
    grads_reference = 1 / k - 1 / torch.tanh(k)
    assert torch.allclose(grads, grads_reference, atol=1e-6)


def test_vmf3d_log_cmk_small_kappa(dtype: torch.dtype = torch.float64) -> None:
    """Test the vMF-3D normalizer at and near kappa = 0.

    The uniform-distribution limit is log(C_3(0)) = -log(4 pi); values
    must hit it and both values and gradients must stay finite all the
    way to 0.
    """
    k = torch.tensor(
        data=[0.0, 1e-8, 1e-4, 1e-2],
        requires_grad=True,
        dtype=dtype,
    )

    res = VonMisesFisher3DLoss._log_cmk_scaled(k, eps=1e-8)
    assert torch.all(torch.isfinite(res))
    assert torch.isclose(
        res[0], torch.tensor(-np.log(4 * np.pi), dtype=dtype), atol=1e-9
    )
    assert torch.allclose(
        res, torch.full_like(res, -np.log(4 * np.pi)), atol=2e-2
    )

    grads = _compute_elementwise_gradient(res, k)
    assert torch.all(torch.isfinite(grads))
    # d/dk log(exp(k) C_3(k)) = 1 - A_3(k) -> 1 - k/3 for small k; only
    # checked where the eps-regularization error eps/k**2 is subdominant.
    assert torch.isclose(
        grads[3], torch.tensor(1 - 1e-2 / 3, dtype=dtype), atol=1e-3
    )


def test_vmf3d_log_cmk_matches_bessel_path(
    dtype: torch.dtype = torch.float64,
) -> None:
    """Test the closed-form normalizer against the Bessel-based path.

    `_log_cmk_scaled(k, eps) - k` and `log_cmk_exact(3, k)` are independent
    implementations of log C_3(k) (elementary closed form vs.
    `scipy.special.iv`), so agreement validates both.
    """
    k = torch.tensor(
        data=[0.1, 1.0, 10.0, 100.0, 500.0],
        requires_grad=True,
        dtype=dtype,
    )

    res = VonMisesFisher3DLoss._log_cmk_scaled(k, eps=1e-8) - k
    res_bessel = VonMisesFisherLoss.log_cmk_exact(3, k)
    assert torch.allclose(res, res_bessel, atol=1e-6)

    grads = _compute_elementwise_gradient(res, k)
    grads_bessel = _compute_elementwise_gradient(res_bessel, k)
    assert torch.allclose(grads, grads_bessel, atol=1e-6)


def test_vmf3d_loss_elements(dtype: torch.dtype = torch.float64) -> None:
    """Test `VonMisesFisher3DLoss` elements against the plain NLL formula."""
    torch.manual_seed(0)
    n = 64
    direction = torch.randn(n, 3, dtype=dtype)
    direction = direction / direction.norm(dim=1, keepdim=True)
    target = torch.randn(n, 3, dtype=dtype)
    target = target / target.norm(dim=1, keepdim=True)
    kappa = 10 ** (torch.rand(n, dtype=dtype) * 4 - 1)  # 0.1 ... 1000

    prediction = torch.cat([direction, kappa.unsqueeze(1)], dim=1)
    elements = VonMisesFisher3DLoss()._forward(prediction, target)

    log_cmk_reference = (
        torch.log(kappa)
        - kappa
        - torch.log(2 * np.pi * (1 - torch.exp(-2 * kappa)))
    )
    dotprod = (kappa.unsqueeze(1) * direction * target).sum(dim=1)
    elements_reference = -log_cmk_reference - dotprod
    assert torch.allclose(elements, elements_reference, atol=1e-6)


def test_vmf3d_loss_no_gradient_dead_zone(
    dtype: torch.dtype = torch.float64,
) -> None:
    """Test that the kappa-gradient has a stationary point for any alignment.

    The optimum concentration satisfies A_3(kappa) = coth(kappa) - 1/kappa =
    cos(theta). A derivative that is discontinuous somewhere in kappa leaves
    a band of cos(theta) values with no stationary point, so gradient descent
    pins kappa at the discontinuity for those events. The alignments below
    place the optimum at kappa ~ 100...20000.
    """
    loss = VonMisesFisher3DLoss()
    for cos_theta in [0.9905, 0.995, 0.999, 0.99995]:
        sin_theta = float(np.sqrt(1 - cos_theta**2))
        kappa = torch.logspace(1, 5, 500, dtype=dtype, requires_grad=True)
        n = kappa.size(0)
        direction = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype).repeat(n, 1)
        target = torch.tensor(
            [[sin_theta, 0.0, cos_theta]], dtype=dtype
        ).repeat(n, 1)

        prediction = torch.cat([direction, kappa.unsqueeze(1)], dim=1)
        elements = loss._forward(prediction, target)
        elements.sum().backward()
        grads = kappa.grad

        # A_3 is monotone, so d(loss)/d(kappa) = A_3(kappa) - cos(theta)
        # must cross zero exactly once on a grid spanning the optimum.
        signs = torch.sign(grads)
        flips = (signs[1:] != signs[:-1]).nonzero().flatten()
        assert len(flips) == 1, (cos_theta, len(flips))

        kappa_star = kappa[flips[0]].detach()
        a3 = 1 / torch.tanh(kappa_star) - 1 / kappa_star
        assert torch.isclose(
            a3, torch.tensor(cos_theta, dtype=dtype), atol=1e-3
        )


def test_vmf3d_loss_fp32_high_kappa_small_angle() -> None:
    """Test float32 precision of the loss at high kappa and small angles.

    The angular part of the loss is kappa * (1 - cos(theta)); computing it
    via the squared chord ||t - mu||**2 keeps it accurate in float32 even
    when 1 - cos(theta) is comparable to the float32 epsilon.
    """
    theta = 1e-3
    kappa_value = 1e4
    direction = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    kappa = torch.tensor([[kappa_value]], dtype=torch.float32)
    prediction = torch.cat([direction, kappa], dim=1)
    target_aligned = direction.clone()
    target_off = torch.tensor(
        [[np.sin(theta), 0.0, np.cos(theta)]], dtype=torch.float32
    )

    loss = VonMisesFisher3DLoss()
    angular_part = (
        loss._forward(prediction, target_off)
        - loss._forward(prediction, target_aligned)
    ).double()

    expected = kappa_value * (1 - np.cos(theta))
    assert torch.isclose(
        angular_part,
        torch.tensor([expected], dtype=torch.float64),
        rtol=1e-3,
    )

    # Extreme concentrations must not overflow values or gradients.
    prediction_extreme = torch.tensor(
        [[0.0, 0.0, 1.0, 1e7]], dtype=torch.float32, requires_grad=True
    )
    elements = loss._forward(prediction_extreme, target_off)
    assert torch.all(torch.isfinite(elements))
    elements.sum().backward()
    assert torch.all(torch.isfinite(prediction_extreme.grad))


@pytest.mark.parametrize("from_logits", [True, False])
def test_focal_bce_reduces_to_weighted_bce(from_logits: bool) -> None:
    """With gamma=0 the focal loss is the alpha-weighted BCE."""
    torch.manual_seed(0)
    logits = torch.randn(64, 1)
    prediction = logits if from_logits else torch.sigmoid(logits)
    target = torch.randint(0, 2, (64, 1)).float()
    alpha = 0.3

    focal = FocalBinaryCrossEntropyLoss(
        gamma=0.0, alpha=alpha, from_logits=from_logits
    )
    bce = BinaryCrossEntropyLoss(from_logits=from_logits)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    expected = alpha_t * bce(prediction, target, return_elements=True)

    assert torch.allclose(
        focal(prediction, target, return_elements=True), expected, atol=1e-6
    )


def test_cauchy_homoscedastic_closed_form() -> None:
    """With frac=0 the loss is log(1 + (r/alpha)^2) + log(alpha)."""
    alpha = 0.5
    prediction = torch.tensor([[0.0, 1.0], [2.0, -1.0]])
    target = torch.tensor([[0.5, 1.0], [0.0, 0.0]])
    residual = (prediction - target).abs()
    expected = torch.mean(
        torch.log1p((residual / alpha) ** 2) + np.log(alpha), dim=-1
    )

    loss = CauchyLoss(alpha=alpha, frac=0.0)
    assert torch.allclose(
        loss(prediction, target, return_elements=True), expected
    )


@pytest.mark.parametrize("k", [0.3, 2.0, 10.0])
def test_sp_cauchy_density_is_normalized(k: float) -> None:
    """Exp(-loss) / (4 pi) integrates to one over the sphere (d = 3)."""
    n = 400_000
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = 1 - 2 * i / n
    phi = torch.pi * (1 + 5**0.5) * i
    r = torch.sqrt(1 - z**2)
    points = torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=1)
    mu = torch.tensor([0.0, 0.0, 1.0]).expand(n, 3)
    prediction = torch.cat([mu, torch.full((n, 1), k)], dim=1)

    loss = spCauchyLoss()(prediction, points.float(), return_elements=True)
    integral = torch.exp(-loss.double()).mean()  # mean over uniform points
    assert torch.isclose(
        integral, torch.tensor(1.0, dtype=torch.float64), rtol=1e-3
    )


def test_sp_cauchy_large_concentration_is_finite() -> None:
    """Very confident predictions give finite losses and gradients."""
    target = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    prediction = torch.tensor(
        [[0.0, 0.0, 1.0, 1e8], [0.0, 0.0, 1.0, 1e8]], requires_grad=True
    )
    loss = spCauchyLoss()(prediction, target, return_elements=True)
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert torch.isfinite(prediction.grad).all()
    assert loss[0] < loss[1]


@pytest.mark.parametrize(
    "loss",
    [
        MAELoss(),
        MSELoss(),
        RMSELoss(),
        LogCoshLoss(),
        NegCosLoss(),
        CauchyLoss(alpha=0.04, frac=0.0),
        CauchyLoss(alpha=3.0, frac=0.0),
        EnsembleLoss([MSELoss(), CauchyLoss(alpha=0.1, frac=0.0)], [2.0, 1.0]),
    ],
)
def test_lower_bound_holds_and_is_reached(loss: LossFunction) -> None:
    """The loss never goes below its bound and reaches it when exact."""
    torch.manual_seed(0)
    target = torch.randn(256, 3)
    prediction = target + torch.randn(256, 3)
    bound = loss.lower_bound
    assert bound is not None
    elements = loss(prediction, target, return_elements=True)
    assert (elements >= bound - 1e-6).all()
    assert torch.isclose(loss(target, target), torch.tensor(bound), atol=1e-6)


@pytest.mark.parametrize("eps", [0.0, 0.03, 0.2])
def test_bce_label_smoothing_lower_bound_is_the_minimum(eps: float) -> None:
    """The lower bound is the smallest loss, at logit log((2 - eps) / eps)."""
    loss = BinaryCrossEntropyLoss(from_logits=True, label_smoothing=eps)
    logits = torch.linspace(-30, 30, 600001).unsqueeze(1)
    for label in (0.0, 1.0):
        values = loss(
            logits, torch.full_like(logits, label), return_elements=True
        )
        assert values.min().item() == pytest.approx(loss.lower_bound, abs=1e-6)
        if eps > 0:
            best = logits[values.argmin()].item()
            sign = 1 if label else -1
            # the loss is evaluated in float32: flat to ~1e-7 within ~5e-3
            assert best == pytest.approx(
                sign * np.log((2 - eps) / eps), abs=1e-2
            )


def _unit_vectors(n: int, seed: int = 0) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    v = torch.randn(n, 3, generator=generator, dtype=torch.float64)
    return v / v.norm(dim=1, keepdim=True)


def test_spcauchy_knots_interpolate_log_alpha() -> None:
    """At a knot the knot's scale is used, between knots log(alpha) is
    lerped."""
    rho, knots = [0.93, 0.98, 0.994], [0.0, 0.9, 1.0]
    loss = spCauchyLoss(rho=rho, rho_knots=knots)
    alphas = loss.alphas
    prediction = _unit_vectors(4, 2)
    target = _unit_vectors(4, 3)
    trackness = torch.tensor(
        [[-1.0], [0.9], [0.95], [2.0]], dtype=torch.float64
    )
    got = loss(
        prediction, torch.cat([target, trackness], 1), return_elements=True
    )
    u = 0.5 * ((prediction - target) ** 2).sum(1)
    expected_alpha = torch.tensor(
        [
            alphas[0],  # clamped below the first knot
            alphas[1],
            np.sqrt(alphas[1] * alphas[2]),  # midway in log(alpha)
            alphas[2],  # clamped above the last knot
        ],
        dtype=torch.float64,
    )
    expected = 2 * torch.log1p(2 * u / expected_alpha**2)
    assert torch.allclose(got.double(), expected, rtol=1e-5)


def test_spcauchy_knots_bound_is_zero_and_reached() -> None:
    """The loss is non-negative and 0 for exact predictions."""
    loss = spCauchyLoss(rho=[0.93, 0.994], rho_knots=[0.0, 1.0])
    assert loss.lower_bound == 0.0
    target = _unit_vectors(20, 4)
    trackness = torch.rand(20, 1, dtype=torch.float64)
    full_target = torch.cat([target, trackness], 1)
    assert loss(target, full_target, return_elements=True).abs().max() < 1e-6
    elements = loss(_unit_vectors(20, 5), full_target, return_elements=True)
    assert (elements >= 0).all()


def _grouped_cauchy(alpha: Any = (0.01, 0.02, 0.5)) -> CauchyLoss:
    return CauchyLoss(alpha=list(alpha), groups=[[0, 1], [2], [3]], frac=0.0)


def test_cauchy_new_alpha_replaces_the_device_copies() -> None:
    """Setting alpha after a call matches a loss built with that alpha."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1)
    prediction = torch.randn(100, 4, device=device)
    target = torch.randn(100, 4, device=device)
    annealed = _grouped_cauchy()
    annealed(prediction, target)
    annealed._alpha = [0.1, 0.2, 0.3]
    fresh = _grouped_cauchy([0.1, 0.2, 0.3])
    assert torch.equal(
        annealed(prediction, target, return_elements=True),
        fresh(prediction, target, return_elements=True),
    )
    assert annealed.lower_bound == fresh.lower_bound


def test_spcauchy_two_property_knots_interpolate_bilinearly() -> None:
    """Log(alpha) is bilinear in the cell and clamped beyond the grid."""
    rho = [[0.90, 0.95], [0.98, 0.995]]
    loss = spCauchyLoss(
        rho=rho, rho_knots=[0.0, 1.0], rho_knots_2=[100.0, 1100.0]
    )
    log_alpha = np.log(np.asarray(loss.alphas))
    assert log_alpha.shape == (2, 2)
    prediction, target = _unit_vectors(3, 12), _unit_vectors(3, 13)
    properties = torch.tensor(
        [[0.5, 600.0], [1.0, 100.0], [2.0, 5000.0]], dtype=torch.float64
    )
    got = loss(
        prediction, torch.cat([target, properties], 1), return_elements=True
    )
    u = 0.5 * ((prediction - target) ** 2).sum(1)
    expected_alpha = torch.tensor(
        np.exp(
            [
                log_alpha.mean(),  # centre of the cell
                log_alpha[1, 0],  # at a corner
                log_alpha[1, 1],  # clamped beyond both edges
            ]
        ),
        dtype=torch.float64,
    )
    expected = 2 * torch.log1p(2 * u / expected_alpha**2)
    assert torch.allclose(got.double(), expected, rtol=1e-5)
    assert loss.lower_bound == 0.0
