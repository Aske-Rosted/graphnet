"""Unit tests for LossFunction classes."""

import numpy as np
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


def test_focal_bce_logits_match_probabilities() -> None:
    """Logit and probability inputs give the same focal loss."""
    torch.manual_seed(0)
    logits = torch.randn(64, 1)
    target = torch.randint(0, 2, (64, 1)).float()

    from_logits = FocalBinaryCrossEntropyLoss(from_logits=True)
    from_probs = FocalBinaryCrossEntropyLoss(from_logits=False)

    assert torch.allclose(
        from_logits(logits, target, return_elements=True),
        from_probs(torch.sigmoid(logits), target, return_elements=True),
        atol=1e-5,
    )


def test_focal_bce_down_weights_easy_examples() -> None:
    """A confident correct prediction is penalised less than by BCE."""
    target = torch.ones(1, 1)
    prediction = torch.tensor([[0.95]])
    focal = FocalBinaryCrossEntropyLoss(gamma=2.0, alpha=0.5)
    bce = BinaryCrossEntropyLoss()

    assert focal(prediction, target) < 0.5 * bce(prediction, target)


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


def test_cauchy_heteroscedastic_uses_extra_columns() -> None:
    """With frac=1 the columns after the target width are the scales."""
    prediction = torch.tensor([[1.0, 2.0]])  # value, scale
    target = torch.tensor([[0.0]])
    expected = torch.log1p(torch.tensor(0.25)) + np.log(2.0)

    loss = CauchyLoss(frac=1.0)
    assert torch.allclose(loss(prediction, target), expected)


def test_sp_cauchy_prefers_aligned_confident_prediction() -> None:
    """The loss decreases towards the target and grows when confidently off."""
    target = torch.tensor([[0.0, 0.0, 1.0]])
    aligned = torch.tensor([[0.0, 0.0, 1.0, 10.0]])
    orthogonal = torch.tensor([[1.0, 0.0, 0.0, 10.0]])
    vague = torch.tensor([[1.0, 0.0, 0.0, 0.1]])
    loss = spCauchyLoss()

    assert loss(aligned, target) < loss(orthogonal, target)
    assert loss(vague, target) < loss(orthogonal, target)


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


@pytest.mark.parametrize("from_logits", [True, False])
def test_binary_cross_entropy_lower_bound(from_logits: bool) -> None:
    """(Focal) BCE is bounded by 0 and approaches it for confident hits."""
    target = torch.tensor([[0.0], [1.0]])
    confident = torch.tensor([[-30.0], [30.0]])
    if not from_logits:
        confident = torch.sigmoid(confident)
    for loss in (
        BinaryCrossEntropyLoss(from_logits=from_logits),
        FocalBinaryCrossEntropyLoss(from_logits=from_logits),
    ):
        assert loss.lower_bound == 0.0
        assert 0.0 <= float(loss(confident, target)) < 1e-6


def test_unbounded_losses_have_no_lower_bound() -> None:
    """Losses with a predicted concentration or scale report None."""
    assert spCauchyLoss().lower_bound is None
    assert VonMisesFisher3DLoss().lower_bound is None
    assert CauchyLoss(frac=1.0).lower_bound is None
    assert EnsembleLoss([MSELoss(), spCauchyLoss()]).lower_bound is None


def test_cauchy_learned_alpha_converges_to_residual_scale() -> None:
    """A learned alpha converges to the Cauchy scale of the residuals."""
    torch.manual_seed(0)
    scales = torch.tensor([0.02, 3.0])
    residuals = (
        torch.distributions.Cauchy(0.0, 1.0).sample((20000, 2)) * scales
    )
    loss = CauchyLoss(alpha=1.0, frac=0.0, learn_alpha=True, nb_outputs=2)
    optimizer = torch.optim.Adam(loss.parameters(), lr=0.05)
    zeros = torch.zeros_like(residuals)
    for _ in range(400):
        optimizer.zero_grad()
        loss(residuals, zeros).backward()
        optimizer.step()
    learned = torch.exp(loss.log_alpha[0].detach())
    assert torch.allclose(learned, scales, rtol=0.05)


def test_cauchy_conditioned_alpha_separates_populations() -> None:
    """Conditioning on a label gives each population its own scale."""
    torch.manual_seed(0)
    n = 20000
    label = (torch.rand(n, 1) > 0.5).float()  # e.g. 1 = track, 0 = cascade
    scale = torch.where(label > 0.5, 0.01, 1.0)
    residuals = torch.distributions.Cauchy(0.0, 1.0).sample((n, 1)) * scale
    target = torch.cat([torch.zeros(n, 1), label], dim=1)
    loss = CauchyLoss(
        alpha=0.1, frac=0.0, learn_alpha=True, nb_outputs=1, n_conditions=1
    )
    optimizer = torch.optim.Adam(loss.parameters(), lr=0.05)
    for _ in range(600):
        optimizer.zero_grad()
        loss(residuals, target).backward()
        optimizer.step()
    base, slope = loss.log_alpha.detach()[:, 0]
    assert torch.isclose(torch.exp(base), torch.tensor(1.0), rtol=0.1)
    assert torch.isclose(torch.exp(base + slope), torch.tensor(0.01), rtol=0.1)


def test_cauchy_learned_alpha_lower_bound() -> None:
    """The bound is the batch-mean log(alpha), reached at zero residual."""
    loss = CauchyLoss(
        alpha=[0.1, 2.0],
        frac=0.0,
        learn_alpha=True,
        nb_outputs=2,
        n_conditions=1,
    )
    with torch.no_grad():
        loss.log_alpha[1] = torch.tensor([-1.0, 0.5])
    values = torch.randn(64, 2)
    target = torch.cat([values, torch.rand(64, 1)], dim=1)
    exact = loss(values, target)
    assert loss.lower_bound is not None
    assert torch.isclose(exact, torch.tensor(loss.lower_bound), atol=1e-6)
    off = loss(values + 1.0, target, return_elements=True)
    assert float(off.mean()) > loss.lower_bound


def test_cauchy_groups_of_one_match_ungrouped() -> None:
    """Singleton groups reproduce the column-wise loss."""
    torch.manual_seed(0)
    prediction, target = torch.randn(32, 3), torch.randn(32, 3)
    plain = CauchyLoss(alpha=0.3, frac=0.0)
    grouped = CauchyLoss(alpha=0.3, frac=0.0, groups=[[0], [1], [2]])
    assert torch.allclose(
        plain(prediction, target, return_elements=True),
        grouped(prediction, target, return_elements=True),
    )
    assert np.isclose(plain.lower_bound, grouped.lower_bound)


def test_cauchy_group_is_rotation_invariant() -> None:
    """Within a group only the length of the residual matters."""
    alpha = 0.1
    target = torch.zeros(2, 3)
    residuals = torch.tensor([[1.0, 0.0, 0.3], [0.6, -0.8, 0.3]])
    grouped = CauchyLoss(alpha=alpha, frac=0.0, groups=[[0, 1], [2]])
    loss = grouped(residuals, target, return_elements=True)
    assert torch.isclose(loss[0], loss[1])
    expected = (
        1.5 * np.log1p(1.0 / alpha**2)
        + np.log1p(0.09 / alpha**2)
        + 3 * np.log(alpha)
    ) / 3
    assert torch.isclose(loss[0], torch.tensor(expected, dtype=torch.float))

    plain = CauchyLoss(alpha=alpha, frac=0.0)(
        residuals, target, return_elements=True
    )
    assert not torch.isclose(plain[0], plain[1])


def test_cauchy_group_lower_bound() -> None:
    """The bound is the size-weighted mean log(alpha), reached when exact."""
    loss = CauchyLoss(
        alpha=[0.05, 0.2, 3.0], frac=0.0, groups=[[0, 1], [2], [3]]
    )
    values = torch.randn(16, 4)
    expected = (2 * np.log(0.05) + np.log(0.2) + np.log(3.0)) / 4
    assert np.isclose(loss.lower_bound, expected)
    assert torch.isclose(
        loss(values, values), torch.tensor(expected, dtype=torch.float)
    )
    off = loss(values + 0.1, values, return_elements=True)
    assert (off > expected).all()


def test_cauchy_group_learned_alpha_is_likelihood_scale() -> None:
    """Learned group scales recover those of multivariate Cauchy samples."""
    torch.manual_seed(0)
    n = 40000
    scales = torch.tensor([0.05, 2.0])
    # d-dimensional Cauchy: normal vector over an independent |normal|
    pair = torch.randn(n, 2) / torch.randn(n, 1).abs() * scales[0]
    single = torch.randn(n, 1) / torch.randn(n, 1).abs() * scales[1]
    residuals = torch.cat([pair, single], dim=1)
    loss = CauchyLoss(
        alpha=1.0, frac=0.0, learn_alpha=True, groups=[[0, 1], [2]]
    )
    optimizer = torch.optim.Adam(loss.parameters(), lr=0.05)
    zeros = torch.zeros_like(residuals)
    for _ in range(400):
        optimizer.zero_grad()
        loss(residuals, zeros).backward()
        optimizer.step()
    assert torch.allclose(loss.alphas(), scales, rtol=0.05)
    assert set(loss.monitored_values()) == {"alpha_0", "alpha_1"}
    assert CauchyLoss(alpha=1.0, frac=0.0).monitored_values() == {}


def test_sp_cauchy_fixed_rho_matches_predicted() -> None:
    """A fixed rho equals the predicted form with k = rho / (1 - rho)."""
    torch.manual_seed(0)
    rho = 0.9
    mu = torch.nn.functional.normalize(torch.randn(64, 3), dim=1)
    target = torch.nn.functional.normalize(torch.randn(64, 3), dim=1)
    k = torch.full((64, 1), rho / (1 - rho))
    fixed = spCauchyLoss(rho=rho)(mu, target, return_elements=True)
    predicted = spCauchyLoss()(
        torch.cat([mu, k], dim=1), target, return_elements=True
    )
    assert torch.allclose(fixed, predicted, rtol=1e-5)


def test_sp_cauchy_fixed_rho_is_chord_cauchy_with_bound() -> None:
    """The loss is 2 log(1 + chord^2 / alpha^2) above its lower bound."""
    rho = 0.995
    loss = spCauchyLoss(rho=rho)
    alpha_squared = (1 - rho) ** 2 / rho
    torch.manual_seed(0)
    mu = torch.nn.functional.normalize(torch.randn(64, 3), dim=1)
    target = torch.nn.functional.normalize(torch.randn(64, 3), dim=1)
    chord_squared = ((mu - target) ** 2).sum(dim=1)
    expected = 2 * torch.log1p(chord_squared / alpha_squared)
    elements = loss(mu, target, return_elements=True)
    assert torch.allclose(elements - loss.lower_bound, expected, rtol=1e-4)
    assert torch.isclose(
        loss(target, target), torch.tensor(loss.lower_bound), atol=1e-5
    )
