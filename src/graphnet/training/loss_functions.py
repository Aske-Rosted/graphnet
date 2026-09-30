"""Collection of loss functions.

All loss functions inherit from `LossFunction` which ensures a common syntax,
handles per-event weights, etc.
"""

from abc import abstractmethod
from typing import Any, Optional, Union, List, Dict

import numpy as np
import scipy.special
import torch
from torch import Tensor
from torch import nn
from torch.nn.functional import (
    one_hot,
    binary_cross_entropy,
    binary_cross_entropy_with_logits,
    softplus,
)

from graphnet.models.model import Model
from graphnet.utilities.decorators import final


class LossFunction(Model):
    """Base class for loss functions in `graphnet`."""

    def __init__(self, **kwargs: Any) -> None:
        """Construct `LossFunction`, saving model config."""
        super().__init__(**kwargs)

    @property
    def lower_bound(self) -> Optional[float]:
        """Return a lower bound of the (unweighted, averaged) loss, if known.

        The bound is the value the loss cannot go below for any prediction
        and target, e.g. 0 for squared errors. It is used to measure how far
        a task is from its best possible loss, e.g. for balancing several
        losses. `None` means that no finite bound is known, as for
        likelihoods with a predicted concentration or width, which can
        become arbitrarily confident.

        NOTE: With per-event `weights`, the bound scales with the mean
        weight.
        """
        return None

    def monitored_values(self) -> Dict[str, float]:
        """Return quantities of the loss worth logging during training.

        E.g. the current value of a learned scale. Empty by default.
        """
        return {}

    @final
    def forward(  # type: ignore[override]
        self,
        prediction: Tensor,
        target: Tensor,
        weights: Optional[Tensor] = None,
        return_elements: bool = False,
    ) -> Tensor:
        """Forward pass for all loss functions.

        Args:
            prediction: Tensor containing predictions. Shape [N,P]
            target: Tensor containing targets. Shape [N,T]
            return_elements: Whether elementwise loss terms should be returned.
                The alternative is to return the averaged loss across examples.

        Returns:
            Loss, either averaged to a scalar (if `return_elements = False`) or
            elementwise terms with shape [N,] (if `return_elements = True`).
        """
        elements = self._forward(prediction, target)
        if weights is not None:
            elements = elements * weights
        assert elements.size(dim=0) == target.size(
            dim=0
        ), "`_forward` should return elementwise loss terms."

        return elements if return_elements else torch.mean(elements)

    @abstractmethod
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Syntax like `.forward`, for implentation in inheriting classes."""


class MAELoss(LossFunction):
    """Mean absolute error loss."""

    @property
    def lower_bound(self) -> float:
        """Return 0; the loss is non-negative."""
        return 0.0

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        return torch.mean(torch.abs(prediction - target), dim=-1)


class MSELoss(LossFunction):
    """Mean squared error loss."""

    @property
    def lower_bound(self) -> float:
        """Return 0; the loss is non-negative."""
        return 0.0

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        # Check(s)
        assert prediction.dim() == 2
        if target.dim() != prediction.dim():
            target = target.squeeze(1)
        assert prediction.size() == target.size()

        elements = torch.mean((prediction - target) ** 2, dim=-1)
        return elements


class RMSELoss(MSELoss):
    """Root mean squared error loss."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        # Check(s)
        elements = super()._forward(prediction, target)
        elements = torch.sqrt(elements)
        return elements


class LogCoshLoss(LossFunction):
    """Log-cosh loss function.

    Acts like x^2 for small x; and like |x| for large x.
    """

    @classmethod
    def _log_cosh(cls, x: Tensor) -> Tensor:  # pylint: disable=invalid-name
        """Numerically stable version on log(cosh(x)).

        Used to avoid `inf` for even moderately large differences.
        See [https://github.com/keras-team/keras/blob/v2.6.0/keras/losses.py#L1580-L1617] # noqa: E501
        """
        return x + softplus(-2.0 * x) - np.log(2.0)

    @property
    def lower_bound(self) -> float:
        """Return 0; the loss is non-negative."""
        return 0.0

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        diff = prediction - target
        elements = self._log_cosh(diff)
        return elements


class CrossEntropyLoss(LossFunction):
    """Compute cross-entropy loss for classification tasks.

    Predictions are an [N, num_class]-matrix of logits (i.e., non-softmax'ed
    probabilities), and targets are an [N,1]-matrix with integer values in
    (0, num_classes - 1).
    """

    def __init__(
        self,
        options: Union[int, List[Any], Dict[Any, int]],
        *args: Any,
        **kwargs: Any,
    ):
        """Construct CrossEntropyLoss."""
        # Base class constructor
        super().__init__(*args, **kwargs)

        # Member variables
        self._options = options
        self._nb_classes: int
        if isinstance(self._options, int):
            assert self._options in [torch.int32, torch.int64]
            assert (
                self._options >= 2
            ), f"Minimum of two classes required. Got {self._options}."
            self._nb_classes = options  # type: ignore
        elif isinstance(self._options, list):
            self._nb_classes = len(self._options)  # type: ignore
        elif isinstance(self._options, dict):
            self._nb_classes = len(
                np.unique(list(self._options.values()))
            )  # type: ignore
        else:
            raise ValueError(
                f"Class options of type {type(self._options)} not supported"
            )

        self._loss = nn.CrossEntropyLoss(reduction="none")

    @property
    def lower_bound(self) -> float:
        """Return 0; the loss is non-negative."""
        return 0.0

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Transform outputs to angle and prepare prediction."""
        if isinstance(self._options, int):
            # Integer number of classes: Targets are expected to be in
            # (0, nb_classes - 1).

            # Target integers are positive
            assert torch.all(target >= 0)

            # Target integers are consistent with the expected number of class.
            assert torch.all(target < self._options)

            assert target.dtype in [torch.int32, torch.int64]
            target_integer = target

        elif isinstance(self._options, list):
            # List of classes: Mapping target classes in list onto
            # (0, nb_classes - 1). Example:
            #    Given options: [1, 12, 13, ...]
            #    Yields: [1, 13, 12] -> [0, 2, 1, ...]
            target_integer = torch.tensor(
                [self._options.index(value) for value in target]
            )

        elif isinstance(self._options, dict):
            # Dictionary of classes: Mapping target classes in dict onto
            # (0, nb_classes - 1). Example:
            #     Given options: {1: 0, -1: 0, 12: 1, -12: 1, ...}
            #     Yields: [1, -1, -12, ...] -> [0, 0, 1, ...]
            target_integer = torch.tensor(
                [self._options[int(value)] for value in target]
            )

        else:
            assert False, "Shouldn't reach here."

        target_one_hot: Tensor = one_hot(target_integer, self._nb_classes).to(
            prediction.device
        )

        return self._loss(prediction.float(), target_one_hot.float())


class BinaryCrossEntropyLoss(LossFunction):
    """Compute binary cross entropy loss."""

    def __init__(self, from_logits: bool = False, *args: Any, **kwargs: Any):
        """Construct BinaryCrossEntropyLoss.

        Args:
            from_logits: Whether the predictions are logits.
                NOTE: If True, the predictions are expected to be raw scores
                (i.e., not passed through a sigmoid function). If False, the
                predictions are expected to be probabilities
                (i.e., passed through a sigmoid function).
        """
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    @property
    def lower_bound(self) -> float:
        """Return 0; the loss is non-negative."""
        return 0.0

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if self._from_logits:
            return binary_cross_entropy_with_logits(
                prediction.float(), target.float(), reduction="none"
            )
        else:
            return binary_cross_entropy(
                prediction.float(), target.float(), reduction="none"
            )


class FocalBinaryCrossEntropyLoss(BinaryCrossEntropyLoss):
    """Compute the focal binary cross entropy loss.

    Down-weights well-classified examples by a factor `(1 - p_t)**gamma`
    and balances the two classes with `alpha`, following Lin et al.,
    "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        from_logits: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        """Construct FocalBinaryCrossEntropyLoss.

        Args:
            gamma: Focusing parameter. `gamma = 0` recovers the (alpha
                weighted) binary cross entropy.
            alpha: Weight of the positive class; the negative class is
                weighted by `1 - alpha`.
            from_logits: Whether the predictions are logits (raw scores)
                rather than probabilities.
        """
        super().__init__(from_logits, *args, **kwargs)
        self._gamma = gamma
        self._alpha = alpha

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        bce = super()._forward(prediction, target)
        probability = prediction.float()
        if self._from_logits:
            probability = torch.sigmoid(probability)
        target = target.float()
        alpha_t = self._alpha * target + (1 - self._alpha) * (1 - target)
        p_t = probability * target + (1 - probability) * (1 - target)
        return alpha_t * (1 - p_t) ** self._gamma * bce


class LogCMK(torch.autograd.Function):
    """MIT License.

    Copyright (c) 2019 Max Ryabinin

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
    _____________________

    From [https://github.com/mryab/vmf_loss/blob/master/losses.py] Modified to
    use modified Bessel function instead of exponentially scaled ditto
    (i.e. `.ive` -> `.iv`) as indicated in [1812.04616] in spite of suggestion
    in Sec. 8.2 of this paper. The change has been validated through comparison
    with exact calculations for `m=2` and `m=3` and found to yield the correct
    results.
    """

    @staticmethod
    def forward(
        ctx: Any, m: int, kappa: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name,arguments-differ
        """Forward pass."""
        dtype = kappa.dtype
        ctx.save_for_backward(kappa)
        ctx.m = m
        ctx.dtype = dtype
        kappa = kappa.double()
        iv = torch.from_numpy(
            scipy.special.iv(m / 2.0 - 1, kappa.cpu().numpy())
        ).to(kappa.device)
        return (
            (m / 2.0 - 1) * torch.log(kappa)
            - torch.log(iv)
            - (m / 2) * np.log(2 * np.pi)
        ).type(dtype)

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name,arguments-differ
        """Backward pass for LogCMK computation.

        Mathematical Background:
        -----------------------
        For the von Mises-Fisher distribution, the gradient of log C_m(κ) with
        respect to κ is given by the ratio of modified Bessel functions:

        ∂/∂κ log C_m(κ) = (m/2-1)/κ - I_{m/2}(κ)/I_{m/2-1}(κ)

        For m=3, this simplifies to the exact formula:
        ∂/∂κ log C_3(κ) = 1/κ - 1/tanh(κ)

        For small κ values, we use the Taylor series approximation:
        f(κ) = -κ/3 + κ³/45 - 2κ⁵/945 + O(κ⁷)

        The first-order approximation -κ/3 provides sufficient accuracy for
        |κ| < 1e-6, with truncation error bounded by |κ|³/45 ≲ O(10⁻²¹).

        Implementation Details:
        ----------------------
        Uses boolean masking to avoid double evaluation and RuntimeWarnings:
        - Small κ: |κ| < 1e-6 → gradient = -κ/3 (Taylor approximation)
        - Large κ: |κ| ≥ 1e-6 → gradient = 1/κ - 1/tanh(κ) (exact formula)

        References:
        ----------
        [1] von Mises-Fisher distribution: Wikipedia
        [2] arXiv:1812.04616, Section 8.2
        [3] MIT License (c) 2019 Max Ryabinin - Modified for GraphNeT

        Args:
            ctx: Autograd context containing saved tensors and metadata.
            grad_output: Gradient with respect to the output tensor.

        Returns:
            Tuple of gradients: (None for m, gradient w.r.t. κ).
        """
        kappa = ctx.saved_tensors[0]
        m = ctx.m
        dtype = ctx.dtype
        kappa = kappa.double().cpu().numpy()
        if np.isclose(m, 3, atol=1e-6):
            # Initialize gradient array
            grads = np.zeros_like(kappa)

            # Handle small kappa values (including zero) to avoid division by zero
            small_mask = np.abs(kappa) < 1e-6
            grads[small_mask] = -kappa[small_mask] / 3

            # Handle large kappa values
            large_mask = ~small_mask
            if np.any(large_mask):
                kappa_large = kappa[large_mask]
                grads[large_mask] = 1 / kappa_large - 1 / np.tanh(kappa_large)
        else:
            grads = -(
                (scipy.special.iv(m / 2.0, kappa))
                / (scipy.special.iv(m / 2.0 - 1, kappa))
            )
        return (
            None,
            grad_output
            * torch.from_numpy(grads).to(grad_output.device).type(dtype),
        )


class VonMisesFisherLoss(LossFunction):
    """General class for calculating von Mises-Fisher loss.

    Requires implementation for specific dimension `m` in which the target and
    prediction vectors need to be prepared.
    """

    @classmethod
    def log_cmk_exact(
        cls, m: int, kappa: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name
        """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss exactly."""
        return LogCMK.apply(m, kappa)

    @classmethod
    def log_cmk_approx(
        cls, m: int, kappa: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name
        """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss approx.

        [https://arxiv.org/abs/1812.04616] Sec. 8.2 with additional minus sign.
        """
        v = m / 2.0 - 0.5
        a = torch.sqrt((v + 1) ** 2 + kappa**2)
        b = v - 1
        return -a + b * torch.log(b + a)

    @classmethod
    def log_cmk(
        cls, m: int, kappa: Tensor, kappa_switch: float = 100.0
    ) -> Tensor:  # pylint: disable=invalid-name
        """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss.

        Since `log_cmk_exact` is diverges for `kappa` >~ 700 (using float64
        precision), and since `log_cmk_approx` is unaccurate for small `kappa`,
        this method automatically switches between the two at `kappa_switch`,
        ensuring continuity at this point.
        """
        kappa_switch = torch.tensor([kappa_switch]).to(kappa.device)
        mask_exact = kappa < kappa_switch

        # Ensure continuity at `kappa_switch`
        offset = cls.log_cmk_approx(m, kappa_switch) - cls.log_cmk_exact(
            m, kappa_switch
        )
        ret = cls.log_cmk_approx(m, kappa) - offset
        ret[mask_exact] = cls.log_cmk_exact(m, kappa[mask_exact])
        return ret

    def _evaluate(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate von Mises-Fisher loss for a vector in D dimensons.

        This loss utilises the von Mises-Fisher distribution, which is a
        probability distribution on the (D - 1) sphere in D-dimensional space.

        Args:
            prediction: Predicted vector, of shape [batch_size, D].
            target: Target unit vector, of shape [batch_size, D].

        Returns:
            Elementwise von Mises-Fisher loss terms.
        """
        # Check(s)
        assert prediction.dim() == 2
        assert target.dim() == 2
        assert prediction.size() == target.size()

        # Computing loss
        m = target.size()[1]
        k = torch.norm(prediction, dim=1)
        dotprod = torch.sum(prediction * target, dim=1)
        elements = -self.log_cmk(m, k) - dotprod
        return elements

    @abstractmethod
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError


class VonMisesFisher2DLoss(VonMisesFisherLoss):
    """Von Mises-Fisher loss function vectors in the 2D plane."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate von Mises-Fisher loss for an angle in the 2D plane.

        Args:
            prediction: Output of the model. Must have shape [N, 2] where 0th
                column is a prediction of `angle` and 1st column is an estimate
                of `kappa`.
            target: Target tensor, extracted from graph object.

        Returns:
            loss: Elementwise von Mises-Fisher loss terms. Shape [N,]
        """
        # Check(s)
        assert prediction.dim() == 2 and prediction.size()[1] == 2
        assert target.dim() == 2
        assert prediction.size()[0] == target.size()[0]

        # Formatting target
        angle_true = target[:, 0]
        t = torch.stack(
            [
                torch.cos(angle_true),
                torch.sin(angle_true),
            ],
            dim=1,
        )

        # Formatting prediction
        angle_pred = prediction[:, 0]
        kappa = prediction[:, 1]
        p = kappa.unsqueeze(1) * torch.stack(
            [
                torch.cos(angle_pred),
                torch.sin(angle_pred),
            ],
            dim=1,
        )

        return self._evaluate(p, t)


class EuclideanDistanceLoss(LossFunction):
    """Mean squared error in three dimensions."""

    @property
    def lower_bound(self) -> float:
        """Return 0; the loss is non-negative."""
        return 0.0

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate 3D Euclidean distance between predicted and target.

        Args:
            prediction: Output of the model. Must have shape [N, 3]
            target: Target tensor, extracted from graph object.

        Returns:
            Elementwise von Mises-Fisher loss terms. Shape [N,]
        """
        return torch.sqrt(
            (prediction[:, 0] - target[:, 0]) ** 2
            + (prediction[:, 1] - target[:, 1]) ** 2
            + (prediction[:, 2] - target[:, 2]) ** 2
        )


class VonMisesFisher3DLoss(LossFunction):
    """Negative log-likelihood of the 3D von Mises-Fisher distribution.

    The density on the unit sphere is `f(t | mu, kappa) = C_3(kappa) *
    exp(kappa * mu . t)` with normalization `C_3(kappa) = kappa /
    (4*pi*sinh(kappa))`. Substituting `sinh(kappa) = exp(kappa) *
    (1 - exp(-2*kappa)) / 2` and, exactly for unit vectors,
    `kappa * (1 - mu . t) = kappa * ||t - mu||**2 / 2`, the negative
    log-likelihood becomes

    `kappa * ||t - mu||**2 / 2 - log(kappa / (2*pi*(1 - exp(-2*kappa))))`

    where no term grows like `kappa` — the `exp(kappa)` of the normalizer
    cancels analytically against the alignment term. Every operation is
    elementary and on-device (no Bessel functions), the derivative is
    continuous in `kappa` (no approximation boundary for gradient descent
    on the concentration to stall at), and the squared-chord angular term
    stays accurate in float32 at high concentration, where `1 - cos(theta)`
    is comparable to the floating-point epsilon.

    All arithmetic runs in the dtype of the inputs; resolving sub-degree
    angular information requires the direction vectors in float32 or better.
    """

    def __init__(self, eps: float = 1e-8, **kwargs: Any) -> None:
        """Construct `VonMisesFisher3DLoss`.

        Args:
            eps: Regularization of the normalizer, evaluated as
                `(kappa + eps) / (1 - exp(-2*kappa) + 2*eps)` so that value
                and gradient stay finite at `kappa = 0` while the ratio
                keeps its exact limit of 1/2 (the uniform value
                `-log(4*pi)`). Gradients lose accuracy below
                `kappa ~ sqrt(eps)`, where the distribution is effectively
                uniform.
        """
        super().__init__(**kwargs)
        self._eps = eps

    @staticmethod
    def _log_cmk_scaled(kappa: Tensor, eps: float) -> Tensor:
        """Calculate `log(exp(kappa) * C_3(kappa))`.

        Args:
            kappa: Concentration parameters, of shape [batch_size,].
            eps: Regularization, see the constructor.

        Returns:
            Elementwise `log(exp(kappa) * C_3(kappa))`, shape [batch_size,].
        """
        return (
            torch.log(kappa + eps)
            - torch.log1p(2 * eps - torch.exp(-2 * kappa))
            - np.log(2 * np.pi)
        )

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate von Mises-Fisher loss for a direction in the 3D.

        Args:
            prediction: Output of the model. Must have shape [N, 4] where
                columns 0, 1, 2 are predictions of `direction` and last column
                is an estimate of `kappa`.
            target: Target unit vector, extracted from graph object.

        Returns:
            Elementwise von Mises-Fisher loss terms. Shape [N,]
        """
        target = target.reshape(-1, 3)
        # Check(s)
        assert prediction.dim() == 2 and prediction.size()[1] == 4
        assert target.dim() == 2
        assert prediction.size()[0] == target.size()[0]

        # With x the raw model output the task head normalized:
        direction = prediction[:, :3]  # x / (|x| + eps)
        direction_norm = torch.norm(direction, dim=1)  # |x| / (|x| + eps)
        concentration = prediction[:, 3] * direction_norm  # |x|
        unit_direction = direction / direction_norm.unsqueeze(1)  # x / |x|

        # For unit vectors, concentration * (1 - mu . t) equals
        # concentration * ||t - mu||**2 / 2 exactly. The squared-chord form
        # measures the small angular deficit directly, whereas subtracting
        # cos(theta) from 1 loses all significant digits once theta**2
        # approaches the floating-point epsilon.
        sq_chord = torch.sum((target - unit_direction) ** 2, dim=1)
        return concentration * sq_chord / 2 - self._log_cmk_scaled(
            concentration, self._eps
        )


class EnsembleLoss(LossFunction):
    """Chain multiple loss functions together."""

    def __init__(
        self,
        loss_functions: List[LossFunction],
        loss_factors: Optional[List[float]] = None,
        prediction_keys: Optional[List[List[int]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Chain multiple loss functions together.

            Optionally apply a weight to each loss function contribution.

            E.g. Loss = RMSE*0.5 + LogCoshLoss*1.5

        Args:
            loss_functions: A list of loss functions to use.
                Each loss function contributes a term to the overall loss.
            loss_factors: An optional list of factors that will be mulitplied
            to each loss function contribution. Must be ordered according
            to `loss_functions`. If not given, the weights default to 1.
            prediction_keys: An optional list of lists of indices for which
                prediction columns to use for each loss function. If not
                given, all columns are used for all loss functions.
        """
        if loss_factors is None:
            # add weight of 1 - i.e no discrimination
            loss_factors = np.repeat(1, len(loss_functions)).tolist()

        assert len(loss_functions) == len(loss_factors)
        self._factors = loss_factors
        self._loss_functions = loss_functions

        if prediction_keys is not None:
            self._prediction_keys: Optional[List[List[int]]] = prediction_keys
        else:
            self._prediction_keys = None
        super().__init__(*args, **kwargs)

    @property
    def lower_bound(self) -> Optional[float]:
        """Return the weighted sum of the parts' bounds, if all are known."""
        bounds = [loss.lower_bound for loss in self._loss_functions]
        if any(b is None for b in bounds) or any(f < 0 for f in self._factors):
            return None
        return float(sum(f * b for f, b in zip(self._factors, bounds)))

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate loss using multiple loss functions.

        Args:
            prediction: Output of the model.
            target: Target tensor, extracted from graph object.

        Returns:
            Elementwise loss terms. Shape [N,]
        """
        if self._prediction_keys is None:
            prediction_keys = [list(range(prediction.size(1)))] * len(
                self._loss_functions
            )
        else:
            prediction_keys = self._prediction_keys
        for k, (loss_function, prediction_key) in enumerate(
            zip(self._loss_functions, prediction_keys)
        ):
            if k == 0:
                elements = self._factors[k] * loss_function._forward(
                    prediction=prediction[:, prediction_key], target=target
                )
            else:
                elements += self._factors[k] * loss_function._forward(
                    prediction=prediction[:, prediction_key], target=target
                )
        return elements


class RMSEVonMisesFisher3DLoss(EnsembleLoss):
    """Combine the VonMisesFisher3DLoss with RMSELoss."""

    def __init__(self, vmfs_factor: float = 0.05) -> None:
        """VonMisesFisher3DLoss with a RMSE penality term.

            The VonMisesFisher3DLoss will be weighted with `vmfs_factor`.

        Args:
            vmfs_factor: A factor applied to the VonMisesFisher3DLoss term.
            Defaults ot 0.05.
        """
        super().__init__(
            loss_functions=[RMSELoss(), VonMisesFisher3DLoss()],
            loss_factors=[1, vmfs_factor],
            prediction_keys=[[0, 1, 2], [0, 1, 2, 3]],
        )


class NegCosLoss(LossFunction):
    """Negative Cosine error loss."""

    @property
    def lower_bound(self) -> float:
        """Return -1, the negative cosine of aligned vectors."""
        return -1.0

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        # Check(s)
        assert prediction.dim() == 2
        if target.dim() != prediction.dim():
            target = target.squeeze(1)
        assert prediction.size() == target.size()

        reco_norm = torch.nn.functional.normalize(prediction, dim=1)
        orig_norm = torch.nn.functional.normalize(target, dim=1)
        elements = -(reco_norm * orig_norm).sum(dim=1)
        return elements


class CauchyLoss(LossFunction):
    """Cauchy (Lorentzian) loss with a fixed, learned or predicted scale.

    The homoscedastic term of one output column is
    `log(1 + (|x - y| / alpha)**2) + log(alpha)`. The scale `alpha` sets
    which residual size the loss is most sensitive to (its gradient peaks at
    `|x - y| = alpha`).

    Columns can be combined into `groups` that share a scale and enter
    through the length of their joint residual `r`, as in the multivariate
    Cauchy distribution in `d` dimensions:
    `(1 + d) / 2 * log(1 + (|r| / alpha)**2) + d * log(alpha)`. Unlike a sum
    over columns this does not depend on how the residual is oriented with
    respect to the coordinate axes within the group. A group of one column
    is the term above. The loss is the sum over groups divided by the number
    of columns.

    The scale is either

    - fixed (default): a float, or one value per group (column),
    - learned (`learn_alpha`): one scale per group (column), trained with
      the model as the maximum-likelihood Cauchy scale of the residuals, so
      it tightens as the predictions improve and no manual schedule is
      needed. With `n_conditions > 0`, the last `n_conditions` columns of
      the target are conditioning variables `c` (not regressed) and
      `log(alpha) = a + c @ B` is linear in them, e.g. separate scales for
      tracks and cascades when conditioning on the true trackness.
    - predicted: if the prediction carries more columns than the target,
      the extra columns are read as per-element scales `sigma` and enter the
      heteroscedastic term `log(1 + (|x - y| / sigma)**2) + log(sigma)`.
      `frac` blends the two terms: 0 is purely homoscedastic, 1 purely
      heteroscedastic.
    """

    def __init__(
        self,
        alpha: Union[float, List[float]] = 1.0,
        frac: float = 1.0,
        learn_alpha: bool = False,
        nb_outputs: Optional[int] = None,
        n_conditions: int = 0,
        groups: Optional[List[List[int]]] = None,
        **kwargs: Any,
    ) -> None:
        """Construct CauchyLoss.

        Args:
            alpha: Scale of the homoscedastic term; with `learn_alpha` its
                initial value. A float, or one value per group (per output
                column without `groups`).
            frac: Weight of the heteroscedastic term; the homoscedastic term
                is weighted by `1 - frac`.
            learn_alpha: If True, the scale is a learned parameter per
                group (column). Requires `frac = 0` and `groups` or
                `nb_outputs`.
            nb_outputs: Number of regressed target columns.
            n_conditions: Number of trailing target columns that condition
                the learned scale instead of being regressed.
            groups: Partition of the output columns into groups with a
                common scale and a rotation-invariant term, e.g.
                `[[0, 1], [2], [3]]` for (x, y), z and t. Requires
                `frac = 0`. Defaults to one group per column.
        """
        super().__init__(**kwargs)
        self._frac = frac
        self._learn_alpha = learn_alpha
        self._n_conditions = n_conditions
        self._last_bound: Optional[float] = None

        self._group_index: Optional[Tensor] = None
        self._group_size: Optional[Tensor] = None
        if groups is not None:
            assert frac == 0, "`groups` require `frac = 0`."
            columns = sorted(column for group in groups for column in group)
            assert columns == list(
                range(len(columns))
            ), "`groups` must contain every output column exactly once."
            assert nb_outputs in (None, len(columns))
            index = torch.empty(len(columns), dtype=torch.long)
            for number, group in enumerate(groups):
                index[group] = number
            self._group_index = index
            self._group_size = torch.tensor(
                [float(len(group)) for group in groups]
            )

        n_scales = len(groups) if groups is not None else nb_outputs
        self._n_scales = n_scales
        if not isinstance(alpha, (int, float)):
            assert frac == 0, "One alpha per column requires `frac = 0`."
        log_alpha = self._as_log_alpha(alpha)

        if learn_alpha:
            assert frac == 0, "`learn_alpha` requires `frac = 0`."
            assert (
                n_scales is not None
            ), "`learn_alpha` needs `groups` or `nb_outputs`."
            parameter = torch.zeros(1 + n_conditions, n_scales)
            parameter[0] = log_alpha
            self.log_alpha = torch.nn.Parameter(parameter)
        else:
            assert n_conditions == 0, "Conditions require `learn_alpha`."
            self._fixed_alpha = alpha
            self._fixed_log_alpha = log_alpha
        if frac == 0:
            self._last_bound = self._bound(log_alpha.unsqueeze(0))

    def _as_log_alpha(self, alpha: Union[float, List[float]]) -> Tensor:
        """Return log(alpha) with one entry per group (column)."""
        if isinstance(alpha, (int, float)):
            return torch.full((self._n_scales or 1,), float(np.log(alpha)))
        assert self._n_scales in (None, len(alpha)), (
            f"Got {len(alpha)} values of alpha for {self._n_scales} "
            "groups (columns)."
        )
        return torch.log(torch.as_tensor(alpha, dtype=torch.float))

    @property
    def alpha(self) -> Union[float, List[float]]:
        """Return the fixed scale(s).

        The fixed scale can be set during training, e.g. to anneal it
        from a wide to a narrow value. Not available for a learned
        scale.
        """
        if self._learn_alpha:
            raise AttributeError("The scale is learned; see `alphas()`.")
        return self._fixed_alpha

    @alpha.setter
    def alpha(self, alpha: Union[float, List[float]]) -> None:
        if self._learn_alpha:
            raise AttributeError("The scale is learned and cannot be set.")
        self._fixed_alpha = alpha
        self._fixed_log_alpha = self._as_log_alpha(alpha)
        if self._frac == 0:
            self._last_bound = self._bound(self._fixed_log_alpha.unsqueeze(0))

    # Name of the attribute before the scale became a property.
    _alpha = alpha

    def _sizes(self, log_alpha: Tensor) -> Tensor:
        """Return the number of columns of each scale."""
        if self._group_size is None:
            return torch.ones(1).to(log_alpha)
        return self._group_size.to(log_alpha)

    def _bound(self, log_alpha: Tensor) -> float:
        """Return the batch-mean minimum of the homoscedastic loss."""
        sizes = self._sizes(log_alpha).expand(log_alpha.shape[-1])
        per_event = (sizes * log_alpha.detach()).sum(dim=-1) / sizes.sum()
        return float(per_event.mean())

    @property
    def lower_bound(self) -> Optional[float]:
        """Return the (size-weighted) mean log(alpha) for `frac = 0`.

        Each term is minimal at zero residual, where it equals `d *
        log(alpha)`. For a learned scale this is the bound at the
        current scale for the most recent batch. With a heteroscedastic
        part (`frac > 0`) the scale is predicted and the loss has no
        meaningful lower bound.
        """
        if self._frac != 0:
            return None
        return self._last_bound

    def alphas(self) -> Tensor:
        """Return the current scale of each group (column).

        For a conditioned scale this is the scale at zero conditions.
        """
        if self._learn_alpha:
            return torch.exp(self.log_alpha[0].detach())
        return torch.exp(self._fixed_log_alpha)

    def monitored_values(self) -> Dict[str, float]:
        """Return the learned scales (and their condition slopes)."""
        if not self._learn_alpha:
            return {}
        values = {
            f"alpha_{i}": float(alpha) for i, alpha in enumerate(self.alphas())
        }
        for c, slopes in enumerate(self.log_alpha[1:].detach()):
            for i, slope in enumerate(slopes):
                values[f"log_alpha_{i}_slope_{c}"] = float(slope)
        return values

    def _log_alpha(self, conditions: Optional[Tensor], like: Tensor) -> Tensor:
        """Return log(alpha), broadcastable to [N, n_scales]."""
        if not self._learn_alpha:
            return self._fixed_log_alpha.to(like).unsqueeze(0)
        log_alpha = self.log_alpha[0].unsqueeze(0)
        if conditions is not None:
            log_alpha = (
                log_alpha + conditions.to(log_alpha) @ self.log_alpha[1:]
            )
        return log_alpha

    def _homoscedastic(
        self,
        prediction: Tensor,
        target: Tensor,
        conditions: Optional[Tensor] = None,
    ) -> Tensor:
        squared = (prediction - target) ** 2
        log_alpha = self._log_alpha(conditions, squared)
        if self._group_index is not None:
            assert squared.size(1) == self._group_index.numel(), (
                f"`groups` cover {self._group_index.numel()} columns, got "
                f"{squared.size(1)}."
            )
            squared = (
                torch.zeros(
                    squared.size(0), log_alpha.size(1), dtype=squared.dtype
                )
                .to(squared.device)
                .index_add_(1, self._group_index.to(squared.device), squared)
            )
        sizes = self._sizes(log_alpha)
        terms = (1 + sizes) / 2 * torch.log1p(
            squared * torch.exp(-2 * log_alpha)
        ) + sizes * log_alpha
        n_columns = prediction.size(1)
        if self._frac == 0:
            self._last_bound = float(
                (sizes * log_alpha.detach())
                .expand_as(terms)
                .sum(dim=-1)
                .mean()
                / n_columns
            )
        return (1 - self._frac) * terms.sum(dim=-1) / n_columns

    def _heteroscedastic(
        self, prediction: Tensor, target: Tensor, uncertainty: Tensor
    ) -> Tensor:
        return self._frac * torch.mean(
            torch.log1p((torch.abs(prediction - target) / uncertainty) ** 2)
            + torch.log(uncertainty),
            dim=-1,
        )

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        assert prediction.dim() == 2
        if target.dim() != prediction.dim():
            target = target.squeeze(1)

        conditions = None
        if self._n_conditions > 0:
            conditions = target[:, -self._n_conditions :]
            target = target[:, : -self._n_conditions]

        # Columns beyond the target width are per-element scales.
        uncertainty = prediction[:, target.size(1) :]
        prediction = prediction[:, : target.size(1)]
        assert prediction.size() == target.size(), (
            f"Prediction size {prediction.size()} and target size "
            f"{target.size()} do not match."
        )

        if self._frac == 0:
            if uncertainty.shape[1] > 0:
                self.warning_once(
                    "Uncertainty columns are provided but `frac` is 0; "
                    "they are ignored."
                )
            return self._homoscedastic(prediction, target, conditions)

        uncertainty = torch.clamp(uncertainty, min=1e-6)
        if self._frac == 1:
            return self._heteroscedastic(prediction, target, uncertainty)
        return self._homoscedastic(prediction, target) + self._heteroscedastic(
            prediction, target, uncertainty
        )


class spCauchyLoss(LossFunction):
    """Spherical Cauchy negative log-likelihood.

    The prediction holds a direction `mu` (first `d` columns, unit norm) and
    a non-negative magnitude `k` (last column), mapped to the concentration
    `rho = k / (1 + k)` in `[0, 1)`. The spherical Cauchy density on the unit
    sphere in `d` dimensions is

        f(x) = C_d * ((1 - rho^2) / (1 + rho^2 - 2 rho mu.x))^(d - 1),

    with a constant `C_d` independent of `rho`. In terms of `k` the ratio is
    `(1 + 2k) / (1 + 2k(1 + k)(1 - mu.x))`, which is evaluated directly to
    stay accurate for large `k` (where `1 - rho` underflows). The constant is
    omitted from the loss.

    Distribution: Kato, S. & McCullagh, P. (2020), "Some properties of a
    Cauchy family on the sphere derived from the Möbius transformations",
    Bernoulli 26(4), https://arxiv.org/abs/1510.07679. Regression framework:
    Tsagris, M., Papastamoulis, P. & Kato, S. (2025), Statistics and
    Computing 35:51, https://arxiv.org/abs/2409.03292.

    With a fixed concentration `rho` the prediction holds the direction only
    and the loss is, up to a constant, the rotation-invariant Cauchy loss
    `(d - 1) * log(1 + |mu - x|**2 / alpha**2)` of the chord between
    prediction and target, with `alpha**2 = (1 - rho)**2 / rho`.
    """

    def __init__(
        self, rho: Optional[float] = None, dim: int = 3, **kwargs: Any
    ) -> None:
        """Construct spCauchyLoss.

        Args:
            rho: Fixed concentration in `[0, 1)`. If None (default), the
                concentration is predicted per event (last column of the
                prediction).
            dim: Dimension `d` of the vectors; only used for the lower
                bound of the loss with a fixed `rho`.
        """
        super().__init__(**kwargs)
        assert rho is None or 0 <= rho < 1, "`rho` must be in [0, 1)."
        self._rho = rho
        self._dim = dim

    @property
    def alpha(self) -> float:
        """Return the scale of the equivalent Cauchy loss on the chord.

        `alpha**2 = (1 - rho)**2 / rho` for a fixed `rho`, approximately the
        angular scale in radians. Setting it changes `rho`, e.g. to anneal
        the loss from a wide to a narrow scale. Not available for a
        predicted concentration.
        """
        if self._rho is None:
            raise AttributeError("The concentration is predicted.")
        return float((1 - self._rho) / np.sqrt(self._rho))

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if self._rho is None:
            raise AttributeError("The concentration is predicted.")
        # k (1 + k) = 1 / alpha^2 with k = rho / (1 - rho)
        k = 0.5 * (np.sqrt(1 + 4 / alpha**2) - 1)
        self._rho = float(k / (1 + k))

    # Same attribute name as the scale of `CauchyLoss`.
    _alpha = alpha

    @property
    def lower_bound(self) -> Optional[float]:
        """Return the loss of an exact prediction for a fixed `rho`.

        With a predicted concentration the loss is unbounded.
        """
        if self._rho is None:
            return None
        k = self._rho / (1 - self._rho)
        return float(-(self._dim - 1) * np.log1p(2 * k))

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        assert prediction.dim() == 2
        assert target.dim() == 2
        assert prediction.size(0) == target.size(0)

        prediction = prediction.float()
        if self._rho is None:
            # Last column is the magnitude setting the concentration rho.
            dim = prediction.size(1) - 1
            k = prediction[:, dim]
        else:
            dim = prediction.size(1)
            assert dim == self._dim
            k = torch.full_like(prediction[:, 0], self._rho / (1 - self._rho))
        assert dim > 1
        assert target.size(1) == dim

        mu = prediction[:, :dim]
        # 1 - mu.x from the chord between the unit vectors: `1 - dot` would
        # round to steps of ~6e-8 (0.02 deg) in single precision.
        one_minus_dot = 0.5 * ((mu - target.float()) ** 2).sum(dim=-1)
        log_density = torch.log1p(2.0 * k) - torch.log1p(
            2.0 * k * (1.0 + k) * one_minus_dot
        )
        return -(dim - 1) * log_density
