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
from torch.distributions import Beta
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

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        return torch.mean(torch.abs(prediction - target), dim=-1)


class MSELoss(LossFunction):
    """Mean squared error loss."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        # Check(s)
        assert prediction.dim() == 2
        if target.dim() != prediction.dim():
            target = target.squeeze(1)
        assert prediction.size() == target.size()

        elements = torch.mean((prediction - target) ** 2, dim=-1)
        return elements


class RMSEAdjustedLoss(LossFunction):

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:

        assert prediction.dim() == 2
        assert prediction.size() == target.size()

        elements = torch.mean(
            (prediction - target) ** 2 / ((1 + target) ** (0.5)), dim=-1
        )
        return elements


class RMSELoss(MSELoss):
    """Root mean squared error loss."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        # Check(s)
        elements = super()._forward(prediction, target)
        elements = torch.sqrt(elements)
        return elements


class PoissonLoss(LossFunction):

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:

        loss_func = nn.PoissonNLLLoss(log_input=False, reduction="none")
        test = loss_func(prediction.float(), target.float())
        return test


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
        ratio: float = 1,
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

        self._loss = nn.CrossEntropyLoss(
            weight=torch.Tensor([ratio, 1]), reduction="none"
        )

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


class FocalCrossEntropyLoss(LossFunction):
    """Compute cross-entropy loss for classification tasks.

    Predictions are an [N, num_class]-matrix of logits (i.e., non-softmax'ed
    probabilities), and targets are an [N,1]-matrix with integer values in
    (0, num_classes - 1).
    """

    def __init__(
        self,
        options: Union[int, List[Any], Dict[Any, int]],
        ratio: float = 1,
        gamma: float = 2,
        alpha: float = 1,
        *args: Any,
        **kwargs: Any,
    ):
        """Construct CrossEntropyLoss."""
        # Base class constructor
        super().__init__(*args, **kwargs)

        # Member variables
        self._gamma = gamma
        self._alpha = alpha
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

        self._loss = nn.CrossEntropyLoss(
            weight=torch.Tensor([ratio, 1]), reduction="none"
        )

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

        cross_loss = self._loss(prediction.float(), target_one_hot.float())

        a_t = self._alpha * target_one_hot.float() + (1 - self._alpha) * (
            1 - target_one_hot.float()
        )
        # p_t = prediction.float()*target + (1-prediction.float())*(1-target_one_hot.float())
        p_t = torch.exp(-cross_loss)
        return (1 - p_t) ** self._gamma * cross_loss


class BinaryCrossEntropyLoss(LossFunction):
    """Compute binary cross entropy loss.

    Predictions are vector probabilities (i.e., values between 0 and 1), and
    targets should be 0 and 1.
    """

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return binary_cross_entropy(
            prediction.float(), target.float(), reduction="none"
        )


class BinaryCrossEntropyLossLogits(LossFunction):
    """Compute binary cross entropy loss.

    Predictions are vector probabilities (i.e., values between 0 and 1), and
    targets should be 0 and 1.
    """

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return binary_cross_entropy_with_logits(
            prediction.float(), target.float(), reduction="none"
        )


class FocalBinaryCrossEntropyLoss(LossFunction):
    """Compute binary cross entropy loss.

    Predictions are vector probabilities (i.e., values between 0 and 1), and
    targets should be 0 and 1.
    """

    def __init__(
        self,
        gamma: float = 2,
        alpha: float = 0.25,
        logits: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        """Construct CrossEntropyLoss."""
        # Base class constructor
        super().__init__(*args, **kwargs)

        self._gamma = gamma
        self._alpha = alpha

        self._logits = logits

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if self._logits:
            binary_loss = binary_cross_entropy_with_logits(
                prediction.float(),
                target.float(),
                reduction="none",
            )
            prediction = torch.sigmoid(prediction.float())
        else:
            binary_loss = binary_cross_entropy(
                prediction.float(),
                target.float(),
                reduction="none",
            )

        a_t = self._alpha * target + (1 - self._alpha) * (1 - target)
        p_t = prediction * target + (1 - prediction) * (1 - target)
        return a_t * (1 - p_t) ** self._gamma * binary_loss


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
        """Backward pass."""
        kappa = ctx.saved_tensors[0]
        m = ctx.m
        dtype = ctx.dtype
        kappa = kappa.double().cpu().numpy()
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

    def __init__(
        self, kappa_switch: float = 100.0, contamination: float = 0.0
    ) -> None:
        """Construct VonMisesFisherLoss.

        Args:
            kappa_switch: The value of `kappa` at which the exact and approximate
                calculation of $log C_{m}(k)$ switch.
        """
        super().__init__()
        self._kappa_switch = kappa_switch
        self._contamination = contamination

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
        nu = m / 2.0 - 1  # the order of the Bessel function
        a = torch.sqrt((nu) ** 2 + kappa**2)
        return -a + nu * torch.log(nu + a)

    @classmethod
    def log_cmk(
        cls, m: int, kappa: Tensor, kappa_switch: float
    ) -> Tensor:  # pylint: disable=invalid-name
        """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss.

        Since `log_cmk_exact` is diverges for `kappa` >~ 700 (using float64
        precision), and since `log_cmk_approx` is unaccurate for small `kappa`,
        this method automatically switches between the two at `kappa_switch`,
        ensuring continuity at this point.
        """

        kappa_switch = torch.tensor([kappa_switch]).to(kappa.device)

        if kappa_switch > 0:

            mask_exact = kappa < kappa_switch

            # Ensure continuity at `kappa_switch`
            offset = cls.log_cmk_approx(m, kappa_switch) - cls.log_cmk_exact(
                m, kappa_switch
            )
            ret = cls.log_cmk_approx(m, kappa) - offset
            ret[mask_exact] = cls.log_cmk_exact(m, kappa[mask_exact])
        else:
            # If kappa_switch is 0, we always use the approximation
            ret = cls.log_cmk_approx(m, kappa)
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

        elements = -self.log_cmk(m, k, self._kappa_switch) - dotprod
        if self._contamination > 0.0:
            # Add contamination term
            uniform_log_prob = self._log_uniform_sphere_torch(
                m
            ) * torch.ones_like(elements)
            elements = torch.logsumexp(
                torch.stack(
                    [
                        np.log(1 - self._contamination) + elements,
                        np.log(self._contamination) + uniform_log_prob,
                    ],
                    dim=0,
                ),
                dim=0,
            )
        return elements

    def _log_uniform_sphere_torch(self, m):
        # m: integer >= 1 (ambient dimension)
        return -(
            torch.log(torch.tensor(2.0))
            + (m / 2.0) * torch.log(torch.tensor(torch.pi))
            - torch.lgamma(torch.tensor(m / 2.0))
        )

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


class VonMisesFisher3DLoss(VonMisesFisherLoss):
    """Von Mises-Fisher loss function vectors in the 3D plane."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Construct VonMisesFisher3DLoss."""
        super().__init__(*args, **kwargs)

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate von Mises-Fisher loss for a direction in the 3D.

        Args:
            prediction: Output of the model. Must have shape [N, 4] where
                columns 0, 1, 2 are predictions of `direction` and last column
                is an estimate of `kappa`.
            target: Target tensor, extracted from graph object.

        Returns:
            Elementwise von Mises-Fisher loss terms. Shape [N,]
        """
        target = target.reshape(-1, 3)
        # Check(s)
        assert prediction.dim() == 2 and prediction.size()[1] == 4
        assert target.dim() == 2
        assert prediction.size()[0] == target.size()[0]

        kappa = prediction[:, 3]
        p = kappa.unsqueeze(1) * prediction[:, [0, 1, 2]]
        return self._evaluate(p, target)


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


class RegressionAsMulticlassification(LossFunction):
    """Regression as multiclassification loss function.

    This loss function is used to train a regression model as a
    multiclassification model. The target is a continuous value, which is
    binned into discrete classes. The model output is a probability
    distribution over the classes.
    """

    def __init__(
        self,
        target_ranges: List[float],
        n_bins: int,
        single_target: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Construct RegressionAsMulticlassification.

        Args:
            target_ranges: List of ranges for each target.
            n_bins: Number of bins for each target.
            single_target: If True there is only a single
                correct bin, meaning that the prediction
                can be normalized to 1. If True, the
                prediction will be normalized to 1.
        """
        # Base class constructor
        super().__init__(*args, **kwargs)

        # Member variables
        self._target_ranges = target_ranges
        self._target_bin_sizes = n_bins
        self._single_target = single_target
        # Create the binning of the ranges

    def _forward(self, prediction, target):

        non_batch_dims = np.arange(len(target))[1:]
        # digitize the target
        digitized_target = []
        for i in range(len(target)):
            digitized_target.append(
                np.digitize(target[i], self._target_ranges)
            )
        # Create empty matrix in the shape of batch_size x n_bins
        matrix_dim = (
            np.ones((len(self._target_ranges)), dtype=int)
            * self._target_bin_sizes
        )
        # Create the batched matrix
        matrix = np.zeros((prediction.shape[0], *matrix_dim), dtype=int)
        # add the batch dimension to the digitized target
        digitized_target = np.stack(
            [np.arange(prediction.shape[0])] + digitized_target, axis=0
        )

        matrix[tuple(digitized_target)] += 1
        matrix = torch.tensor(matrix).float().to(prediction.device)

        if self._single_target:
            prediction = prediction / torch.sum(
                prediction, dim=non_batch_dims, keepdim=True
            )

        loss = binary_cross_entropy(
            prediction.float(), matrix.float(), reduction="none"
        )
        loss = torch.sum(loss, dim=non_batch_dims)
        return loss


# class CauchyLoss(LossFunction):
#     """Cauchy loss function."""
#     def __init__(self, alpha : int = 0.1, **kwargs: Any) -> None:
#         self._alpha = alpha
#         super().__init__(**kwargs)

#     def _forward(self, prediction: Tensor, target: Tensor, alpha: Optional[Tensor] = None) -> Tensor:
#         """Implement loss calculation."""
#         # Check(s)
#         if alpha is not None:
#             alpha = self._alpha

#         assert prediction.dim() == 2
#         if target.dim() != prediction.dim():
#             target = target.squeeze(1)
#         assert prediction.size() == target.size()
#         elements = torch.sum(alpha**2/2*(torch.log(1 + ((abs(prediction - target))/alpha)**2)), dim=-1)
#         return elements


class CauchyLoss(LossFunction):
    """Cauchy loss function with heterocedastic uncertainty."""

    def __init__(
        self,
        alpha: float = 1.0,
        frac: float = 1,
        cold_start: int = -1,
        phase_in_steps: int = 10000,
        **kwargs: Any,
    ) -> None:
        """Construct CauchyLoss.

        Args:
            alpha: A fixed alpha value used for the Cauchy loss.
                Defaults to 1.0.
                frac: A fraction of the loss that is calculated using the uncertainty
                1 means that the loss is fully heteroscedastic, 0 means that the loss is fully
                homoscedastic.
        """
        # send to device
        self._alpha = alpha
        self._frac = frac
        self._cfrac = frac
        self._cold_start = cold_start
        self._phase_in_steps = phase_in_steps
        self._phase_in_count = 0

        super().__init__(**kwargs)

    def determine_frac(self, epoch: int) -> None:
        """Determine the fraction of the loss that is calculated using the
        uncertainty.

        Args:
            epoch: The current epoch number.
        """
        assert (
            self._frac > 0
        ), "frac must be greater than 0 in order to phase in the uncertainty."

        if epoch < self._cold_start:
            self._cfrac = 1e-4
        elif self._cfrac < self._frac:
            self._phase_in_count += 1
            self._cfrac = self._cfrac * (
                self._phase_in_count / self._phase_in_steps
            )

    def homoscedastic(self, prediction, target) -> bool:
        """Calculate the homoscedastic loss."""
        elements = (1 - self._frac) * torch.mean(
            torch.log1p(((abs(prediction - target)) / self._alpha) ** 2)
            + np.log(self._alpha),
            dim=-1,
        )
        # offset to ensure positive loss values excact for the homoscedastic case
        # elements -= (1-self._frac) * np.log(self._alpha)
        # assert torch.all(elements >= 0), f"Loss values should be positive: but found {elements}: predictions{prediction}  target{target}"
        return elements

    def heteroscedastic(self, prediction, target, uncertainty) -> Tensor:
        """Calculate the heteroscedastic loss."""
        elements = self._frac * torch.mean(
            torch.log1p(((abs(prediction - target)) / uncertainty) ** 2)
            + torch.log(uncertainty),
            dim=-1,
        )
        # offset to ensure positive loss values in the heteroscedastic case we use the minimum value that the uncertainty can take
        # elements -= self._frac * np.log(1e-6)  # This is the minimum value that the uncertainty can take
        # assert torch.all(elements >= 0), f"Loss values should be positive: but found {elements}: predictions{prediction}  target{target}"
        return elements

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        # Check(s)

        if self._cold_start >= 0:
            self.determine_frac(self.current_epoch)

        assert prediction.dim() == 2
        if target.dim() != prediction.dim():
            target = target.squeeze(1)

        # Extract the uncertainty from the last column of the prediction
        uncertainty = prediction[:, target.size(1) :]
        prediction = prediction[:, : target.size(1)]

        assert (
            prediction.size() == target.size()
        ), f"Prediction size {prediction.size()} and target size {target.size()} do not match."

        if (uncertainty.shape[1] > 0) & (self._cfrac == 0.0):
            self.warning_once(
                "uncertainty is provided, but frac is set to 0. The uncertainty will be ignored."
            )
        # Ensure that uncertainty is non-negative
        if uncertainty.shape[1] > 0:
            uncertainty = torch.clamp(uncertainty, min=1e-6)

        if self._cfrac == 0:
            # If the loss is fully homoscedastic, we use the fixed alpha value
            elements = self.homoscedastic(prediction, target)
        elif self._cfrac == 1:
            # If the loss is fully heteroscedastic, we use the uncertainty
            elements = self.heteroscedastic(prediction, target, uncertainty)
        else:
            # If the loss is a mix of homoscedastic and heteroscedastic, we use both
            elements = self.homoscedastic(
                prediction, target
            ) + self.heteroscedastic(prediction, target, uncertainty)
        return elements


class CauchyVonMisesFisher3DLoss(EnsembleLoss):
    """Combine the VonMisesFisher3DLoss with CauchyLoss."""

    def __init__(
        self,
        vmfs_factor: float = 0.05,
        alpha=0.1,
        kappa_switch=0,
        contamination=0.0,
    ) -> None:
        """VonMisesFisher3DLoss with a Cauchy penality term.

            The VonMisesFisher3DLoss will be weighted with `vmfs_factor`.

        Args:
            vmfs_factor: A factor applied to the VonMisesFisher3DLoss term.
            Defaults ot 0.05.
        """

        loss_function = CauchyLoss(alpha=alpha, frac=0.0)
        prediction_keys = [[0, 1, 2], [0, 1, 2, 3]]
        super().__init__(
            loss_functions=[
                loss_function,
                VonMisesFisher3DLoss(
                    kappa_switch=kappa_switch, contamination=contamination
                ),
            ],
            loss_factors=[(1 - vmfs_factor), vmfs_factor],
            prediction_keys=prediction_keys,
        )


class GaussianLoss(LossFunction):
    """Gaussian loss function.

    This loss function is used to train a regression model with assuming
    Gaussian uncertainty.
    """

    def __init__(self, std: float = 0.1, **kwargs: Any) -> None:
        """Construct GaussianCauchyLoss."""
        self._std = std
        super().__init__(**kwargs)

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        # Check(s)
        assert prediction.dim() == 2
        if target.dim() != prediction.dim():
            target = target.squeeze(1)
        assert prediction.size() == target.size()
        loss = nn.GaussianNLLLoss(reduction="none")
        # Calculate the loss
        # The std is assumed to be constant, so we can use it directly
        # in the loss function.
        # The loss

        elements = loss(prediction, target, self._std)
        return elements


class HeterocedasticGaussianLoss(LossFunction):
    """Heteroscedastic Gaussian loss function.

    This loss function is used to train a regression model with heteroscedastic
    uncertainty. The model output is a mean and a standard deviation, which are
    used to calculate the loss.
    """

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        # Check(s)
        assert prediction.dim() == 2
        if target.dim() != prediction.dim():
            target = target.squeeze(1)
        assert all(prediction.size() == (target.size() * np.array([1, 2])))

        # Extract the mean and standard deviation from the prediction

        # It is important that you task does not allow for negative standard deviations
        std = prediction[:, target.size(1) :]
        prediction = prediction[:, : target.size(1)]
        # Calculate the loss
        loss = nn.GaussianNLLLoss(reduction="none")
        elements = loss(prediction, target, std)
        elements = torch.sum(elements, dim=-1)
        return elements


class GaussianVonMisesFisher3DLoss(EnsembleLoss):
    """Combine the VonMisesFisher3DLoss with CauchyLoss."""

    def __init__(
        self, vmfs_factor: float = 0.05, alpha=0.1, heteroscedastic=False
    ) -> None:
        """VonMisesFisher3DLoss with a Cauchy penality term.

            The VonMisesFisher3DLoss will be weighted with `vmfs_factor`.

        Args:
            vmfs_factor: A factor applied to the VonMisesFisher3DLoss term.
            Defaults ot 0.05.
        """
        if heteroscedastic:
            if alpha != 0.1:
                self.warning(
                    "Heteroscedastic Gaussian loss does not use the alpha parameter, "
                    "it is only used for the Gaussian loss. "
                    "The alpha parameter will be ignored."
                )
            loss_function = HeterocedasticGaussianLoss()
            prediction_keys = [[0, 1, 2, 3, 4, 5], [0, 1, 2, -1]]
        else:
            loss_function = GaussianLoss(std=alpha)
            prediction_keys = [[0, 1, 2], [0, 1, 2, 3]]
        super().__init__(
            loss_functions=[loss_function, VonMisesFisher3DLoss()],
            loss_factors=[1, vmfs_factor],
            prediction_keys=prediction_keys,
        )


class BetaNLLLoss(LossFunction):
    """Beta Negative Log Likelihood Loss.

    This loss function is used to train a regression model with Beta
    distribution uncertainty. The model output is a mean and a standard
    deviation, which are used to calculate the loss.
    """

    def __init__(self, eps=1e-5, **kwargs: Any) -> None:
        """Construct BetaNLLLoss."""
        self._eps = eps  # Small value to avoid division by zero
        super().__init__(**kwargs)

    def _beta_nll_loss(
        self, target: Tensor, alpha: Tensor, beta: Tensor
    ) -> Tensor:
        """Calculate the Beta Negative Log Likelihood Loss."""
        # clamp target to [0, 1] +/- self._eps
        target = (
            target * (1 - 2 * self._eps) + self._eps
        )  # maps [0,1] -> [eps, 1-eps]
        # Calculate the Beta NLL loss using the torc
        dist = torch.distributions.Beta(alpha, beta)
        return -dist.log_prob(target)

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        # Check(s)
        assert prediction.dim() == 2
        if target.dim() != prediction.dim():
            target = target.squeeze(1)
        # task should have 5 outputs: alpha, beta, and two additional outputs
        # which are the from alpha and the mean, variance, precision, calculated by
        assert all(prediction.size() == (target.size() * np.array([1, 5])))

        # Extract the mean and standard deviation from the prediction
        alpha = prediction[:, [0]]
        beta = prediction[:, [1]]
        # Calculate the loss
        return self._beta_nll_loss(target, alpha, beta)
