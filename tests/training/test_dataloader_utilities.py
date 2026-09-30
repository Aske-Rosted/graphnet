"""Unit tests for dataloader utilities.

@NOTE: These utility methods should be deprecated in favour of the indicated
member methods in `DataLoader`.
"""

from typing import Tuple

import pytest
import torch

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.constants import TEST_PARQUET_DATA, TEST_SQLITE_DATA
from graphnet.models.graphs import KNNGraph
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.training.utils import make_train_validation_dataloader

# Configuration
NB_EVENTS_TOTAL = 5

graph_definition = KNNGraph(
    detector=IceCubeDeepCore(),
    node_definition=NodesAsPulses(),
    nb_nearest_neighbours=8,
    input_feature_names=FEATURES.DEEPCORE,
)


# Unit test(s)
def test_none_selection() -> None:
    """Test agreement of the two ways to calculate this loss."""
    (
        train_dataloader,
        test_dataloader,
    ) = make_train_validation_dataloader(
        db=TEST_SQLITE_DATA,
        graph_definition=graph_definition,
        selection=None,
        pulsemaps=["SRTInIcePulses"],
        features=FEATURES.DEEPCORE,
        truth=TRUTH.DEEPCORE,
        batch_size=1,
    )

    assert len(train_dataloader) + len(test_dataloader) == NB_EVENTS_TOTAL


@pytest.mark.parametrize(
    "selection",
    [
        (0, 1, 2, 3, 4),
        (0, 1, 3, 4),
        (0, 1),
    ],
)
def test_array_selection(selection: Tuple[int]) -> None:
    """Test agreement of the two ways to calculate this loss."""
    train_dataloader, test_dataloader = make_train_validation_dataloader(
        db=TEST_SQLITE_DATA,
        graph_definition=graph_definition,
        selection=list(selection),
        pulsemaps=["SRTInIcePulses"],
        features=FEATURES.DEEPCORE,
        truth=TRUTH.DEEPCORE,
        batch_size=1,
    )

    assert len(train_dataloader) + len(test_dataloader) == len(selection)


def test_empty_selection() -> None:
    """Test agreement of the two ways to calculate this loss."""
    try:
        _ = make_train_validation_dataloader(
            db=TEST_SQLITE_DATA,
            graph_definition=graph_definition,
            selection=list(),
            pulsemaps=["SRTInIcePulses"],
            features=FEATURES.DEEPCORE,
            truth=TRUTH.DEEPCORE,
            batch_size=1,
        )
        assert False  # Is expected to throw `ValueError`.
    except ValueError:
        pass


def test_parquet() -> None:
    """Test agreement of the two ways to calculate this loss."""
    try:
        _ = make_train_validation_dataloader(
            db=TEST_PARQUET_DATA,
            graph_definition=graph_definition,
            selection=None,
            pulsemaps=["SRTInIcePulses"],
            features=FEATURES.DEEPCORE,
            truth=TRUTH.DEEPCORE,
            batch_size=1,
        )
        assert False  # Is expected to throw `AssertionError`.
    except AssertionError as e:
        assert str(e).startswith("Format of input file")
        assert str(e).endswith("is not supported.")


def test_compute_budget_bucketing() -> None:
    """Mini-batches respect the budget and keep all multi-node graphs."""
    from torch_geometric.data import Data
    from graphnet.training.utils import collator_compute_budget_bucketing

    lengths = [1, 2, 3, 5, 8, 13, 21, 4, 4, 30]
    graphs = [Data(x=torch.zeros(n, 1), n_tokens=n) for n in lengths]
    collator = collator_compute_budget_bucketing(
        max_compute=200, parameter="n_tokens", gamma=2.0
    )
    batches = collator(graphs)

    kept = sorted(int(n) for b in batches for n in b.n_tokens)
    assert kept == sorted(n for n in lengths if n > 1)
    for batch in batches:
        max_length = int(batch.n_tokens.max())
        # A single graph may exceed the budget on its own.
        assert batch.num_graphs == 1 or (
            batch.num_graphs * max_length**2 <= 200
        )
