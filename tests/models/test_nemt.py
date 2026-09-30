"""Unit tests for `NeutrinoEventMultitaskTransformer`."""

import os
from typing import Any, Dict

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from graphnet.constants import CONFIG_DIR
from graphnet.models import Model
from graphnet.models.gnn import NeutrinoEventMultitaskTransformer
from graphnet.utilities.config import ModelConfig


def _net(**kwargs: Any) -> NeutrinoEventMultitaskTransformer:
    args: Dict[str, Any] = dict(
        n_attention_blocks=3,
        n_rel=1,
        inject_cls_after=1,
        hidden_dim=32,
        num_heads=4,
        n_features=6,
        n_tasks=3,
        shared_tokens=2,
        out_dim=8,
        cross_attention=[1, 2],
    )
    args.update(kwargs)
    return NeutrinoEventMultitaskTransformer(**args).eval()


def _batch(sizes: list) -> Batch:
    torch.manual_seed(0)
    return Batch.from_data_list([Data(x=torch.randn(n, 6)) for n in sizes])


def test_output_shape() -> None:
    """One `out_dim` block per task and event."""
    net = _net()
    assert net.nb_outputs == 3 * 8
    assert net(_batch([5, 9])).shape == (2, 24)


def test_output_shape_with_token_multiplier() -> None:
    """Several tokens per task are projected to a single `out_dim` block."""
    net = _net(token_multiplier=2)
    assert net(_batch([5, 9])).shape == (2, 24)


def test_padding_does_not_leak_between_events() -> None:
    """An event's output does not depend on the other events in the batch."""
    net = _net()
    batch = _batch([5, 9])
    with torch.no_grad():
        together = net(batch)
        alone = net(Batch.from_data_list([batch.get_example(0)]))
    assert torch.allclose(together[0], alone[0], atol=1e-5)


def test_example_config_builds_and_runs() -> None:
    """The example multitask config builds and predicts."""
    config = ModelConfig.load(
        os.path.join(
            CONFIG_DIR, "models", "example_multitask_transformer_model.yml"
        )
    )
    model = Model.from_config(config, trust=True).eval()

    rng = np.random.default_rng(0)
    pulses = np.column_stack(
        [
            rng.choice([-250.0, 0.0, 250.0], size=(40, 3)),
            rng.uniform(0, 2000, size=40),
            rng.uniform(0.2, 5, size=40),
        ]
    )
    graph = model._data_representation(
        input_features=pulses,
        input_feature_names=["dom_x", "dom_y", "dom_z", "dom_time", "charge"],
    )
    with torch.no_grad():
        preds = model(Batch.from_data_list([graph]))
    assert [p.shape[1] for p in preds] == [1, 1, 4]
