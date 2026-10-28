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


def test_spacetime_encoder_with_sinusoid_matches_epjc() -> None:
    """With the EPJ-C constants the sinusoidal path equals the EPJ-C one."""
    from graphnet.models.components.embedding import (
        SpacetimeEncoder,
        SpacetimeEncoderEPJC,
    )

    torch.manual_seed(0)
    reference = SpacetimeEncoderEPJC(seq_length=8)
    encoder = SpacetimeEncoder(seq_length=8, time_scale=3e4 / 500 * 3e-1)
    encoder.load_state_dict(reference.state_dict())
    x = torch.randn(2, 5, 4) * 0.01
    assert torch.allclose(encoder(x), reference(x))


def _pair(**kwargs: Any) -> tuple:
    """Two identical networks, with and without the factored bias."""
    torch.manual_seed(1)
    factored = _net(embed_bias=False, factor_rel_bias=True, **kwargs)
    full = _net(embed_bias=False, factor_rel_bias=False, **kwargs)
    full.load_state_dict(factored.state_dict())
    return factored.double(), full.double()


def _double_batch(sizes: list) -> Batch:
    batch = _batch(sizes)
    batch.x = batch.x.double()
    return batch


def test_factored_rel_bias_matches_full_encoding() -> None:
    """Factored relative bias: same outputs and gradients."""
    factored, full = _pair()
    batch = _double_batch([5, 17, 9])
    out_factored, out_full = factored(batch), full(batch)
    assert torch.allclose(out_factored, out_full, rtol=1e-10, atol=1e-12)
    out_factored.square().sum().backward()
    out_full.square().sum().backward()
    for (name, a), (_, b) in zip(
        factored.named_parameters(), full.named_parameters()
    ):
        if a.grad is None:
            assert b.grad is None, name
            continue
        assert torch.allclose(a.grad, b.grad, rtol=1e-8, atol=1e-10), name


def test_rel_bias_basis_extends_the_linear_bias() -> None:
    """Zero weights on the Gaussian terms give the linear bias; non-zero
    weights change the output."""
    torch.manual_seed(7)
    linear = _net(embed_bias=False, per_block_rel_bias=True).double()
    basis = _net(
        embed_bias=False, per_block_rel_bias=True, rel_bias_widths=[0.1, 1.0]
    ).double()
    state = linear.state_dict()
    weight = state["rel_proj.0.weight"]
    state["rel_proj.0.weight"] = torch.cat(
        [weight, torch.zeros(weight.shape[0], 2, dtype=weight.dtype)], 1
    )
    basis.load_state_dict(state)
    batch = _double_batch([6, 10])
    assert torch.allclose(basis(batch), linear(batch), rtol=1e-10, atol=1e-12)
    with torch.no_grad():
        basis.rel_proj[0].weight[:, 1:] = 0.5
    assert not torch.allclose(basis(batch), linear(batch))


def test_per_block_rel_bias_trains_every_projection() -> None:
    """Two rel blocks with own projections: all get gradients, padding
    stays local."""
    net = _net(
        n_attention_blocks=4,
        n_rel=2,
        inject_cls_after=2,
        cross_attention=[2, 3],
        embed_bias=False,
        per_block_rel_bias=True,
        rel_bias_widths=[0.1, 0.3, 1.0],
    )
    assert len(net.rel_proj) == 2
    assert net.rel_proj[0].weight.shape == (8, 4)
    assert not hasattr(net, "rel_pos")
    batch = _batch([5, 9])
    net(batch).square().sum().backward()
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, name
    with torch.no_grad():
        together = net(batch)
        alone = net(Batch.from_data_list([batch.get_example(0)]))
    assert torch.allclose(together[0], alone[0], atol=1e-5)
