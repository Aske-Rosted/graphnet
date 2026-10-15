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


def test_spacetime_encoder_uses_configured_time_column() -> None:
    """The interval is built from the configured time column and scale."""
    from graphnet.models.components.embedding import SpacetimeEncoder

    encoder = SpacetimeEncoder(
        seq_length=2,
        output_dim=1,
        columns=(0, 1, 2, 4),
        time_scale=0.6,
        clip=None,
        apply_sin_emb=False,
    )
    with torch.no_grad():
        encoder.projection.weight.fill_(1.0)
        encoder.projection.bias.fill_(0.0)
    # Same position, times 0 and 1 in column 4; column 3 is a decoy.
    x = torch.tensor([[[0.0, 0.0, 0.0, 5.0, 0.0], [0.0, 0.0, 0.0, -5.0, 1.0]]])
    interval = encoder(x)[0, :, :, 0]
    # Timelike separation: -(1 * 0.6)
    assert torch.allclose(interval, torch.tensor([[0.0, -0.6], [-0.6, 0.0]]))


def test_closed_gate_leaves_tokens_unchanged() -> None:
    """With the gates closed, the token-only blocks have no effect."""
    net = _net()
    batch = _batch([5, 9])
    with torch.no_grad():
        for gate in net.gates:
            gate.fill_(-1e4)
        reference = net(batch)
        for i in range(len(net.xTAMS)):
            net.xTAMS[i] = _RandomBlock()
        assert torch.allclose(net(batch), reference)


class _RandomBlock(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(x) * 100


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


def test_factored_rel_bias_matches_with_attention_dropout() -> None:
    """With attention dropout (rows no longer sum to one) in training."""
    factored, full = _pair(dropout=0.3)
    factored.train()
    full.train()
    batch = _double_batch([6, 11])
    torch.manual_seed(7)
    out_factored = factored(batch)
    torch.manual_seed(7)
    out_full = full(batch)
    assert torch.allclose(out_factored, out_full, rtol=1e-10, atol=1e-12)


def test_affine_factors_reproduce_the_encoding() -> None:
    """`d * w + b` from the factors equals the encoder output."""
    from graphnet.models.components.embedding import SpacetimeEncoder

    torch.manual_seed(2)
    x = torch.randn(3, 7, 6, dtype=torch.float64)
    for clip in (None, 0.5):
        encoder = SpacetimeEncoder(
            8, output_dim=8, clip=clip, apply_sin_emb=False
        ).double()
        distance, weight, bias = encoder.affine_factors(x)
        assert torch.allclose(
            distance.unsqueeze(-1) * weight + bias, encoder(x)
        )


def test_factor_rel_bias_has_no_effect_with_sinusoidal_embedding() -> None:
    """With `embed_bias` the encoding is not affine; the flag is ignored."""
    torch.manual_seed(3)
    a = _net(embed_bias=True, factor_rel_bias=True)
    b = _net(embed_bias=True, factor_rel_bias=False)
    b.load_state_dict(a.state_dict())
    batch = _batch([4, 9])
    assert torch.equal(a(batch), b(batch))
