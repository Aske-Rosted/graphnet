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


def test_attention_basis_factors_match_full_encoding() -> None:
    """`Attention_rel` with K basis factors equals the built encoding."""
    from graphnet.models.components.attention_blocks import Attention_rel

    torch.manual_seed(4)
    attention = Attention_rel(16, num_heads=2, attn_drop=0.3).double()
    x = torch.randn(2, 7, 16, dtype=torch.float64)
    phi = torch.randn(2, 3, 7, 7, dtype=torch.float64)
    weight = torch.randn(3, 8, dtype=torch.float64)
    bias = torch.randn(8, dtype=torch.float64)
    encoding = torch.einsum("bkij,kc->bijc", phi, weight) + bias
    for train in (False, True):  # with dropout the rows do not sum to one
        attention.train(train)
        torch.manual_seed(5)
        factored = attention(x, x, x, (phi, weight, bias))
        torch.manual_seed(5)
        full = attention(x, x, x, encoding)
        assert torch.allclose(factored, full, rtol=1e-10, atol=1e-12)


def test_per_block_linear_rel_bias_matches_shared_encoder() -> None:
    """Per-block projections without widths reproduce the affine bias."""
    torch.manual_seed(6)
    reference = _net(embed_bias=False).double()
    per_block = _net(embed_bias=False, per_block_rel_bias=True).double()
    state = reference.state_dict()
    state["rel_proj.0.weight"] = state.pop("rel_pos.projection.weight")
    state["rel_proj.0.bias"] = state.pop("rel_pos.projection.bias")
    per_block.load_state_dict(state)
    batch = _double_batch([5, 12])
    assert torch.allclose(
        per_block(batch), reference(batch), rtol=1e-10, atol=1e-12
    )


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


def test_shared_basis_projection_and_argument_checks() -> None:
    """Without `per_block_rel_bias` one projection serves all rel blocks."""
    net = _net(
        n_attention_blocks=4,
        n_rel=2,
        inject_cls_after=2,
        cross_attention=[2, 3],
        embed_bias=False,
        rel_bias_widths=[0.2],
    )
    assert len(net.rel_proj) == 1
    assert net(_batch([4, 7])).shape == (2, 24)
    for kwargs in (
        dict(embed_bias=True, rel_bias_widths=[0.2]),
        dict(embed_bias=False, rel_bias_widths=[0.0]),
        dict(
            embed_bias=False,
            n_rel=0,
            inject_cls_after=1,
            per_block_rel_bias=True,
        ),
    ):
        try:
            _net(**kwargs)
        except AssertionError:
            continue
        raise AssertionError(f"accepted {kwargs}")


def _mha_and_qk_norm(
    dim: int = 32, heads: int = 4
) -> "tuple[torch.nn.MultiheadAttention, Any]":
    from graphnet.models.components.attention_blocks import QKNormAttention

    torch.manual_seed(0)
    mha = torch.nn.MultiheadAttention(dim, heads, batch_first=True).eval()
    with torch.no_grad():
        mha.in_proj_bias.normal_()
        mha.out_proj.bias.normal_()
    attention = QKNormAttention(dim, heads).eval()
    missing, unexpected = attention.load_state_dict(
        mha.state_dict(), strict=False
    )
    assert missing == ["log_temperature"] and unexpected == []
    return mha, attention


def test_qk_norm_attention_projections_match_multihead_attention() -> None:
    """Without the normalization `QKNormAttention` is the loaded
    `nn.MultiheadAttention`, with float, boolean and 2D masks."""
    mha, attention = _mha_and_qk_norm()
    scale = attention.head_dim**-0.5
    attention._normalize = lambda q, k: (q * scale, k)  # type: ignore
    x = torch.randn(3, 7, 32)
    padding = torch.zeros(3, 7, dtype=torch.bool)
    padding[0, 5:] = True
    padding[2, 3:] = True
    float_padding = torch.zeros(3, 7).masked_fill(padding, -torch.inf)
    causal = torch.triu(torch.ones(7, 7, dtype=torch.bool), 1)
    for masks in (
        {},
        {"key_padding_mask": float_padding},
        {"key_padding_mask": padding},
        {"key_padding_mask": padding, "attn_mask": causal},
    ):
        with torch.no_grad():
            reference = mha(x, x, x, need_weights=False, **masks)[0]
            output = attention(x, x, x, **masks)[0]
        assert torch.allclose(output, reference, atol=1e-5), masks


def test_qk_norm_attention_ignores_query_and_key_scale() -> None:
    """The logits are `tau * cos(q, k)`: scaling the query and key projections
    (weights and biases) leaves the output unchanged."""
    _, attention = _mha_and_qk_norm()
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        reference = attention(x, x, x)[0]
        attention.in_proj_weight[:64] *= 50.0
        attention.in_proj_bias[:64] *= 50.0
        assert torch.allclose(attention(x, x, x)[0], reference, atol=1e-5)
        attention.log_temperature += 1.0
        assert not torch.allclose(attention(x, x, x)[0], reference, atol=1e-3)


def test_qk_norm_backbone() -> None:
    """`qk_norm` swaps the attention of the plain and token-only blocks, adds
    only temperatures, keeps padding local and trains everything."""
    from graphnet.models.components.attention_blocks import QKNormAttention

    kwargs: Dict[str, Any] = dict(
        n_attention_blocks=4,
        n_rel=2,
        inject_cls_after=2,
        cross_attention=[2, 3],
        embed_bias=False,
        per_block_rel_bias=True,
        rel_bias_widths=[0.1, 0.3, 1.0],
    )
    plain = _net(**kwargs)
    net = _net(qk_norm=True, **kwargs)
    assert not any(
        isinstance(m, QKNormAttention) for m in plain.modules()
    ), "qk_norm=False must keep nn.MultiheadAttention"
    for block in list(net.ESA[2:]) + list(net.xTAMS):
        assert isinstance(block.attn, QKNormAttention)
    new_keys = set(net.state_dict()) - set(plain.state_dict())
    assert new_keys == {
        "ESA.2.attn.log_temperature",
        "ESA.3.attn.log_temperature",
        "xTAMS.0.attn.log_temperature",
        "xTAMS.1.attn.log_temperature",
    }
    assert set(plain.state_dict()) <= set(net.state_dict())
    assert new_keys <= net.no_weight_decay()

    batch = _batch([5, 9])
    net.train()
    net(batch).square().sum().backward()
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, name
    net.eval()
    with torch.no_grad():
        together = net(batch)
        alone = net(Batch.from_data_list([batch.get_example(0)]))
    assert torch.allclose(together[0], alone[0], atol=1e-5)
