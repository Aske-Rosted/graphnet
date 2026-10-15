"""Unit tests for `graphnet.models.components.embedding`."""

import pytest
import torch

from graphnet.models.components.embedding import signed_four_distance


def test_signed_four_distance_without_synchronisation() -> None:
    """Same values as list indexing, and no GPU synchronisation."""
    torch.manual_seed(0)
    x = torch.randn(2, 9, 6)
    columns = (0, 2, 4, 5)
    pos = x[:, :, [0, 2, 4]]
    time = x[:, :, 5]
    interval = (pos[:, :, None] - pos[:, None, :]).pow(2).sum(-1) - (
        (time[:, :, None] - time[:, None, :]) * 0.6
    ).pow(2)
    expected = torch.sign(interval) * torch.sqrt(torch.abs(interval))
    assert torch.equal(signed_four_distance(x, 0.6, columns), expected)
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU for the synchronisation check")
    x = x.cuda()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        signed_four_distance(x, 0.6, columns)
    finally:
        torch.cuda.set_sync_debug_mode("default")
