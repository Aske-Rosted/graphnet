"""Unit tests for `array_to_sequence`."""

from typing import List

import pytest
import torch

from graphnet.models.utils import array_to_sequence

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _batch(lengths: List[int], device: str) -> tuple:
    generator = torch.Generator().manual_seed(len(lengths))
    x = torch.randn(sum(lengths), 6, generator=generator).to(device)
    batch_idx = torch.repeat_interleave(
        torch.arange(len(lengths)), torch.tensor(lengths)
    ).to(device)
    return x, batch_idx


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "lengths", [[1], [3, 1, 7], [5, 5, 5, 2], [40, 1, 13]]
)
def test_known_lengths_give_the_same_sequence(
    device: str, lengths: List[int]
) -> None:
    """Lengths from the CPU give the result of counting on the device."""
    x, batch_idx = _batch(lengths, device)
    counted = array_to_sequence(x, batch_idx)
    given = array_to_sequence(x, batch_idx, seq_length=lengths)
    assert torch.equal(counted[0], given[0])
    assert torch.equal(counted[1], given[1])
    assert counted[2].tolist() == given[2].tolist() == lengths


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_known_lengths_do_not_synchronise() -> None:
    """With the lengths given, the conversion never waits for the GPU."""
    lengths = [12, 3, 30]
    x, batch_idx = _batch(lengths, "cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        array_to_sequence(x, batch_idx, seq_length=lengths)
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_excluding_value_is_rejected() -> None:
    """Inputs containing the excluding value are rejected in both modes."""
    x, batch_idx = _batch([2, 3], "cpu")
    x[1, 2] = torch.inf
    with pytest.raises(ValueError):
        array_to_sequence(x, batch_idx)
    with pytest.raises(RuntimeError):
        array_to_sequence(x, batch_idx, seq_length=[2, 3])
