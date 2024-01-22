"""Implementation of the NodeTimeRNN model.

(cannot be used as a standalone model)
"""
import torch

from graphnet.models.gnn.gnn import GNN
from graphnet.utilities.config import save_model_config
from torch_geometric.data import Data
from typing import Optional


class Node_RNN(GNN):
    """Implementation of the RNN model architecture.

    The model takes as input the typical DOM data format and transforms it into
    a time series of DOM activations pr. DOM. before applying a RNN layer and
    outputting the an RNN output for each DOM.
    """

    @save_model_config
    def __init__(
        self,
        nb_inputs: int,
        hidden_size: int,
        num_layers: int,
        RNN_dropout: float = 0.5,
        embedding_dim: int = 0,
    ) -> None:
        """Construct `NodeTimeRNN`.

        Args:
            nb_inputs: Number of features in the input data.
            hidden_size: Number of features for the RNN output and hidden layers.
            num_layers: Number of layers in the RNN.
            nb_neighbours: Number of neighbours to use when reconstructing the graph representation.
            RNN_dropout: Dropout fractio to use in the RNN. Defaults to 0.5.
            embedding_dim: Embedding dimension of the RNN. Defaults to no embedding.
        """
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._embedding_dim = embedding_dim
        self._nb_inputs = nb_inputs

        super().__init__(nb_inputs, hidden_size + 5)

        if self._embedding_dim != 0:
            self._nb_inputs = self._embedding_dim * 2 * nb_inputs

        self._rnn = torch.nn.GRU(
            num_layers=self._num_layers,
            input_size=self._nb_inputs,
            hidden_size=self._hidden_size,
            batch_first=True,
            dropout=RNN_dropout,
        )
        self._emb = SinusoidalPosEmb(dim=self._embedding_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """Apply learnable forward pass to the GNN."""
        cutter = data.cutter.cumsum(0)[:-1]
        if self._embedding_dim != 0:
            time_series = self._emb(data.time_series * 4096).reshape(
                (
                    data.time_series.shape[0],
                    self._embedding_dim * 2 * data.time_series.shape[-1],
                )
            )
        else:
            time_series = data.time_series

        time_series = torch.nn.utils.rnn.pack_sequence(
            time_series.tensor_split(cutter.cpu()), enforce_sorted=False
        )

        rnn_out = self._rnn(time_series)[-1][0]
        # s = 0
        # packed_sequences = []
        # for e in data.n_doms.cumsum(dim=0):
        #     packed_sequences.append(torch.nn.utils.rnn.pack_sequence(
        #         time_series[s:e], enforce_sorted=True
        #     ))
        #     # apply rnn layer
        #     rnn_out.append(self._rnn(ts)[-1][0])
        #     s = e
        # # x = self._rnn(x)[-1][0]  # apply rnn layer
        # rnn_out = torch.cat(rnn_out)

        data.x = torch.hstack(
            [data.x, rnn_out]
        )  # reintroduce x/y/z-coordinates and time of first/mean activation for each DOM
        return data


class SinusoidalPosEmb(torch.nn.Module):
    """Sinusoidal positional embedding layer."""

    def __init__(self, dim: int = 16, M: int = 10000) -> None:
        """Construct `SinusoidalPosEmb`.

        Args:
            dim: Embedding dimension.
            M: Number of frequencies.
        """
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable forward pass to the layer."""
        device = x.device
        half_dim = self.dim
        emb = torch.log(torch.tensor(self.M, device=device)) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
