"""Implementation of the NodeTimeRNN model.

(cannot be used as a standalone model)
"""
import torch

from graphnet.models.gnn.gnn import GNN
from graphnet.utilities.config import save_model_config
from torch_geometric.data import Data


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
    ) -> None:
        """Construct `NodeTimeRNN`.

        Args:
            nb_inputs: Number of features in the input data.
            hidden_size: Number of features for the RNN output and hidden layers.
            num_layers: Number of layers in the RNN.
            nb_neighbours: Number of neighbours to use when reconstructing the graph representation.
        """
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._nb_inputs = nb_inputs
        super().__init__(nb_inputs, hidden_size + 5)

        self._rnn = torch.nn.GRU(
            num_layers=self._num_layers,
            input_size=self._nb_inputs,
            hidden_size=self._hidden_size,
            batch_first=True,
            dropout=RNN_dropout,
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Apply learnable forward pass to the GNN."""
        cutter = data.cutter.cumsum(0)[:-1]
        time_series = torch.nn.utils.rnn.pack_sequence(data.time_series.tensor_split(cutter.cpu()), enforce_sorted=False)

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
