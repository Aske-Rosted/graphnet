"""Implementation of the NodeTimeRNN model.

(cannot be used as a standalone model)
"""
import torch

from torch import Tensor
from graphnet.models.gnn.gnn import GNN
from graphnet.utilities.config import save_model_config
from torch_geometric.data import Data

from typing import List, Tuple

import warnings


class Dom_Window_RNN(GNN):
    """Implementation of the RNN model architecture.

    The model takes as input the typical DOM data format and transforms it into
    a time series of DOM activations pr. DOM. before applying a RNN layer and
    outputting the an RNN output for each DOM.
    """

    @save_model_config
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        nb_neighbours: int,
        dropout: float = 0.3,
        elementwise_affine: bool = True,
    ) -> None:
        """Construct `NodeTimeRNN`.

        Args:
            nb_inputs: Number of features in the input data.
            hidden_size: Number of features for the RNN output and hidden layers.
            num_layers: Number of layers in the RNN.
            nb_neighbours: Number of neighbours to use when reconstructing the graph representation.
        """
        self._nb_neighbours = nb_neighbours
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._nb_neighbours = nb_neighbours
        self._nb_inputs = 1
        super().__init__(1, hidden_size + 5)

        self._rnn = torch.nn.GRU(
            num_layers=self._num_layers,
            input_size=1,
            hidden_size=self._hidden_size,
            batch_first=True,
            dropout=dropout,
        )
        # self._rnn = script_lngru(
        #     input_size=1,
        #     hidden_size=self._hidden_size,
        #     num_layers=self._num_layers,
        #     dropout=dropout,
        #     elementwise_affine=elementwise_affine,
        # )
    def forward(self, data: Data) -> torch.Tensor:
        """Apply learnable forward pass to the GNN."""
        data.time_series = data.time_series.reshape(data.time_series.shape[0],data.time_series.shape[1],1)

        # apply rnn layer
        # initial_state = torch.zeros(self._num_layers,data.time_series.shape[1],self._hidden_size,device=data.time_series.device)
        rnn_out = self._rnn(data.time_series)[-1][0]  # apply rnn layer

        # add rnn output to node features
        data.x = torch.hstack(
            [data.x, rnn_out]
        ) 

        return data


# class GRULayer(torch.jit.ScriptModule):
#     def __init__(self, cell, *cell_args):
#         super(GRULayer, self).__init__()
#         self.cell = cell(*cell_args)

#     @torch.jit.script_method
#     def forward(self, input, state):
#         # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
#         inputs = input.unbind(0)
#         outputs = torch.jit.annotate(List[torch.Tensor], [])
#         for i in range(len(inputs)):
#             out, state = self.cell(inputs[i], state)
#             outputs += [out]
#         return torch.stack(outputs), state

# class LayerNormGRUCell(torch.jit.ScriptModule):
#     def __init__(self, input_size, hidden_size,elementwise_affine=True):
#         super(LayerNormGRUCell, self).__init__()
#         self._input_size = input_size
#         self._hidden_size = hidden_size
#         self._weight_ih = torch.nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
#         self._weight_hh = torch.nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))


#         ln = torch.nn.LayerNorm

#         if elementwise_affine:
#             self.ln_ih = ln(3 * hidden_size, elementwise_affine=True)
#             self.ln_hh = ln(3 * hidden_size, elementwise_affine=True)
#         else:
#             self.ln_ih = ln(3 * hidden_size, elementwise_affine=False)
#             self.ln_hh = ln(3 * hidden_size, elementwise_affine=False)
        

#     @torch.jit.script_method
#     def forward(self, input, state):
#         hx = state
#         igates = self.ln_ih(torch.mm(input, self._weight_ih.t()))
#         hgates = self.ln_hh(torch.mm(hx, self._weight_hh.t()))

#         reset_igates, update_igates, new_igates = igates.chunk(3, 1)
#         reset_hgates, update_hgates, new_hgates = hgates.chunk(3, 1)

#         reset = reset_igates + reset_hgates
#         update = update_igates + update_hgates

#         reset = torch.sigmoid(reset)
#         update = torch.sigmoid(update)
#         new = torch.tanh(new_igates + reset * new_hgates)
        
#         hy = (1 - update) * new + update * hx

#         return hy, hy
    

# def init_stacked_gru(num_layers, layer, first_layer_args, other_layer_args):
#     layers = [layer(*first_layer_args)] + [layer(*other_layer_args)
#                                            for _ in range(num_layers - 1)]
#     return torch.nn.ModuleList(layers)


# class StackedGRUWithDropout(torch.jit.ScriptModule):
#     # Necessary for iterating through self.layers and dropout support
#     __constants__ = ['layers', 'num_layers']

#     def __init__(self, num_layers, layer, first_layer_args, other_layer_args,dropout=0.4):
#         super(StackedGRUWithDropout, self).__init__()
#         self.layers = init_stacked_gru(num_layers, layer, first_layer_args,
#                                         other_layer_args)
#         # Introduces a Dropout layer on the outputs of each LSTM layer except
#         # the last layer, with dropout probability = 0.4.
#         self.num_layers = num_layers

#         if (num_layers == 1):
#             warnings.warn("dropout lstm adds dropout layers after all but last "
#                           "recurrent layer, it expects num_layers greater than "
#                           "1, but got num_layers = 1")

#         self.dropout_layer = torch.nn.Dropout(dropout)

#     @torch.jit.script_method
#     def forward(self, input, states):
#         # type: (Tensor, Tensor) -> Tuple[Tensor, List[Tensor]]
#         # List[LSTMState]: One state per layer
#         output_states = torch.jit.annotate(List[Tensor], [])
#         output = input
#         # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
#         i = 0
#         for rnn_layer in self.layers:
#             state = states[i]
#             output, out_state = rnn_layer(output, state)
#             # Apply the dropout layer except the last layer
#             if i < self.num_layers - 1:
#                 output = self.dropout_layer(output)
#             output_states += [out_state]
#             i += 1
#         return output, output_states
    
# def script_lngru(input_size, hidden_size, num_layers,dropout=False,elementwise_affine=True):
#     '''Returns a ScriptModule that mimics a PyTorch native LSTM.'''

#     # The following are not implemented.
#     stack_type = StackedGRUWithDropout
#     layer_type = GRULayer

#     return stack_type(num_layers, layer_type,
#                       first_layer_args=[LayerNormGRUCell, input_size, hidden_size,elementwise_affine], other_layer_args=[LayerNormGRUCell, hidden_size * 1,                                      hidden_size,elementwise_affine], dropout=dropout)

