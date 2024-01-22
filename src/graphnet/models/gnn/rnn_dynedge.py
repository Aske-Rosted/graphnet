"""RNN_DynEdge model implementation."""
from typing import List, Optional, Tuple, Union

import torch
from graphnet.models.gnn.gnn import GNN
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.gnn.dynedge_kaggle_tito import DynEdgeTITO
from graphnet.models.rnn.node_rnn import Node_RNN

# from graphnet.models.rnn.dom_window_rnn import Dom_Window_RNN
from graphnet.models.rnn.node_transformer import Node_Transformer

from graphnet.utilities.config import save_model_config
from torch_geometric.data import Data


class RNN_DynEdge(GNN):
    """The RNN_DynEdge model class.

    works only with 2 dimensional time series of (charge, time)
    """

    @save_model_config
    def __init__(
        self,
        nb_inputs: int,
        *,
        nb_neighbours: int = 8,
        RNN_layers: int = 2,
        RNN_hidden_size: int = 64,
        RNN_dropout: float = 0.5,
        features_subset: Optional[Union[List[int], slice]] = None,
        dyntrans_layer_sizes: Optional[List[Tuple[int, ...]]] = None,
        post_processing_layer_sizes: Optional[List[int]] = None,
        readout_layer_sizes: Optional[List[int]] = None,
        global_pooling_schemes: Optional[Union[str, List[str]]] = None,
        # add_global_variables_after_pooling: bool = False,
        embedding_dim: Optional[int] = None,
        n_head: int = 16,
    ):
        """Initialize the RNN_DynEdge model.

        Args:
            nb_inputs (int): Number of input features.
            nb_neighbours (int, optional): Number of neighbours to consider.
                Defaults to 8.
            RNN_layers (int, optional): Number of RNN layers.
                Defaults to 1.
            RNN_hidden_size (int, optional): Size of the hidden state of the RNN. Also determines the size of the output of the RNN.
                Defaults to 64.
            RNN_dropout (float, optional): Dropout to use in the RNN. Defaults  to 0.5.
            features_subset (Optional[Union[List[int], slice]], optional): Subset of features to use.
            dyntrans_layer_sizes (Optional[List[Tuple[int, ...]]], optional): List of tuples representing the sizes of the hidden layers of the DynTrans model.
            post_processing_layer_sizes (Optional[List[int]], optional): List of integers representing the sizes of the hidden layers of the post-processing model.
            readout_layer_sizes (Optional[List[int]], optional): List of integers representing the sizes of the hidden layers of the readout model.
            global_pooling_schemes (Optional[Union[str, List[str]]], optional): Pooling schemes to use. Defaults to None.
            embedding_dim (Optional[int], optional): Embedding dimension of the RNN. Defaults to None ie. no embedding.
            n_head (int, optional): Number of heads to use in the DynTrans model. Defaults to 16.
        """
        self._nb_neighbours = nb_neighbours
        self._nb_inputs = nb_inputs
        self._RNN_layers = RNN_layers
        self._RNN_hidden_size = RNN_hidden_size  # RNN_hidden_size
        self._RNN_dropout = RNN_dropout
        self._embedding_dim = embedding_dim
        self._n_head = n_head

        self._features_subset = features_subset
        if dyntrans_layer_sizes is None:
            dyntrans_layer_sizes = [
                (256, 256),
                (256, 256),
                (256, 256),
                (256, 256),
            ]
        else:
            dyntrans_layer_sizes = [
                tuple(layer_sizes) for layer_sizes in dyntrans_layer_sizes
            ]

        self._dyntrans_layer_sizes = dyntrans_layer_sizes
        self._post_processing_layer_sizes = post_processing_layer_sizes
        self._global_pooling_schemes = global_pooling_schemes
        # self._add_global_variables_after_pooling = (
        #     add_global_variables_after_pooling
        # )
        if readout_layer_sizes is None:
            readout_layer_sizes = [
                256,
                128,
            ]
        self._readout_layer_sizes = readout_layer_sizes

        super().__init__(nb_inputs, self._readout_layer_sizes[-1])

        self._rnn = Node_RNN(
            num_layers=self._RNN_layers,
            nb_inputs=2,
            hidden_size=self._RNN_hidden_size,
            # nb_neighbours=self._nb_neighbours,
            RNN_dropout=self._RNN_dropout,
            embedding_dim=self._embedding_dim,
        )

        # self._node_transformer = Node_Transformer(
        #     nb_inputs=2,
        #     hidden_size=self._RNN_hidden_size,
        #     RNN_dropout=self._RNN_dropout,
        #     num_layers=self._RNN_layers,
        # )

        # self._rnn = Dom_Window_RNN(
        #     num_layers=self._RNN_layers,
        #     hidden_size=self._RNN_hidden_size,
        #     nb_neighbours=self._nb_neighbours,
        #     dropout=self._RNN_dropout,
        # )

        # self._dynedge = DynEdge(
        #     nb_inputs=self._RNN_hidden_size + 5,
        #     nb_neighbours=self._nb_neighbours,
        #     dynedge_layer_sizes=self._dynedge_layer_sizes,
        #     post_processing_layer_sizes=self._post_processing_layer_sizes,
        #     readout_layer_sizes=self._readout_layer_sizes,
        #     global_pooling_schemes=self._global_pooling_schemes,
        #     add_global_variables_after_pooling=self._add_global_variables_after_pooling,
        # )
        self._dynedge_tito = DynEdgeTITO(
            nb_inputs=self._RNN_hidden_size + 5,
            # nb_neighbours=self._nb_neighbours,
            dyntrans_layer_sizes=self._dyntrans_layer_sizes,
            features_subset=self._features_subset,
            global_pooling_schemes=self._global_pooling_schemes,
            use_global_features=True,
            use_post_processing_layers=True,
            post_processing_layer_sizes=self._post_processing_layer_sizes,
            readout_layer_sizes=self._readout_layer_sizes,
            n_head=self._n_head,
            nb_neighbours=self._nb_neighbours,
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Apply learnable forward pass of the RNN and DynEdge models."""
        data = self._rnn(data)
        # data = self._node_transformer(data)
        readout = self._dynedge_tito(data)

        return readout
