"""Implementation of the DynEdge GNN model architecture."""
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum

from graphnet.models.components.layers import DynEdgeConv
from graphnet.utilities.config import save_model_config
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import calculate_xyzt_homophily
from torch_geometric.nn.pool import knn_graph

GLOBAL_POOLINGS = {
    "min": scatter_min,
    "max": scatter_max,
    "sum": scatter_sum,
    "mean": scatter_mean,
}


class DynEdgeHE(GNN):
    """DynEdge (dynamical edge convolutional) model."""

    @save_model_config
    def __init__(
        self,
        nb_inputs: int,
        *,
        nb_neighbours: int = 2,
        features_subset: Optional[Union[List[int], slice]] = None,
        dynedge_layer_sizes: Optional[List[Tuple[int, ...]]] = None,
        readout_layer_sizes: Optional[List[int]] = None,
        global_pooling_schemes: Optional[Union[str, List[str]]] = None,
        add_global_variables_after_pooling: bool = False,
        reduction_layers: Optional[List[bool]] = None,
    ):
        """Construct `DynEdgeHE`.

        Args:
            nb_inputs: Number of input features on each node.
            nb_neighbours: Number of neighbours to used in the k-nearest
                neighbour clustering which is performed after each (dynamical)
                edge convolution.
            features_subset: The subset of latent features on each node that
                are used as metric dimensions when performing the k-nearest
                neighbours clustering. Defaults to [0,1,2].
            dynedge_layer_sizes: The layer sizes, or latent feature dimenions,
                used in the `DynEdgeConv` layer. Each entry in
                `dynedge_layer_sizes` corresponds to a single `DynEdgeConv`
                layer; the integers in the corresponding tuple corresponds to
                the layer sizes in the multi-layer perceptron (MLP) that is
                applied within each `DynEdgeConv` layer. That is, a list of
                size-two tuples means that all `DynEdgeConv` layers contain a
                two-layer MLP.
                Defaults to [(128, 256), (336, 256), (336, 256), (336, 256)].
            readout_layer_sizes: Hidden layer sizes in the MLP following the
                post-processing _and_ optional global pooling. As this is the
                last layer(s) in the model, the last layer in the read-out
                yields the output of the `DynEdge` model. Defaults to [128,].
            global_pooling_schemes: The list global pooling schemes to use.
                Options are: "min", "max", "mean", and "sum".
            add_global_variables_after_pooling: Whether to add global variables
                after global pooling. The alternative is to  added (distribute)
                them to the individual nodes before any convolutional
                operations.
            reduction_layers: Whether to perform a reduction after the EdgeConv operation. Should be a list of booleans of the same length as dynedge_layer_sizes. Defaults to [False] * len(dynedge_layer_sizes).
        """
        # Latent feature subset for computing nearest neighbours in DynEdge.
        if features_subset is None:
            features_subset = slice(0, 3)

        # DynEdge layer sizes
        if dynedge_layer_sizes is None:
            dynedge_layer_sizes = [
                (
                    128,
                    256,
                ),
                (
                    336,
                    256,
                ),
                (
                    336,
                    256,
                ),
                (
                    336,
                    256,
                ),
            ]
        else:
            dynedge_layer_sizes = [
                tuple(sizes) for sizes in dynedge_layer_sizes
            ]

        assert isinstance(dynedge_layer_sizes, list)
        assert len(dynedge_layer_sizes)
        assert all(isinstance(sizes, tuple) for sizes in dynedge_layer_sizes)
        assert all(len(sizes) > 0 for sizes in dynedge_layer_sizes)
        assert all(
            all(size > 0 for size in sizes) for sizes in dynedge_layer_sizes
        )

        self._dynedge_layer_sizes = dynedge_layer_sizes
        if reduction_layers is None:
            reduction_layers = [False] * len(dynedge_layer_sizes)
        # Post-processing layer sizes

        reduction_inds = torch.squeeze(
            torch.argwhere(Tensor(reduction_layers)), dim=1
        )
        reduction_inds = (
            torch.cat([reduction_inds, Tensor([len(reduction_layers)])], dim=0)
            + 1
        )
        self._reduction_inds = reduction_inds
        post_processing_layer_sizes: List[List[int]] = []
        for ind, prev_ind in zip(
            reduction_inds.type(torch.uint8),
            torch.cat((torch.tensor([0]), reduction_inds[:-1])).type(
                torch.uint8
            ),
        ):
            post_processing_layer_sizes.append(
                [
                    int(
                        sum(
                            [
                                out
                                for _, out in self._dynedge_layer_sizes[
                                    prev_ind:ind
                                ]
                            ]
                        )
                    ),
                    int(
                        2
                        * sum(
                            [
                                out
                                for _, out in self._dynedge_layer_sizes[
                                    prev_ind:ind
                                ]
                            ]
                        )
                    ),
                    int(
                        0.5
                        * sum(
                            [
                                out
                                for _, out in self._dynedge_layer_sizes[
                                    prev_ind:ind
                                ]
                            ]
                        )
                    ),
                ]
            )
            if prev_ind == 0:
                post_processing_layer_sizes[0][0] += int(2 * nb_inputs) + 5

        assert isinstance(post_processing_layer_sizes, list)
        assert len(post_processing_layer_sizes)
        # assert all(size > 0 for size in post_processing_layer_sizes)

        self._post_processing_layer_sizes = post_processing_layer_sizes

        # Read-out layer sizes
        if readout_layer_sizes is None:
            readout_layer_sizes = [
                128,
            ]

        assert isinstance(readout_layer_sizes, list)
        assert len(readout_layer_sizes)
        assert all(size > 0 for size in readout_layer_sizes)

        self._readout_layer_sizes = readout_layer_sizes

        # Global pooling scheme(s)
        if isinstance(global_pooling_schemes, str):
            global_pooling_schemes = [global_pooling_schemes]

        if isinstance(global_pooling_schemes, list):
            for pooling_scheme in global_pooling_schemes:
                assert (
                    pooling_scheme in GLOBAL_POOLINGS
                ), f"Global pooling scheme {pooling_scheme} not supported."
        else:
            assert global_pooling_schemes is None

        self._global_pooling_schemes = global_pooling_schemes

        if add_global_variables_after_pooling:
            assert self._global_pooling_schemes, (
                "No global pooling schemes were request, so cannot add global"
                " variables after pooling."
            )
        self._add_global_variables_after_pooling = (
            add_global_variables_after_pooling
        )

        # Base class constructor
        super().__init__(nb_inputs, self._readout_layer_sizes[-1])

        # Remaining member variables()
        self._activation = torch.nn.LeakyReLU()
        self._nb_inputs = nb_inputs
        self._nb_global_variables = 5 + nb_inputs
        self._nb_neighbours = nb_neighbours
        self._features_subset = features_subset
        self._reduction_soft = torch.nn.Sequential(
            torch.nn.ReLU(), torch.nn.Softmax(dim=0)
        )
        self._construct_layers()
        if reduction_layers:
            self._reduction_layers = reduction_layers
            assert len(self._reduction_layers) == len(
                self._dynedge_layer_sizes
            )
        else:
            self._reduction_layers = [False for _ in self._dynedge_layer_sizes]

    def _construct_layers(self) -> None:
        """Construct layers (torch.nn.Modules)."""
        # Convolutional operations
        nb_input_features = self._nb_inputs
        if not self._add_global_variables_after_pooling:
            nb_input_features += self._nb_global_variables

        self._conv_layers = torch.nn.ModuleList()
        nb_latent_features = nb_input_features
        for sizes in self._dynedge_layer_sizes:
            layers = []
            layer_sizes = [nb_latent_features] + list(sizes)
            for ix, (nb_in, nb_out) in enumerate(
                zip(layer_sizes[:-1], layer_sizes[1:])
            ):
                if ix == 0:
                    nb_in *= 2
                layers.append(torch.nn.Linear(nb_in, nb_out))
                layers.append(self._activation)

            conv_layer = DynEdgeConv(
                torch.nn.Sequential(*layers),
                aggr="add",
                nb_neighbors=(
                    self._nb_neighbours
                    + torch.floor(Tensor([len(self._conv_layers) / 3]) * 3)
                ),
                features_subset=self._features_subset,
            )
            self._conv_layers.append(conv_layer)

            nb_latent_features = nb_out

        # Post-processing operations
        nb_latent_features = (
            sum(sizes[-1] for sizes in self._dynedge_layer_sizes)
            + nb_input_features
        )

        post_processing_layers = []
        # layer_sizes = [nb_latent_features] + list(
        #     self._post_processing_layer_sizes
        # )
        nb_out_tot = 0
        for post_sizes in self._post_processing_layer_sizes:
            for nb_in, nb_out in zip(post_sizes[:-1], post_sizes[1:]):
                post_processing_layers.append(torch.nn.Linear(nb_in, nb_out))
                post_processing_layers.append(self._activation)
            nb_out_tot += nb_out

        self._post_processing_list = torch.nn.ModuleList()
        for ind in range(len(self._post_processing_layer_sizes)):
            self._post_processing_list.append(
                torch.nn.Sequential(
                    *post_processing_layers[ind * 4 : (ind + 1) * 4]
                )
            )

        # self._post_processing = torch.nn.Sequential(*post_processing_layers)

        # Read-out operations
        nb_poolings = (
            len(self._global_pooling_schemes)
            if self._global_pooling_schemes
            else 1
        )
        nb_latent_features = nb_out_tot * nb_poolings
        if self._add_global_variables_after_pooling:
            nb_latent_features += self._nb_global_variables

        readout_layers = []
        layer_sizes = [nb_latent_features] + list(self._readout_layer_sizes)
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            readout_layers.append(torch.nn.Linear(nb_in, nb_out))
            readout_layers.append(self._activation)

        self._readout = torch.nn.Sequential(*readout_layers)

    def _global_pooling(self, x: Tensor, batch: LongTensor) -> Tensor:
        """Perform global pooling."""
        assert self._global_pooling_schemes
        pooled = []
        for pooling_scheme in self._global_pooling_schemes:
            pooling_fn = GLOBAL_POOLINGS[pooling_scheme]
            pooled_x = pooling_fn(x, index=batch, dim=0)
            if isinstance(pooled_x, tuple) and len(pooled_x) == 2:
                # `scatter_{min,max}`, which return also an argument, vs.
                # `scatter_{mean,sum}`
                pooled_x, _ = pooled_x
            pooled.append(pooled_x)

        return torch.cat(pooled, dim=1)

    def _calculate_global_variables(
        self,
        x: Tensor,
        edge_index: LongTensor,
        batch: LongTensor,
        *additional_attributes: Tensor,
    ) -> Tensor:
        """Calculate global variables."""
        # Calculate homophily (scalar variables)
        h_x, h_y, h_z, h_t = calculate_xyzt_homophily(x, edge_index, batch)

        # Calculate mean features
        global_means = scatter_mean(x, batch, dim=0)

        # Add global variables
        global_variables = torch.cat(
            [
                global_means,
                h_x,
                h_y,
                h_z,
                h_t,
            ]
            + [attr.unsqueeze(dim=1) for attr in additional_attributes],
            dim=1,
        )

        return global_variables

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        # Convenience variables
        x, edge_index, batch = data.x, data.edge_index, data.batch

        global_variables = self._calculate_global_variables(
            x,
            edge_index,
            batch,
            torch.log10(data.n_pulses),
        )

        # Distribute global variables out to each node
        if not self._add_global_variables_after_pooling:
            distribute = (
                batch.unsqueeze(dim=1) == torch.unique(batch).unsqueeze(dim=0)
            ).type(torch.float)

            global_variables_distributed = torch.sum(
                distribute.unsqueeze(dim=2)
                * global_variables.unsqueeze(dim=0),
                dim=1,
            )

            x = torch.cat((x, global_variables_distributed), dim=1)

        # DynEdge-convolutions
        skip_connections = [x]
        batch_list = [batch]
        for conv_layer, reduction_bool in zip(
            self._conv_layers, self._reduction_layers
        ):
            x, edge_index = conv_layer(x, edge_index, batch)
            skip_connections.append(x)
            if reduction_bool:
                x, edge_index, batch = self._reduce(x, edge_index, batch)
                batch_list.append(batch)
        # Skip-cat

        for ind, prev_ind in zip(
            self._reduction_inds.type(torch.uint8) + 1,
            torch.cat((torch.tensor([0]), self._reduction_inds[:-1] + 1)).type(
                torch.uint8
            ),
        ):
            x = torch.cat(skip_connections[prev_ind:ind], dim=1)
            if prev_ind == 0:
                res_list = [x]
            else:
                res_list.append(x)

        del skip_connections
        # Post-processing
        for post_processing, res, count in zip(
            self._post_processing_list, res_list, range(len(res_list))
        ):
            x = post_processing(res)
            res_list[count] = x

        # (Optional) Global pooling
        if self._global_pooling_schemes:
            x = torch.cat(
                [
                    self._global_pooling(x, batch=batch)
                    for x, batch in zip(res_list, batch_list)
                ],
                dim=1,
            )
            if self._add_global_variables_after_pooling:
                x = torch.cat(
                    [
                        x,
                        global_variables,
                    ],
                    dim=1,
                )
        del batch_list
        del res_list
        # Read-out
        x = self._readout(x)

        return x

    def _reduce(
        self, x: Tensor, edge_index: Tensor, batch: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Reduce number of nodes.

        reduces number of nodes by keeping the top 10% of nodes with highest
        input in high dimensional arbitrary feature

        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch indices
        Returns:
            x: Reduced node features
            edge_index: Reduced edge indices
            batch: Reduced batch indices
        """
        temp = self._reduction_soft(x[:, -1])
        # For each batch, find the top 10% of nodes
        new_ind = torch.tensor([], dtype=torch.int, device=x.device)
        for batch_id in torch.unique(batch):
            mask = batch == batch_id
            # find first entrance True in mask
            first = torch.argmax(mask * 1)
            n_nodes = self.reduce_to_val(mask, x.device)
            keep_ind = torch.topk(temp[mask], n_nodes, dim=0)[1]

            new_ind = torch.cat((new_ind, keep_ind + first), dim=0)

        x = x[new_ind]
        batch = batch[new_ind]
        # recompute edge_index
        edge_index = knn_graph(x, k=len(edge_index), batch=batch)

        return x, edge_index, batch

    @torch.jit.script
    def reduce_to_val(mask: Tensor, device: torch.device) -> int:
        """Calculate the amount of nodes to keep.

        Args:
            mask: Mask of nodes
            device: device in use
        Returns:
            n_nodes: Number of nodes to keep
        """
        n_nodes = torch.tensor(int(sum(mask)), device=device)
        min_nodes = torch.min(n_nodes, torch.tensor(100))
        n_nodes = torch.max(torch.stack([min_nodes, n_nodes * 0.05])).type(
            torch.int
        )
        return n_nodes
