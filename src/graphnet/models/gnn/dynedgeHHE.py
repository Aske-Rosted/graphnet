"""Implementation of the DynEdge GNN model architecture."""
from typing import List, Optional, Sequence, Tuple, Union, Callable

import torch
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from torch_geometric.nn import dense_diff_pool
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse
from torch_geometric.nn.pool import knn_graph

from graphnet.models.components.layers import DynEdgeConv
from graphnet.utilities.config import save_model_config
from graphnet.models.gnn.gnn import GNN
from graphnet.models import Model
from graphnet.models.utils import calculate_xyzt_homophily
from graphnet.models.components.pool import group_by


from math import ceil

GLOBAL_POOLINGS = {
    "min": scatter_min,
    "max": scatter_max,
    "sum": scatter_sum,
    "mean": scatter_mean,
}


class Diff(GNN):
    """Network used in Hierarchical GNN type architectures.

    Implementation of GNN with a DynEdge layers and a optional pooling layer,
    which can be used for embedding or pooling in a Hierarchical GNN type
    architecture.
    """

    def __init__(
        self,
        dynedge_layer_sizes: List[tuple],
        nb_input: int,
        nb_neighbour: int = 2,
        output_activation: Optional[Callable] = None,
        post_process_size: Optional[int] = None,
        features_subset: Optional[Sequence[int]] = None,
    ):
        """Construct the embedding/pooling GNN.

        Args:
            dynedge_layer_sizes: list of tuples containing the sizes of the layers in the dynedge architecture
            nb_input: number of input features
            nb_neighbour: number of neighbours to use in the dynedge layer
            output_activation: activation function to use in the optional pooling layer
            post_process_size: size of the optional post processing layer
            features_subset: list of features to use
        """
        # Check arguments
        assert (post_process_size is None) == (
            output_activation is None
        ), "Post process size and output activation must be both None or both not None"

        # Base class constructor
        if post_process_size:
            super().__init__(nb_input, post_process_size)
        else:
            super().__init__(
                nb_input, sum(layer[-1] for layer in dynedge_layer_sizes)
            )

        self._dynedge_layer_size = dynedge_layer_sizes
        self._nb_input = nb_input
        self._nb_neighbour = nb_neighbour
        self._post_process_size = post_process_size
        self._features_subset = features_subset
        self._output_activation = output_activation

        self._construct_dynedge_layers()

        if output_activation:
            self._output_activation = output_activation
        else:
            self._output_activation = None

    def forward(
        self, data: Data
    ) -> Tensor:  # x: Tensor, edge_index: Tensor, batch: Tensor
        """Apply learnable forward pass to the sub GNN."""
        # Apply convolutional layers
        x, edge_index, batch = data.x, data.edge_index, data.batch

        skip_connections = []
        for conv_layer in self._conv_layers:
            x, edge_index = conv_layer(x, edge_index, batch)
            skip_connections.append(x)

        x = torch.cat(skip_connections, dim=1)

        if self._output_activation:
            x = self._post_process(x)

        return x

    def _construct_dynedge_layers(self) -> None:
        """Construct layers (torch.nn.Modules)."""
        # Convolutional operations

        self._conv_layers = torch.nn.ModuleList()
        nb_latent_features = self._nb_inputs
        for sizes in self._dynedge_layer_size:
            layers = []
            layer_sizes = [nb_latent_features] + list(sizes)
            for ix, (nb_in, nb_out) in enumerate(
                zip(layer_sizes[:-1], layer_sizes[1:])
            ):
                if ix == 0:
                    nb_in *= 2
                layers.append(torch.nn.Linear(nb_in, nb_out))
                layers.append(torch.nn.LeakyReLU())

            conv_layer = DynEdgeConv(
                torch.nn.Sequential(*layers),
                aggr="add",
                nb_neighbors=self._nb_neighbour,
                features_subset=self._features_subset,
            )
            self._conv_layers.append(conv_layer)

            nb_latent_features = nb_out

        # Post-processing
        out_dim = sum(layer[-1] for layer in self._dynedge_layer_size)

        if self._output_activation:
            post_process_layers = [
                torch.nn.Linear(out_dim, self._post_process_size)
            ]
            post_process_layers.append(self._output_activation())

            self._post_process = torch.nn.Sequential(*post_process_layers)


class HighENet(GNN):
    """Implementation of the high-energy GNN model architecture.

    based on the DynEdge and the diffpooling architectures
    """

    @save_model_config
    def __init__(
        self,
        nb_inputs: int,
        nb_neighbours: List[int],
        first_layer_size: int = 3000,
        RNN_out_size: int = 512,
        red_factor: float = 0.2,
        dynedge_layer_sizes: Optional[List[List[Tuple[int, ...]]]] = None,
        last_layer_size: Optional[List[Tuple[int, ...]]] = None,
        pool_activation: str = "Tanh",
        post_process_size: Optional[int] = None,
        post_processing_layer_sizes: Optional[List[int]] = None,
        # readout_layer_sizes: Optional[List[int]] = None,
        features_subset: Optional[List[int]] = None,
    ) -> None:
        """Construct `HighENet`.

        Args:
            nb_inputs: Number of features in the input data.
            nb_neighbours: length of the list determines the number of poolings and the values of the number of neighbours for each excluding first layer
            red_factor: reduction factor for the number of nodes after each pooling must be between 0 and 1
            first_layer_size: number of nodes in the first layer
            RNN_out_size: number of nodes in the RNN output layer
            dynedge_layer_sizes: list of tuples containing the sizes of the layers in the dynedge architecture
            last_layer_size: list of tuples containing the sizes of the layers in the last layer of the DNN architecture
            pool_activation: activation function to use in the pooling layer
            post_process_size: size of the post processing layer
            post_processing_layer_sizes: list of sizes of the layers in the post processing layer
            features_subset: list of features to use
        """
        # assertion checks
        assert 0 < red_factor < 1, "Reduction factor must be between 0 and 1"
        assert all(
            isinstance(neigh, int) for neigh in nb_neighbours
        ), "Number of neighbours must be an integer"

        # Member variables
        self._nb_inputs = nb_inputs
        self._first_layer_neighbours = nb_neighbours.pop(0)
        self._nb_neighbours = nb_neighbours
        self._red_factor = red_factor
        self._post_process_size = post_process_size
        self._first_layer_size = first_layer_size
        self._features_subset = features_subset
        self._RNN_out_size = RNN_out_size

        self._pool_activation = getattr(torch.nn, pool_activation)
        if dynedge_layer_sizes is None:
            standard_layer_sizes: List[Tuple[int, ...]] = [(16, 32), (64, 32)]
            self._dynedge_layer_sizes = []
            for _ in range(len(self._nb_neighbours)):
                self._dynedge_layer_sizes.append(standard_layer_sizes)
        else:
            self._dynedge_layer_sizes = []
            for layers in dynedge_layer_sizes:
                temp_layer = [tuple(layer) for layer in layers]
                self._dynedge_layer_sizes.append(temp_layer)
        assert len(self._dynedge_layer_sizes) == len(
            self._nb_neighbours
        ), "Number of pooling layers must be equal to the number of pooling layers sizes"

        if last_layer_size is None:
            last_layer_size = [(32, 64), (128, 256), (512, 1024)]
        self._last_layer_size = last_layer_size

        # assert all(isinstance(sizes,tuple) for sizes in dynedge_layer_sizes), "Layer sizes must be a tuple"

        # out_dim = sum(layer[-1] for layer in dynedge_layer_sizes)

        if post_processing_layer_sizes is None:
            post_processing_layer_sizes = [256, 128]

        self._post_processing_layer_sizes = [
            sum(layer[-1] for layer in self._last_layer_size)
        ] + post_processing_layer_sizes
        # self._post_processing_layer_sizes = [out_dim * (len(self._nb_neighbours)+1) + nb_inputs] + post_processing_layer_sizes
        # self._dynedge_layer_sizes = dynedge_layer_sizes

        # Base class constructor
        super().__init__(nb_inputs, 1)

        self.RNN_net = NodeTimeRNN(
            self._nb_inputs,
            self._RNN_out_size,
            num_layers=1,
            nb_neighbours=self._first_layer_neighbours,
        )

        self.GNN_pool = torch.nn.ModuleList()
        self.GNN_embed = torch.nn.ModuleList()

        self.GNN_pool.append(
            Diff(
                dynedge_layer_sizes=self._dynedge_layer_sizes[0],
                nb_input=self._RNN_out_size + 5,
                output_activation=self._pool_activation,
                post_process_size=first_layer_size,
                nb_neighbour=self._first_layer_neighbours,
            )
        )
        self.GNN_embed.append(
            Diff(
                self._dynedge_layer_sizes[0],
                self._RNN_out_size + 5,
                nb_neighbour=self._first_layer_neighbours,
            )
        )
        out_dim = sum(layer[-1] for layer in self._dynedge_layer_sizes[0])
        for ix, (n_neigh, layer_sizes) in enumerate(
            zip(self._nb_neighbours[:-1], self._dynedge_layer_sizes[1:])
        ):

            self.GNN_pool.append(
                Diff(
                    layer_sizes,
                    nb_input=out_dim,
                    nb_neighbour=n_neigh,
                    output_activation=self._pool_activation,
                    post_process_size=ceil(
                        first_layer_size * red_factor ** (ix + 1)
                    ),
                )
            )
            self.GNN_embed.append(Diff(layer_sizes, out_dim, n_neigh))
            out_dim = sum(layer[-1] for layer in layer_sizes)

        self.GNN_pool.append(
            Diff(
                last_layer_size,
                nb_input=out_dim,
                nb_neighbour=self._nb_neighbours[-1],
                output_activation=self._pool_activation,
                post_process_size=1,
            )
        )
        self.GNN_embed.append(
            Diff(
                last_layer_size,
                nb_input=out_dim,
                nb_neighbour=self._nb_neighbours[-1],
            )
        )

        post_processing_layers = []
        for nb_in, nb_out in zip(
            self._post_processing_layer_sizes[:-1],
            self._post_processing_layer_sizes[1:],
        ):
            post_processing_layers.append(torch.nn.Linear(nb_in, nb_out))
            post_processing_layers.append(torch.nn.LeakyReLU())

        self._post_processing = torch.nn.Sequential(*post_processing_layers)

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass to the GNN."""
        # x, edge_index, batch = data.x, data.edge_index, data.batch

        data = self.RNN_net(data)

        # x, edge_index, batch = data.x, data.edge_index, data.batch

        link_loss = []
        ent_loss = []
        # skip_connections = [x]

        for pool, embed, n_b in zip(
            self.GNN_pool,
            self.GNN_embed,
            [self._first_layer_neighbours] + self._nb_neighbours,
        ):

            s = to_dense_batch(
                pool(data.x, data.edge_index, data.batch), data.batch
            )[0]
            data.x = embed(data.x, data.edge_index, data.batch)
            data.x, mask = to_dense_batch(data.x, data.batch)
            data.edge_index = to_dense_adj(data.edge_index, data.batch)
            data.x, data.edge_index, l_tmp, e_tmp = dense_diff_pool(
                data.x, data.edge_index, s, mask=mask
            )
            data.batch = (
                (
                    torch.ones(
                        (data.x.shape[0], data.x.shape[1]), device=self.device
                    ).T
                    * (torch.arange(data.x.shape[0], device=self.device) + 1)
                ).T.flatten()
                - 1
            ).type(torch.int64)
            data.x = data.x.reshape(
                data.x.shape[0] * data.x.shape[1], data.x.shape[2]
            )

            # really not happy with this implementation... TODO: find a better way to recompute the edge index.
            data.edge_index = knn_graph(
                x=data.x,
                k=n_b,
                batch=data.batch,
            ).to(self.device)

            link_loss.append(l_tmp)
            ent_loss.append(e_tmp)

        data.x = self._post_processing(data.x)

        return data.x  # , sum(link_loss), sum(ent_loss)


class NodeTimeRNN(GNN):
    """Implementation of the RNN model architecture.

    The model takes as input the typical DOM data format and transforms it into
    a time series of DOM activations pr. DOM. before applying a RNN layer and
    outputting the an RNN output for each DOM.
    """

    def __init__(
        self,
        nb_inputs: int,
        nb_outputs: int,
        num_layers: int,
        nb_neighbours: int,
    ) -> None:
        """Construct `NodeTimeRNN`.

        Args:
            nb_inputs: Number of features in the input data.
            nb_outputs: Number of features in the output data.
            num_layers: Number of layers in the RNN.
            nb_neighbours: Number of neighbours to use when reconstructing the graph representation.
        """
        super().__init__(nb_inputs, nb_outputs)

        self._nb_neighbours = nb_neighbours
        self._rnn = torch.nn.RNN(
            num_layers=num_layers,
            input_size=nb_inputs,
            hidden_size=nb_outputs,
            batch_first=True,
        )

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass to the GNN."""
        x, batch = data.x, data.batch

        # DOM_index = group_by(data.x[:,:3],batch)
        x = NodeTimeRNN.dom_id(x, batch, self.device)

        # get location of first occurence of each unique value

        # dom_data_list, xyzt_list, new_batch= NodeTimeRNN.dom_data_to_list(x, batch, unique)

        dom_activation_sort = x[:, -1].sort()[-1]
        x, sort_batch = x[dom_activation_sort], batch[dom_activation_sort]
        bin_count = torch.bincount(x[:, -1].type(torch.int64)).cumsum(0)
        sort_batch = sort_batch[bin_count - 1]
        x = x[:, :-1]
        x = torch.tensor_split(x, bin_count.cpu()[:-1])
        lengths_index = (
            torch.as_tensor([v.size(0) for v in x]).sort()[-1].flip(0)
        )
        sort_batch = sort_batch[lengths_index]

        x = sorted(x, key=len, reverse=True)
        xyztt = torch.stack(
            [
                torch.cat(
                    [
                        v[0, :3],
                        torch.as_tensor(
                            [v[:, 4].mean(), v[:, 4].max()], device=self.device
                        ),
                    ]
                )
                for v in x
            ]
        )
        x = torch.nn.utils.rnn.pack_sequence(x, enforce_sorted=True)

        x = self._rnn(x)[-1][0]

        x = torch.hstack([xyztt, x])

        batch, sort_index = sort_batch.sort()
        data.x = x[sort_index]
        data.batch = batch
        edge_index = knn_graph_ignore(
            x=data.x,
            k=self._nb_neighbours,
            batch=data.batch,
        ).to(self.device)
        data.edge_index = edge_index
        return data

    @torch.jit.script
    def dom_data_to_list(
        x: Tensor, batch: Tensor, unique: Tensor
    ) -> Tuple[List[Tensor], List[Tensor], Tensor]:
        """Convert DOM data to list of DOM activations.

        Args:
            x: DOM data
            batch: batch index
            unique: unique DOM index
        """
        new_batch = []
        dom_data_list = []
        xyzt_list = []

        for i in unique:
            mask = torch.where(x[:, -1] == i)
            temp_index = mask[0]
            new_batch.append(batch[temp_index[0]])
            dom_data = x[:, :-1][temp_index]
            xyzt = torch.cat(
                (dom_data[0, :3], dom_data[:, 3].mean().unsqueeze(0))
            )

            dom_data_list.append(dom_data)
            xyzt_list.append(xyzt)

        return dom_data_list, xyzt_list, new_batch

    def dom_id(x: Tensor, batch: Tensor, device: torch.device) -> Tensor:
        """Create unique DOM index.

        Args:
            x: DOM data
            batch: batch index
            device: device to use
        """
        inverse_matrix = torch.zeros(
            (x.shape[0], 3), device=device, dtype=torch.int64
        )
        for i in range(3):
            _, inverse = torch.unique(x[:, i], return_inverse=True)
            inverse_matrix[:, i] = inverse

        inverse_matrix = torch.hstack([batch.unsqueeze(1), inverse_matrix])
        for i in range(3):
            inverse_matrix[:, 1] = inverse_matrix[:, 0] + (
                (torch.max(inverse_matrix[:, 0]) + 1)
                * (inverse_matrix[:, 1] + 1)
            )

            _, inverse_matrix[:, 1] = torch.unique(
                inverse_matrix[:, 1], return_inverse=True
            )

            inverse_matrix = inverse_matrix[:, -(3 - i) :]

        inverse_matrix = inverse_matrix.flatten()
        x = torch.hstack([x, inverse_matrix.unsqueeze(1)])
        return x


@torch.jit.ignore
def knn_graph_ignore(
    x: Tensor, k: int, batch: Optional[Tensor] = None
) -> Tensor:
    """Create a kNN graph based on the input data.

    Args:
        x: Input data.
        k: Number of neighbours.
        batch: Batch index.
    """
    return knn_graph(x=x, k=k, batch=batch)
