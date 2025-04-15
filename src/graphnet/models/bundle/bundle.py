"""
Transformer Model for Bundle Rejection
"""

from graphnet.models.gnn.gnn import GNN
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
import torch

class BundleTransformer(GNN):

    def __init__(
        self,
        nb_inputs: int,
        nb_neighbours: int = 8,
        nb_features: int = 5,
    ):

        pass
        
    def forward():

        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        src = torch.rand(10, 32, 512)
        out = encoder_layer(src)
        pass