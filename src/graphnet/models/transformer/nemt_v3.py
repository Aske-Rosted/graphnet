import torch
import torch.nn as nn
from typing import Optional, Set, Tuple

from graphnet.models.components.embedding import (
    SpacetimeEncoder,
    SinusoidalPosEmb,
)
from graphnet.models.components.layers import Block, Block_rel
from graphnet.models.gnn.gnn import GNN

from graphnet.models.utils import array_to_sequence

from torch_geometric.data import Data
from torch import Tensor


class NeutrinoEventMultitaskTransformer_v3(GNN):
    """NeutrinoEventMultiTaskTransformer model."""

    def __init__(
        self,
        n_attention_blocks: int = 2,
        n_rel: int = 1,
        inject_cls_after: Optional[int] = 0,
        hidden_dim: int = 128,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        n_features: int = 36,
        pre_emb_scale: float = 1024.0,
        n_tasks: int = 1,
        shared_tokens: int = 0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        token_multiplier: int = 1,
        out_dim: Optional[int] = None,
        cross_attention: list = [],
    ):
        """Construct `NeutrinoEventMultiTaskTransformer`.

        Args:
            hidden_dim: The latent feature dimension.
            seq_length: The number of pulses in a neutrino event.
            num_layers: The depth of the transformer.
            num_heads: The number of the attention heads.
            mlp_dim: The mlp dimension of FourierEncoder and Transformer.
            max_rel_pos: Maximum relative position for relative position bias.
            num_register_tokens: The number of register tokens.
            scaled_emb: Whether to scale the sinusoidal positional embeddings.
            n_features: The number of features in the input data.
        """
        self.inject_cls_after = inject_cls_after
        tot_out = out_dim if out_dim is not None else hidden_dim
        tot_out = tot_out * n_tasks

        super().__init__(n_features, tot_out)

        self.embedding = SinusoidalPosEmb(
            dim=hidden_dim,
        )
        self.emb_mlp = nn.Sequential(
            nn.Linear(n_features * (hidden_dim), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        ESA = nn.ModuleList()
        for i in range(n_attention_blocks):
            if i < n_rel:
                ESA.append(
                    Block_rel(
                        hidden_dim,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                    )
                )
            else:
                ESA.append(
                    Block(
                        hidden_dim,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                    )
                )
        self.ESA = ESA

        xTAMS = nn.ModuleList()
        gates = nn.ParameterList()
        for i in range(len(cross_attention)):
            xTAMS.append(
                Block(
                    hidden_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                )
            )
            gates.append(
                nn.Parameter(
                    torch.zeros(
                        1, (n_tasks + shared_tokens) * token_multiplier, 1
                    )
                )
            )
        self.xTAMS = xTAMS
        self.gates = gates

        self.task_tokens = nn.Parameter(
            torch.empty(1, n_tasks * token_multiplier, hidden_dim),
            requires_grad=True,
        )
        self.n_tasks = n_tasks

        if shared_tokens > 0:
            self.shared_tokens = nn.Parameter(
                torch.empty(1, shared_tokens * token_multiplier, hidden_dim),
                requires_grad=True,
            )
            nn.init.xavier_normal_(self.shared_tokens, gain=1.0)
        else:
            self.shared_tokens = None

        nn.init.xavier_normal_(self.task_tokens, gain=1.0)

        assert (
            hidden_dim % num_heads == 0
        ), "hidden_dim must be divisible by num_heads"

        self.n_rel = n_rel
        if self.n_rel > 0:
            self.rel_pos = SpacetimeEncoder(hidden_dim // num_heads)

        assert (
            self.n_rel < n_attention_blocks
        ), "n_rel must be less than n_attention_blocks"
        assert (
            inject_cls_after < n_attention_blocks
        ), "inject_cls_after must be less than n_attention_blocks"
        assert (
            inject_cls_after >= self.n_rel
        ), "inject_cls_after must be greater than n_rel"

        assert all(
            c < n_attention_blocks for c in cross_attention
        ), "cross_attention must be less than n_attention_blocks"
        assert all(
            c >= inject_cls_after for c in cross_attention
        ), "cross_attention must be greater than or equal to inject_cls_after"

        self.pre_emb_scale = pre_emb_scale

        if out_dim is not None:
            self.task_out = nn.Sequential(
                nn.Linear(hidden_dim * token_multiplier, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )

        self.dropout = dropout

        self.cross_attention = cross_attention

        self.token_multiplier = token_multiplier

    # is the following necessary
    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """task_tokens should not be subject to weight decay during
        training."""
        return {"task_tokens"}

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass to input data."""
        x, mask, seq_len = array_to_sequence(data.x, data.batch)
        batch_size = mask.shape[0]

        if self.n_rel > 0:
            rel_pos_bias = self.rel_pos(x)

        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf

        cls_token = self.task_tokens.expand(batch_size, -1, -1)
        shrd_token = (
            self.shared_tokens.expand(batch_size, -1, -1)
            if self.shared_tokens is not None
            else None
        )

        n_cls = cls_token.shape[1]
        n_shrd = shrd_token.shape[1] if shrd_token is not None else 0

        x = self.embedding(self.pre_emb_scale * x).flatten(-2)
        x = self.emb_mlp(x)

        for i, blk in enumerate(self.ESA):
            if i == self.inject_cls_after:
                # Inject cls tokens
                x = torch.cat([cls_token, x], 1)
                # Inject shared tokens if they exist
                if shrd_token is not None:
                    x = torch.cat([shrd_token, x], 1)

                attn_mask = create_attn_mask(
                    mask, batch_size, add_tokens=n_cls + n_shrd
                )

            if i < self.n_rel:
                # relative position bias is only used in the first n_rel blocks
                x = blk(x, attn_mask, rel_pos_bias)
            else:
                # no relative position bias in the rest of the blocks
                x = blk(x, None, attn_mask)

            if i in self.cross_attention:
                idx = self.cross_attention.index(i)
                # pull out the tokens from the output attention
                tokens = x[:, : n_cls + n_shrd]
                x = x[:, n_cls + n_shrd :]

                # re-inject the cls tokens after cross attention with gating
                gate = torch.sigmoid(self.gates[idx])
                tokens = tokens * gate * (self.xTAMS[idx](tokens) - tokens)
                x = torch.cat([tokens, x], 1)

        # extract the task tokens from the output of the last ESA block
        x = x[:, : cls_token.shape[1]]

        if self.token_multiplier > 1:
            # reshape the output to (batch_size, n_tasks, token_multiplier*hidden_dim)
            x = x.view(
                batch_size, self.n_tasks, self.token_multiplier * x.shape[-1]
            )

        if self.task_out is not None:
            # ensure correct shape (neccesary if token_multiplier > 1)
            x = self.task_out(x)
        return x.flatten(1, 2)


def create_attn_mask(
    mask: Tensor, batch_size: Tensor, add_tokens: int = 0
) -> Tensor:
    """Create attention mask for transformer."""
    if add_tokens == 0:
        attn_mask = torch.zeros(mask, device=mask.device)
        attn_mask[~mask] = -torch.inf
    else:
        mask = torch.cat(
            [
                torch.ones(
                    batch_size,
                    add_tokens,
                    dtype=mask.dtype,
                    device=mask.device,
                ),
                mask,
            ],
            1,
        )
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
    return attn_mask
