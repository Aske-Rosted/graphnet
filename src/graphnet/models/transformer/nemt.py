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


class NeutrinoEventMultitaskTransformer(GNN):
    """NeutrinoEventMultiTaskTransformer model."""

    def __init__(
        self,
        n_attention_blocks: int = 2,
        n_rel: int = 1,
        support_task_after_n: Optional[int] = None,
        hidden_dim: int = 128,
        seq_length: int = 196,
        num_heads: int = 4,
        mlp_dim: int = 128,
        n_features: int = 36,
        pre_emb_scale: float = 1024.0,
        n_final_tasks: int = 1,
        dropout: float = 0.0,
        token_multiplier: int = 1,
        support_out_dim: Optional[int] = None,
        final_out_dim: Optional[int] = None,
        n_support_tasks: Optional[int] = None,
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

        if support_out_dim is None and final_out_dim is None:
            super().__init__(seq_length, hidden_dim)
        else:
            out_dim = 0
            if support_out_dim is not None:
                out_dim += support_out_dim
            if final_out_dim is not None:
                out_dim += final_out_dim
            super().__init__(seq_length, out_dim)

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
                    Block_rel(hidden_dim, num_heads, mlp_dim, dropout=dropout)
                )
                ESA.append(
                    Block_rel(hidden_dim, num_heads, mlp_dim, dropout=dropout)
                )
            else:
                ESA.append(
                    Block(hidden_dim, num_heads, mlp_dim, dropout=dropout)
                )
                ESA.append(
                    Block(hidden_dim, num_heads, mlp_dim, dropout=dropout)
                )

            if i == support_task_after_n:
                ESA.append(
                    Block(hidden_dim, num_heads, mlp_dim, dropout=dropout)
                )

        self.support_task_after_n = support_task_after_n
        self.token_multiplier = token_multiplier

        self.final_task_token = nn.Parameter(
            torch.empty(1, n_final_tasks * self.token_multiplier, hidden_dim),
            requires_grad=True,
        )
        nn.init.xavier_normal_(self.final_task_token, gain=1.0)

        if support_task_after_n is not None:
            assert (
                n_support_tasks is not None
            ), "n_support_tasks must be provided if support_task_after_n is set"
            self.support_task_token = nn.Parameter(
                torch.empty(
                    1, n_support_tasks * self.token_multiplier, hidden_dim
                ),
                requires_grad=True,
            )
            nn.init.xavier_normal_(self.support_task_token, gain=1.0)
        else:
            support_task_after_n = None
            self.support_task_token = None

        ESA.append(Block(hidden_dim, num_heads, mlp_dim))

        self.ESA = ESA
        assert (
            hidden_dim % num_heads == 0
        ), "hidden_dim must be divisible by num_heads"

        self.rel_pos = SpacetimeEncoder(hidden_dim // num_heads)
        assert (
            n_rel < n_attention_blocks
        ), "n_rel must be less than n_attention_blocks"
        self.n_rel = n_rel
        self.pre_emb_scale = pre_emb_scale

        if support_out_dim is not None:
            self.support_task_out = nn.Sequential(
                nn.Linear(hidden_dim, support_out_dim),
                nn.LayerNorm(support_out_dim),
                nn.GELU(),
                nn.Linear(support_out_dim, support_out_dim),
            )
        if final_out_dim is not None:
            self.final_task_out = nn.Sequential(
                nn.Linear(hidden_dim, final_out_dim),
                nn.LayerNorm(final_out_dim),
                nn.GELU(),
                nn.Linear(final_out_dim, final_out_dim),
            )
        self.support_out_dim = support_out_dim
        self.final_out_dim = final_out_dim
        self.n_support_tasks = n_support_tasks
        self.n_final_tasks = n_final_tasks
        self.dropout = dropout

    # is the following necessary
    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """cls_tocken should not be subject to weight decay during training."""
        return {"support_task_token", "final_task_token"}

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass to input data."""
        x, mask, seq_len = array_to_sequence(data.x, data.batch)
        batch_size = mask.shape[0]
        rel_pos_bias = self.rel_pos(x)

        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf

        cls_token = self.final_task_token.expand(batch_size, -1, -1)
        if self.support_task_token is not None:
            support_cls_token = self.support_task_token.expand(
                batch_size, -1, -1
            )

        x = self.embedding(self.pre_emb_scale * x).flatten(-2)
        x = self.emb_mlp(x)

        for i, blk in enumerate(self.ESA):
            if (self.support_task_token is not None) and (
                i == self.support_task_after_n * 2
            ):
                # If support task token is added, add it after the n_support_tasks
                tmp_attn_mask = create_attn_mask(
                    mask,
                    batch_size,
                    add_tokens=self.n_support_tasks * self.token_multiplier,
                )
                x_sup = torch.cat([support_cls_token, x], 1)
                if i < self.n_rel * 2:
                    x_sup = blk(x_sup, tmp_attn_mask, rel_pos_bias)
                else:
                    x_sup = blk(x_sup, None, tmp_attn_mask)
                sup_out = x_sup[
                    :, : self.n_support_tasks * self.token_multiplier
                ]
                x = x_sup[:, self.n_support_tasks * self.token_multiplier :]
                del tmp_attn_mask
            elif i == len(self.ESA) - 1:
                # If final layer add the final task token
                x = torch.cat([cls_token, x], 1)
                attn_mask = create_attn_mask(
                    mask,
                    batch_size,
                    add_tokens=self.n_final_tasks * self.token_multiplier,
                )
                if i < self.n_rel * 2:
                    x = blk(x, attn_mask, rel_pos_bias)
                else:
                    x = blk(x, None, attn_mask)
                x = x[:, : self.n_final_tasks * self.token_multiplier]
            else:
                if i < self.n_rel * 2:
                    # relative position bias is only used in the first n_rel blocks
                    x = blk(x, attn_mask, rel_pos_bias)
                else:
                    # no relative position bias in the rest of the blocks
                    x = blk(x, None, attn_mask)

        if self.final_out_dim is not None:
            x = self.final_task_out(x)
            x = x.flatten(1)

        if self.support_task_token is not None:
            sup_out = self.support_task_out(sup_out)
            sup_out = sup_out.flatten(1)

        if self.support_task_token is not None:
            x = torch.cat([sup_out, x], 1)
        return x


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
