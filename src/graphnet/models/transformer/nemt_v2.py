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


class NeutrinoEventMultitaskTransformer_v2(GNN):
    """NeutrinoEventMultiTaskTransformer model."""

    def __init__(
        self,
        n_attention_blocks: int = 2,
        n_rel: int = 1,
        inject_cls_after: Optional[int] = 0,
        support_out_after: Optional[int] = None,
        hidden_dim: int = 128,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        n_features: int = 36,
        pre_emb_scale: float = 1024.0,
        n_final_tasks: int = 1,
        dropout: float = 0.0,
        token_multiplier: int = 1,
        shared_token: bool = False,
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
            n_final_tasks: The number of final tasks.
            n_support_tasks: The number of support tasks (if any).
            support_out_after: The layer after which the support task tokens are extracted (if any).
            inject_cls_after: The layer after which the cls tokens are injected.
            token_multiplier: The factor by which the number of tokens is multiplied when injected (e.g
                for multi-task learning or to increase model capacity).
            shared_token: Whether to use a shared token between the support and final tasks.
            support_out_dim: The output dimension for the support tasks (if any).
            final_out_dim: The output dimension for the final tasks (if any).
            cross_attention: Whether to use a cross attention layer between the support and final task tokens after the last ESA.
        """
        self.inject_cls_after = inject_cls_after

        if support_out_dim is None and final_out_dim is None:
            super().__init__(n_features, hidden_dim)
        else:
            out_dim = 0
            if support_out_dim is not None:
                out_dim += support_out_dim * (
                    n_support_tasks if n_support_tasks is not None else 0
                )
            else:
                out_dim += (
                    hidden_dim
                    * (n_support_tasks if n_support_tasks is not None else 0)
                    * token_multiplier
                    * (1 + (1 if shared_token else 0))
                )
            if final_out_dim is not None:
                out_dim += final_out_dim * n_final_tasks
            else:
                out_dim += (
                    hidden_dim
                    * n_final_tasks
                    * token_multiplier
                    * (1 + (1 if shared_token else 0))
                )
            super().__init__(n_features, out_dim)

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

        self.support_out_after = support_out_after
        self.token_multiplier = token_multiplier

        self.final_task_token = nn.Parameter(
            torch.empty(1, n_final_tasks * self.token_multiplier, hidden_dim),
            requires_grad=True,
        )

        if shared_token:
            self.shared_token = nn.Parameter(
                torch.empty(1, 1 * self.token_multiplier, hidden_dim),
                requires_grad=True,
            )
            nn.init.xavier_normal_(self.shared_token, gain=1.0)
        else:
            self.shared_token = None

        nn.init.xavier_normal_(self.final_task_token, gain=1.0)

        if support_out_after is not None:
            assert (
                n_support_tasks is not None
            ), "n_support_tasks must be provided if support_out_after is set"
            assert (
                support_out_after <= n_attention_blocks
            ), "support_out_after must be less than n_attention_blocks"
            assert (
                support_out_after > inject_cls_after
            ), "support_out_after must be greater than inject_cls_after"
            self.support_task_token = nn.Parameter(
                torch.empty(
                    1, n_support_tasks * self.token_multiplier, hidden_dim
                ),
                requires_grad=True,
            )
            nn.init.xavier_normal_(self.support_task_token, gain=1.0)
        else:
            support_out_after = None
            self.support_task_token = None

        self.ESA = ESA
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

        self.pre_emb_scale = pre_emb_scale

        if support_out_dim is not None:
            self.support_task_out = nn.Sequential(
                nn.Linear(
                    hidden_dim
                    * token_multiplier
                    * (1 + (1 if shared_token else 0)),
                    support_out_dim,
                ),
                nn.LayerNorm(support_out_dim),
                nn.GELU(),
                nn.Linear(support_out_dim, support_out_dim),
            )
        if final_out_dim is not None:
            self.final_task_out = nn.Sequential(
                nn.Linear(
                    hidden_dim
                    * token_multiplier
                    * (1 + (1 if shared_token else 0)),
                    final_out_dim,
                ),
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
        """cls_tokens should not be subject to weight decay during training."""
        return {"support_task_token", "final_task_token", "shared_token"}

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass to input data."""
        x, mask, seq_len = array_to_sequence(data.x, data.batch)
        batch_size = mask.shape[0]

        if self.n_rel > 0:
            rel_pos_bias = self.rel_pos(x)

        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf

        cls_token = self.final_task_token.expand(batch_size, -1, -1)
        if self.support_task_token is not None:
            support_cls_token = self.support_task_token.expand(
                batch_size, -1, -1
            )

        if self.shared_token is not None:
            shared_cls_token = self.shared_token.expand(batch_size, -1, -1)

        x = self.embedding(self.pre_emb_scale * x).flatten(-2)
        x = self.emb_mlp(x)

        for i, blk in enumerate(self.ESA):
            if i == self.inject_cls_after:

                # Inject cls tokens
                x = torch.cat([cls_token, x], 1)
                # Inject support task tokens if they exist

                # inject shared token if it exists
                if self.shared_token is not None:
                    x = torch.cat([shared_cls_token, x], 1)

                if self.support_task_token is not None:
                    x = torch.cat([support_cls_token, x], 1)

                add_tokens_count = self.token_multiplier * (
                    self.n_final_tasks
                    + (1 if self.shared_token is not None else 0)
                    + (
                        self.n_support_tasks
                        if self.support_task_token is not None
                        else 0
                    )
                )
                attn_mask = create_attn_mask(
                    mask, batch_size, add_tokens=add_tokens_count
                )

            if i < self.n_rel:
                # relative position bias is only used in the first n_rel blocks
                x = blk(
                    x, key_padding_mask=attn_mask, rel_pos_bias=rel_pos_bias
                )
            else:
                # no relative position bias in the rest of the blocks
                x = blk(x, None, key_padding_mask=attn_mask)

            if (self.support_task_token is not None) and (
                (i + 1) == self.support_out_after
            ):
                # If support task token is added, extract it after the n_support_tasks
                sup_out = x[:, : self.n_support_tasks * self.token_multiplier]
                x = x[:, self.n_support_tasks * self.token_multiplier :]
                # recalculate attn_mask without support tokens
                add_tokens_count = self.token_multiplier * (
                    self.n_final_tasks
                    + (1 if self.shared_token is not None else 0)
                )
                attn_mask = create_attn_mask(
                    mask, batch_size, add_tokens=add_tokens_count
                )

        # If final layer extract the shared token
        if self.shared_token is not None:
            shared_out = x[:, : self.token_multiplier]
            x = x[:, self.token_multiplier :]
            shared_out = shared_out.view(batch_size, -1)

        # extract final task tokens
        x = x[:, : self.n_final_tasks * self.token_multiplier]

        if self.final_out_dim is not None:
            # ensure correct shape (neccesary if token_multiplier > 1)
            x = x.view(batch_size, self.n_final_tasks, -1)

            # Final task output head reduce dimension by token_multiplier
            if self.shared_token is not None:
                x = torch.cat(
                    [
                        x,
                        shared_out.unsqueeze(1).expand(
                            -1, self.n_final_tasks, -1
                        ),
                    ],
                    2,
                )

            x = self.final_task_out(x)
            x = x.flatten(1)

        if self.support_task_token is not None:
            # Support task output head reduce dimension by token_multiplier
            sup_out = sup_out.view(batch_size, self.n_support_tasks, -1)
            if self.shared_token is not None:
                sup_out = torch.cat(
                    [
                        sup_out,
                        shared_out.unsqueeze(1).expand(
                            -1, self.n_support_tasks, -1
                        ),
                    ],
                    2,
                )
            sup_out = self.support_task_out(sup_out)
            sup_out = sup_out.flatten(1)
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
