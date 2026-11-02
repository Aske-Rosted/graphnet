"""Neutrino Event Multitask Transformer (NEMT).

A transformer backbone producing one output token group per task, for
reconstructing several quantities of an event with a single model.
"""

from typing import List, Optional, Set

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from graphnet.models.components.attention_blocks import Block, Block_rel
from graphnet.models.components.embedding import (
    SinusoidalPosEmb,
    SpacetimeEncoder,
)
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import array_to_sequence


class NeutrinoEventMultitaskTransformer(GNN):
    """Transformer with learned task tokens for multitask reconstruction.

    Node features are embedded with sinusoidal embeddings and an MLP, and
    processed by `n_attention_blocks` self-attention blocks, of which the first
    `n_rel` use a relative space-time bias. After block `inject_cls_after`,
    `n_tasks` task tokens (and optionally `shared_tokens` shared tokens) are
    prepended to the sequence. After each block listed in `cross_attention`,
    the tokens are additionally updated by a token-only attention block through
    a learned gate, `t + sigmoid(g) * (block(t) - t)`. The task tokens of the
    final block are projected to `out_dim` and concatenated, giving
    `n_tasks * out_dim` outputs that can be routed to tasks with
    `StandardModel(split=...)`.
    """

    def __init__(
        self,
        n_attention_blocks: int = 2,
        n_rel: int = 1,
        inject_cls_after: int = 0,
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
        cross_attention: Optional[List[int]] = None,
        embed_bias: bool = True,
        spacetime_time_index: int = 3,
        spacetime_time_scale: float = 3e4 / 500 * 3e-1,
    ):
        """Construct `NeutrinoEventMultitaskTransformer`.

        Args:
            n_attention_blocks: Number of self-attention blocks.
            n_rel: Number of leading blocks using the relative space-time
                bias.
            inject_cls_after: Index of the block before which the task (and
                shared) tokens are prepended to the sequence.
            hidden_dim: Latent feature dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Hidden-size ratio of the attention blocks' MLPs.
            n_features: Number of input node features.
            pre_emb_scale: Scale applied to the node features before the
                sinusoidal embedding.
            n_tasks: Number of task token groups (outputs).
            shared_tokens: Number of additional tokens that attend with the
                task tokens but are not returned.
            dropout: Dropout rate in the self-attention blocks.
            drop_path: Stochastic-depth rate in the token-only blocks.
            token_multiplier: Number of tokens per task (and per shared
                token); a task's tokens are concatenated before projection.
            out_dim: If given, each task's tokens are projected to `out_dim`
                by an MLP; otherwise the raw tokens are returned.
            cross_attention: Indices of the blocks after which the tokens are
                updated by a gated token-only attention block.
            embed_bias: If True, the space-time interval is sinusoidally
                embedded before its projection to the relative bias.
            spacetime_time_index: Node-feature column used as time in the
                relative space-time bias (columns 0-2 are the positions).
                E.g. for `ClusterSummaryFeatures` this is the column of
                `time_of_first_hit`.
            spacetime_time_scale: Light travel distance per unit of the
                time column, in units of the position columns (see
                `SpacetimeEncoder`). E.g. 0.6 for times in microseconds and
                positions in units of 500 m.
        """
        cross_attention = list(cross_attention or [])
        tot_out = (out_dim if out_dim is not None else hidden_dim) * n_tasks
        super().__init__(n_features, tot_out)

        assert (
            hidden_dim % num_heads == 0
        ), "hidden_dim must be divisible by num_heads"
        assert (
            n_rel < n_attention_blocks
        ), "n_rel must be less than n_attention_blocks"
        assert (
            inject_cls_after < n_attention_blocks
        ), "inject_cls_after must be less than n_attention_blocks"
        assert (
            inject_cls_after >= n_rel
        ), "inject_cls_after must be greater than or equal to n_rel"
        assert all(
            inject_cls_after <= c < n_attention_blocks for c in cross_attention
        ), "cross_attention blocks must lie in [inject_cls_after, n_blocks)"

        self.embedding = SinusoidalPosEmb(dim=hidden_dim)
        self.emb_mlp = nn.Sequential(
            nn.Linear(n_features * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.ESA = nn.ModuleList(
            [
                (Block_rel if i < n_rel else Block)(
                    hidden_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for i in range(n_attention_blocks)
            ]
        )

        n_tokens = (n_tasks + shared_tokens) * token_multiplier
        self.xTAMS = nn.ModuleList(
            [
                Block(
                    hidden_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                )
                for _ in cross_attention
            ]
        )
        self.gates = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, n_tokens, 1))
                for _ in cross_attention
            ]
        )

        self.task_tokens = nn.Parameter(
            torch.empty(1, n_tasks * token_multiplier, hidden_dim)
        )
        nn.init.xavier_normal_(self.task_tokens, gain=1.0)
        self.shared_tokens: Optional[nn.Parameter] = None
        if shared_tokens > 0:
            self.shared_tokens = nn.Parameter(
                torch.empty(1, shared_tokens * token_multiplier, hidden_dim)
            )
            nn.init.xavier_normal_(self.shared_tokens, gain=1.0)

        head_dim = hidden_dim // num_heads
        if n_rel > 0:
            # without the sinusoidal embedding the raw four-distance is
            # projected unclipped
            self.rel_pos = SpacetimeEncoder(
                head_dim,
                output_dim=head_dim,
                columns=(0, 1, 2, spacetime_time_index),
                time_scale=spacetime_time_scale,
                clip=4.0 if embed_bias else None,
                apply_sin_emb=embed_bias,
            )

        self.task_out: Optional[nn.Module] = None
        if out_dim is not None:
            self.task_out = nn.Sequential(
                nn.Linear(hidden_dim * token_multiplier, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )

        self.n_rel = n_rel
        self.n_tasks = n_tasks
        self.inject_cls_after = inject_cls_after
        self.cross_attention = cross_attention
        self.token_multiplier = token_multiplier
        self.pre_emb_scale = pre_emb_scale

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """Exclude the learned tokens from weight decay."""
        return {"task_tokens", "shared_tokens"}

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass to input data."""
        x, mask, _ = array_to_sequence(data.x, data.batch)
        batch_size = mask.shape[0]

        if self.n_rel > 0:
            rel_pos_bias = self.rel_pos(x)

        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf

        tokens = self.task_tokens.expand(batch_size, -1, -1)
        n_task_tokens = tokens.shape[1]
        n_shared = 0
        if self.shared_tokens is not None:
            shared = self.shared_tokens.expand(batch_size, -1, -1)
            n_shared = shared.shape[1]
            tokens = torch.cat([shared, tokens], 1)
        n_tokens = n_shared + n_task_tokens

        x = self.embedding(self.pre_emb_scale * x).flatten(-2)
        x = self.emb_mlp(x)

        for i, block in enumerate(self.ESA):
            if i == self.inject_cls_after:
                x = torch.cat([tokens, x], 1)
                attn_mask = create_attn_mask(mask, add_tokens=n_tokens)

            if i < self.n_rel:
                x = block(x, attn_mask, rel_pos_bias)
            else:
                x = block(x, None, attn_mask)

            if i in self.cross_attention:
                idx = self.cross_attention.index(i)
                token_part, x = x[:, :n_tokens], x[:, n_tokens:]
                # Gated residual update: interpolate between the tokens and
                # the output of the token-only block.
                gate = torch.sigmoid(self.gates[idx])
                token_part = token_part + gate * (
                    self.xTAMS[idx](token_part) - token_part
                )
                x = torch.cat([token_part, x], 1)

        # Task tokens of the final block (shared tokens are dropped).
        x = x[:, n_shared:n_tokens]
        if self.token_multiplier > 1:
            x = x.reshape(
                batch_size, self.n_tasks, self.token_multiplier * x.shape[-1]
            )
        if self.task_out is not None:
            x = self.task_out(x)
        return x.flatten(1, 2)


def create_attn_mask(mask: Tensor, add_tokens: int = 0) -> Tensor:
    """Create an additive key-padding mask, with `add_tokens` leading tokens.

    Args:
        mask: Boolean mask of valid sequence elements, shape [B, L].
        add_tokens: Number of (always valid) tokens prepended to the sequence.

    Returns:
        Float mask of shape [B, add_tokens + L], 0 where valid and -inf where
        padded.
    """
    if add_tokens > 0:
        mask = torch.cat(
            [
                torch.ones(
                    mask.shape[0],
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
