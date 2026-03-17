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


class NeutrinoEventMultitaskTransformer_v4(GNN):
    """NeutrinoEventMultiTaskTransformer model."""

    def __init__(
        self,
        encoding_blocks: int = 0,
        latent_attention_blocks: int = 2,
        inject_cls_after: Optional[int] = 0,
        hidden_dim: int = 128,
        latent_tokens: int = 32,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        n_features: int = 36,
        pre_emb_scale: float = 1024.0,
        n_tasks: int = 1,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        token_multiplier: int = 1,
        out_dim: Optional[int] = None,
        cross_attention: list = [],
        latent_upscale: int = 1,
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

        self.embedding = SinusoidalPosEmb(dim=hidden_dim)
        self.emb_mlp = nn.Sequential(
            nn.Linear(n_features * (hidden_dim), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        Encoding_Blocks = nn.ModuleList()
        for _ in range(encoding_blocks):
            Encoding_Blocks.append(
                Block_rel(
                    hidden_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout
                )
            )

        self.Encoding_Blocks = Encoding_Blocks

        self.latents = nn.Parameter(
            torch.empty(1, latent_tokens, hidden_dim),
            requires_grad=True,
        )
        nn.init.xavier_normal_(self.latents, gain=1.0)

        if latent_upscale > 1:
            self.latent_upscale = nn.Linear(
                hidden_dim, hidden_dim * latent_upscale
            )
            hidden_dim = hidden_dim * latent_upscale
            num_heads = num_heads * latent_upscale
        else:
            self.latent_upscale = None

        ESA = nn.ModuleList()
        for _ in range(latent_attention_blocks):
            ESA.append(
                Block(
                    hidden_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout
                )
            )
        self.ESA = ESA

        xTAMS = nn.ModuleList()
        for i in range(len(cross_attention)):
            xTAMS.append(
                Block(
                    hidden_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                )
            )
        self.xTAMS = xTAMS

        self.task_tokens = nn.Parameter(
            torch.empty(1, n_tasks * token_multiplier, hidden_dim),
            requires_grad=True,
        )
        self.n_tasks = n_tasks

        nn.init.xavier_normal_(self.task_tokens, gain=1.0)

        assert (
            hidden_dim % num_heads == 0
        ), "hidden_dim must be divisible by num_heads"

        self.rel_pos = SpacetimeEncoder(hidden_dim // num_heads)

        # assert self.n_rel < latent_attention_blocks, "n_rel must be less than latent_attention_blocks"
        assert (
            inject_cls_after < latent_attention_blocks
        ), "inject_cls_after must be less than latent_attention_blocks"

        assert all(
            c < latent_attention_blocks for c in cross_attention
        ), "cross_attention must be less than latent_attention_blocks"
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
        return (
            {"task_tokens", "latents"}
            if hasattr(self, "latents")
            else {"task_tokens"}
        )

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass to input data."""
        x, mask, seq_len = array_to_sequence(data.x, data.batch)
        batch_size = mask.shape[0]

        rel_pos_bias = self.rel_pos(x)

        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf

        cls_token = self.task_tokens.expand(batch_size, -1, -1)

        x = self.embedding(self.pre_emb_scale * x).flatten(-2)
        x = self.emb_mlp(x)

        latents = self.latents.expand(batch_size, -1, -1)

        for i, blk in enumerate(self.Encoding_Blocks):
            # if last encoding block, perform cross attention with latents
            if i == len(self.Encoding_Blocks) - 1:
                # attn_mask = create_attn_mask(mask, batch_size, add_tokens=latents.shape[1])
                x = blk(
                    latents, key_padding_mask=None, rel_pos_bias=None, kv=x
                )
            else:
                x = blk(
                    x, key_padding_mask=attn_mask, rel_pos_bias=rel_pos_bias
                )

        if self.latent_upscale is not None:
            x = self.latent_upscale(x)

        for i, blk in enumerate(self.ESA):
            if i == self.inject_cls_after:
                # Inject cls tokens
                x = torch.cat([cls_token, x], 1)

            x = blk(x, None, None)

            if i in self.cross_attention:
                # pull out the cls tokens for cross attention
                cls = x[:, : cls_token.shape[1]]
                x = x[:, cls_token.shape[1] :]
                cls = self.xTAMS[self.cross_attention.index(i)](cls)
                x = torch.cat([cls, x], 1)

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
