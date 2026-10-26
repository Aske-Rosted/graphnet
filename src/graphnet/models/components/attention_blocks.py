"""Attention and transformer blocks used in graphnet models."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.functional import Tensor
from torch.nn.functional import linear, scaled_dot_product_attention
from pytorch_lightning import LightningModule


class DropPath(LightningModule):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(
        self,
        drop_prob: float = 0.0,
    ):
        """Construct `DropPath`.

        Args:
            drop_prob: Probability of dropping a path during training.
                If 0.0, no paths are dropped. Defaults to None.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self) -> str:
        """Return extra representation of the module."""
        return "p={}".format(self.drop_prob)


class Mlp(LightningModule):
    """Multi-Layer Perceptron (MLP) module."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: nn.Module = nn.GELU,
        dropout_prob: float = 0.0,
    ):
        """Construct `Mlp`.

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden features. Defaults to None.
                If None, it is set to the value of `in_features`.
            out_features: Number of output features. Defaults to None.
                If None, it is set to the value of `in_features`.
            activation: Activation layer. Defaults to `nn.GELU`.
            dropout_prob: Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        if in_features <= 0:
            raise ValueError(
                f"in_features must be greater than 0, got in_features "
                f"{in_features} instead"
            )
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.input_projection = nn.Linear(in_features, hidden_features)
        self.activation = activation()
        self.output_projection = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.output_projection(x)
        x = self.dropout(x)
        return x


class Attention_rel(LightningModule):
    """Attention mechanism with relative position bias."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
    ):
        """Construct 'Attention_rel'.

        Args:
            input_dim: Dimension of the input tensor.
            num_heads: the number of attention heads to use (default: 8)
            qkv_bias: whether to add bias to the query, key, and value
                projections. Defaults to False.
            qk_scale: a scaling factor that multiplies the dot product of query
                and key vectors. Defaults to None. If None, computed as
                :math: `head_dim^(-1/2)`.
            attn_drop: the dropout probability for the attention weights.
                Defaults to 0.0.
            proj_drop: the dropout probability for the output of the attention
                module. Defaults to 0.0.
            attn_head_dim: the feature dimensionality of each attention head.
                Defaults to None. If None, computed as `dim // num_heads`.
        """
        if input_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"dim and num_heads must be greater than 0,"
                f" got input_dim={input_dim} and num_heads={num_heads} instead"
            )

        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim or input_dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.proj_q = nn.Linear(input_dim, all_head_dim, bias=False)
        self.proj_k = nn.Linear(input_dim, all_head_dim, bias=False)
        self.proj_v = nn.Linear(input_dim, all_head_dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, input_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        rel_pos_bias: Optional[
            Union[Tensor, Tuple[Tensor, Tensor, Tensor]]
        ] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        `rel_pos_bias` is either the relative encoding `[B, L, L, C]`
        (`C` the head dimension, shared by the heads), or its affine
        factors `(d, w, b)` from `SpacetimeEncoder.affine_factors` with
        encoding `d[..., None] * w + b`: the same attention without
        building the `[B, L, L, C]` tensor, since
        `sum_c q_c (d w_c + b_c) = d (q.w) + q.b` and
        `sum_j a_ij (d_ij w + b) = (sum_j a_ij d_ij) w + (sum_j a_ij) b`.
        The factors can also hold `K` basis functions of the pair, `phi` of
        shape `[B, K, L, L]` and `w` of shape `[K, C]`, for the encoding
        `sum_k phi_k[..., None] * w_k + b` (factored term by term).
        """
        batch_size, event_length, _ = q.shape

        q = linear(input=q, weight=self.proj_q.weight, bias=self.q_bias)
        q = q.reshape(batch_size, event_length, self.num_heads, -1).permute(
            0, 2, 1, 3
        )
        k = linear(input=k, weight=self.proj_k.weight, bias=None)
        k = k.reshape(batch_size, k.shape[1], self.num_heads, -1).permute(
            0, 2, 1, 3
        )
        v = linear(input=v, weight=self.proj_v.weight, bias=self.v_bias)
        v = v.reshape(batch_size, v.shape[1], self.num_heads, -1).permute(
            0, 2, 1, 3
        )

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        factored = isinstance(rel_pos_bias, tuple)
        if factored:
            distance, rel_weight, rel_bias = rel_pos_bias  # type: ignore
            basis = distance.dim() == 4  # [B, K, L, L] with weights [K, C]
            q_b = q @ rel_bias.to(q.dtype)
            if basis:
                q_w = q @ rel_weight.t().to(q.dtype)  # [B, H, L, K]
                for k in range(distance.shape[1]):
                    attn = attn + distance[:, k : k + 1].to(attn.dtype) * q_w[
                        ..., k
                    ].unsqueeze(-1)
                attn = attn + q_b.unsqueeze(-1)
            else:
                distance = distance.unsqueeze(1)  # [B, 1, L, L]
                q_w = q @ rel_weight.to(q.dtype)  # [B, H, L]
                attn = attn + (
                    distance.to(attn.dtype) * q_w.unsqueeze(-1)
                    + q_b.unsqueeze(-1)
                )
        elif rel_pos_bias is not None:
            bias = torch.einsum("bhic,bijc->bhij", q, rel_pos_bias)
            attn = attn + bias
        if key_padding_mask is not None:
            assert (
                key_padding_mask.dtype == torch.float32
                or key_padding_mask.dtype == torch.float16
            ), "incorrect mask dtype"
            bias = torch.min(
                key_padding_mask[:, None, :], key_padding_mask[:, :, None]
            )
            bias[
                torch.max(
                    key_padding_mask[:, None, :], key_padding_mask[:, :, None]
                )
                < 0
            ] = 0
            attn = attn + bias.unsqueeze(1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)
        if factored:
            # row sums of the (dropped-out) attention, not assumed to be 1
            row_sum = attn.sum(-1)
            if basis:
                weighted_basis = torch.stack(
                    [
                        (attn * distance[:, k : k + 1].to(attn.dtype)).sum(-1)
                        for k in range(distance.shape[1])
                    ],
                    -1,
                )  # [B, H, L, K]
                rel = weighted_basis @ rel_weight.to(
                    attn.dtype
                ) + row_sum.unsqueeze(-1) * rel_bias.to(attn.dtype)
            else:
                weighted_distance = (attn * distance.to(attn.dtype)).sum(-1)
                rel = weighted_distance.unsqueeze(-1) * rel_weight.to(
                    attn.dtype
                ) + row_sum.unsqueeze(-1) * rel_bias.to(attn.dtype)
            x = x + rel.transpose(1, 2).to(x.dtype)
        elif rel_pos_bias is not None:
            x = x + torch.einsum("bhij,bijc->bihc", attn, rel_pos_bias)
        x = x.reshape(batch_size, event_length, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block_rel(LightningModule):
    """Implementation of BEiTv2 Block."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        activation: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        attn_head_dim: Optional[int] = None,
    ):
        """Construct 'Block_rel'.

        Args:
            input_dim: Dimension of the input tensor.
            num_heads: Number of attention heads to use in the `Attention_rel`
            layer.
            mlp_ratio: Ratio of the hidden size of the feedforward network to
                the input size in the `Mlp` layer.
            qkv_bias: Whether or not to include bias terms in the query, key,
                and value matrices in the `Attention_rel` layer.
            qk_scale: Scaling factor for the dot product of the query and key
                matrices in the `Attention_rel` layer.
            dropout: Dropout probability to use in the `Mlp` layer.
            attn_drop: Dropout probability to use in the `Attention_rel` layer.
            drop_path: Probability of applying drop path regularization to the
                output of the layer.
            init_values: Initial value to use for the `gamma_1` and `gamma_2`
                parameters if not `None`.
            activation: Activation function to use in the `Mlp` layer.
            norm_layer: Normalization layer to use.
            attn_head_dim: Dimension of the attention head outputs in the
                `Attention_rel` layer.
        """
        super().__init__()
        self.norm1 = norm_layer(input_dim)
        self.attn = Attention_rel(
            input_dim,
            num_heads,
            attn_drop=attn_drop,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_head_dim=attn_head_dim,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(input_dim)
        mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=input_dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            dropout_prob=dropout,
        )

        if init_values is not None:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones(input_dim), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones(input_dim), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        rel_pos_bias: Optional[
            Union[Tensor, Tuple[Tensor, Tensor, Tensor]]
        ] = None,
        kv: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""
        if self.gamma_1 is None:
            xn = self.norm1(x)
            kv = xn if kv is None else self.norm1(kv)
            x = x + self.drop_path(
                self.attn(
                    xn,
                    kv,
                    kv,
                    rel_pos_bias=rel_pos_bias,
                    key_padding_mask=key_padding_mask,
                )
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            xn = self.norm1(x)
            kv = xn if kv is None else self.norm1(kv)
            x = x + self.drop_path(
                self.gamma_1
                * self.drop_path(
                    self.attn(
                        xn,
                        kv,
                        kv,
                        rel_pos_bias=rel_pos_bias,
                        key_padding_mask=key_padding_mask,
                    )
                )
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class QKNormAttention(nn.Module):
    """Multi-head attention with normalized queries and keys.

    As `nn.MultiheadAttention` (batch first), except that the query and key
    of every head are scaled to unit length before their dot product, and
    the result is multiplied by a learned temperature per head:
    `logit = tau_h * cos(q, k)`. The size of the attention logits is then
    bounded by `tau_h`, instead of growing with the norms of the
    projection weights (attention-logit growth, Dehghani et al. 2023,
    arXiv:2302.05442; Wortsman et al. 2023, arXiv:2309.14322). The
    temperatures are parametrized by their logarithm.

    The parameter names are those of `nn.MultiheadAttention`
    (`in_proj_weight`, `in_proj_bias`, `out_proj`), so its weights load
    directly; only `log_temperature` is new. Unlike there, the key bias is
    not redundant: the keys are normalized after it is added.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        temperature: Optional[float] = None,
    ):
        """Construct `QKNormAttention`.

        Args:
            embed_dim: Dimension of the input and output.
            num_heads: Number of attention heads.
            dropout: Dropout probability on the attention weights.
            temperature: Initial temperature of every head. Defaults to
                `sqrt(head_dim)`, the logit scale of the unnormalized
                attention for queries and keys with unit-variance entries.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads "
                f"({num_heads})."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.in_proj_weight = nn.Parameter(
            torch.empty(3 * embed_dim, embed_dim)
        )
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.out_proj.bias)
        if temperature is None:
            temperature = self.head_dim**0.5
        self.log_temperature = nn.Parameter(
            torch.full((num_heads,), float(temperature)).log()
        )

    def _project(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return the heads of q, k, v, each `[B, num_heads, L, head_dim]`."""
        if query is key and key is value:
            q, k, v = linear(
                query, self.in_proj_weight, self.in_proj_bias
            ).chunk(3, dim=-1)
        else:
            w_q, w_k, w_v = self.in_proj_weight.chunk(3)
            b_q, b_k, b_v = self.in_proj_bias.chunk(3)
            q = linear(query, w_q, b_q)
            k = linear(key, w_k, b_k)
            v = linear(value, w_v, b_v)

        def heads(x: Tensor) -> Tensor:
            return x.unflatten(-1, (self.num_heads, self.head_dim)).transpose(
                1, 2
            )

        return heads(q), heads(k), heads(v)

    def _normalize(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        """Scale queries and keys to unit length, the queries by `tau_h`.

        In float32, so that the normalization is exact under autocast.
        """
        tau = self.log_temperature.float().exp()[:, None, None]
        q32, k32 = q.float(), k.float()
        q = tau * q32 / q32.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        k = k32 / k32.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return q, k

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, None]:
        """Forward pass, with the masks of `nn.MultiheadAttention`.

        Args:
            query: `[B, L, embed_dim]`.
            key: `[B, S, embed_dim]`.
            value: `[B, S, embed_dim]`.
            attn_mask: `[L, S]` or `[B * num_heads, L, S]`; boolean (True:
                not attended) or additive float.
            key_padding_mask: `[B, S]`; boolean (True: padding) or additive
                float.
            need_weights: Must be False (the weights are not returned).

        Returns:
            The output `[B, L, embed_dim]` and None (in place of the
            attention weights of `nn.MultiheadAttention`).
        """
        assert not need_weights, "QKNormAttention does not return weights."
        batch_size, length, _ = query.shape
        q, k, v = self._project(query, key, value)
        q, k = self._normalize(q, k)
        q, k = q.to(v.dtype), k.to(v.dtype)

        mask = None
        if attn_mask is not None:
            mask = _additive_mask(attn_mask, v.dtype)
            if mask.dim() == 3:
                mask = mask.view(batch_size, self.num_heads, *mask.shape[1:])
        if key_padding_mask is not None:
            padding = _additive_mask(key_padding_mask, v.dtype)
            padding = padding[:, None, None, :]
            mask = padding if mask is None else mask + padding

        # the temperature is already in the queries: no 1 / sqrt(head_dim)
        x = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0,
        )
        x = x.transpose(1, 2).reshape(batch_size, length, self.embed_dim)
        return self.out_proj(x), None


def _additive_mask(mask: Tensor, dtype: torch.dtype) -> Tensor:
    """Return `mask` as an additive float mask (boolean True -> -inf)."""
    if mask.dtype == torch.bool:
        additive = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
        return additive.masked_fill(mask, -torch.inf)
    return mask.to(dtype)


class Block(LightningModule):
    """Transformer block."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        activation: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
    ):
        """Construct 'Block'.

        Args:
            input_dim: Dimension of the input tensor.
            num_heads: Number of attention heads to use in the
                `MultiheadAttention` layer.
            mlp_ratio: Ratio of the hidden size of the feedforward network to
                the input size in the `Mlp` layer.
            dropout: Dropout probability to use in the `Mlp` layer.
            attn_drop: Dropout probability to use in the `MultiheadAttention`
                layer.
            drop_path: Probability of applying drop path regularization to the
                output of the layer.
            init_values: Initial value to use for the `gamma_1` and `gamma_2`
                parameters if not `None`.
            activation: Activation function to use in the `Mlp` layer.
            norm_layer: Normalization layer to use.
            qk_norm: If True, use `QKNormAttention` (normalized queries and
                keys, learned temperature per head) instead of
                `nn.MultiheadAttention`. Same parameter names, plus
                `attn.log_temperature`.
        """
        super().__init__()
        self.norm1 = norm_layer(input_dim)
        if qk_norm:
            self.attn = QKNormAttention(
                input_dim, num_heads, dropout=attn_drop
            )
        else:
            self.attn = nn.MultiheadAttention(
                input_dim, num_heads, dropout=attn_drop, batch_first=True
            )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(input_dim)
        mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=input_dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            dropout_prob=dropout,
        )

        if init_values is not None:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((input_dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((input_dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""
        if self.gamma_1 is None:
            xn = self.norm1(x)
            x = x + self.drop_path(
                self.attn(
                    xn,
                    xn,
                    xn,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )[0]
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            xn = self.norm1(x)
            x = x + self.drop_path(
                self.gamma_1
                * self.attn(
                    xn,
                    xn,
                    xn,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )[0]
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
