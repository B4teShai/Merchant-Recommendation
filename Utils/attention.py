# -*- coding: utf-8 -*-
"""
Attention modules for SelfGNN (PyTorch): AdditiveAttention (interval-level pooling, Eq. 6-7),
ScaledDotProductAttention, and MultiHeadSelfAttention (interval-level and instance-level self-attention).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    Additive attention over a sequence; used for interval-level aggregation.
    Input [batch, candidate_size, dim] -> output [batch, dim].
    """

    def __init__(self, query_vector_dim: int, candidate_vector_dim: int):
        super().__init__()
        self.query_vector_dim = query_vector_dim
        self.candidate_vector_dim = candidate_vector_dim
        self.dense = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim, 1).uniform_(-0.1, 0.1)
        )

    def forward(self, candidate_vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            candidate_vectors: [batch_size, candidate_size, candidate_vector_dim]
        Returns:
            [batch_size, candidate_vector_dim]
        """
        dense_out = self.dense(candidate_vectors)
        temp = torch.tanh(dense_out)
        candidate_weights = F.softmax(
            torch.matmul(temp, self.attention_query_vector).squeeze(-1), dim=1
        )
        target = torch.bmm(
            candidate_weights.unsqueeze(1), candidate_vectors
        ).squeeze(1)
        return target

    def attention(self, candidate_vectors: torch.Tensor) -> torch.Tensor:
        """Backward-compatible alias for forward()."""
        return self.forward(candidate_vectors)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention: scores = QK^T/sqrt(d_k), softmax, then apply to V.
    Used inside MultiHeadSelfAttention. Supports additive or boolean attention mask.
    """

    def __init__(self, d_k: int, attn_dropout: float = 0.0):
        super().__init__()
        self.d_k = d_k
        self.attn_dropout = attn_dropout

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q, K, V: [batch, head_num, candidate_num, d_k]
            attn_mask: Optional. If bool: True = attend, False = mask.
                       If float: additive mask (0 = attend, large negative = mask).
        Returns:
            context: [batch, head_num, candidate_num, d_k]
            attn_weights: [batch, head_num, candidate_num, candidate_num]
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, -1e9)
            else:
                scores = scores + attn_mask

        attn = F.softmax(scores, dim=-1)
        if self.attn_dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.attn_dropout)
        context = torch.matmul(attn, V)
        return context, attn

    def attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible alias for forward()."""
        return self.forward(Q, K, V, attn_mask)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention (Q=K=V). Used for interval-level (Eq. 6) and instance-level (Eq. 9).
    Input [batch, seq_len, d_model] -> output [batch, seq_len, d_model].
    """

    def __init__(
        self,
        d_model: int,
        num_attention_heads: int,
        attn_dropout: float = 0.0,
        use_output_projection: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0, (
            "d_model must be divisible by num_attention_heads"
        )
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.scaled_attn = ScaledDotProductAttention(self.d_k, attn_dropout=attn_dropout)
        self.use_output_projection = use_output_projection
        if use_output_projection:
            self.W_O = nn.Linear(d_model, d_model)

    def forward(
        self,
        Q: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            Q: [batch_size, seq_len, d_model]. If K,V None then self-attention (K=V=Q).
            K, V: Optional; if None, set to Q.
            attn_mask: Optional attention mask for scaled dot-product.
            length: Ignored (kept for API compatibility).
        Returns:
            [batch_size, seq_len, d_model]
        """
        if K is None:
            K = Q
        if V is None:
            V = Q

        batch_size = Q.size(0)
        W_Q = self.W_Q(Q)
        W_K = self.W_K(K)
        W_V = self.W_V(V)
        q_s = W_Q.view(batch_size, -1, self.num_attention_heads, self.d_k).transpose(
            1, 2
        )
        k_s = W_K.view(batch_size, -1, self.num_attention_heads, self.d_k).transpose(
            1, 2
        )
        v_s = W_V.view(batch_size, -1, self.num_attention_heads, self.d_v).transpose(
            1, 2
        )
        context, _ = self.scaled_attn(q_s, k_s, v_s, attn_mask=attn_mask)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_attention_heads * self.d_v)
        )
        if self.use_output_projection:
            context = self.W_O(context)
        return context

    def attention(
        self,
        Q: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        length: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Backward-compatible alias for forward()."""
        return self.forward(Q, K=K, V=V, attn_mask=attn_mask, length=length)


if __name__ == "__main__":
    B, L, D = 2, 5, 8
    query_dim = 16

    # AdditiveAttention
    add_attn = AdditiveAttention(query_vector_dim=query_dim, candidate_vector_dim=D)
    x = torch.randn(B, L, D)
    out = add_attn(x)
    assert out.shape == (B, D), f"AdditiveAttention: expected (B, D), got {out.shape}"
    out.sum().backward()
    print("AdditiveAttention: shapes OK, gradients flow.")

    # ScaledDotProductAttention (with heads)
    d_k = 4
    heads = 2
    sdpa = ScaledDotProductAttention(d_k, attn_dropout=0.1)
    q = k = v = torch.randn(B, heads, L, d_k)
    ctx, attn = sdpa(q, k, v)
    assert ctx.shape == (B, heads, L, d_k)
    assert attn.shape == (B, heads, L, L)
    ctx.sum().backward()
    print("ScaledDotProductAttention: shapes OK, gradients flow.")

    # MultiHeadSelfAttention
    d_model = 8
    mh = MultiHeadSelfAttention(d_model, num_attention_heads=2, attn_dropout=0.1)
    q = torch.randn(B, L, d_model)
    out = mh(q)
    assert out.shape == (B, L, d_model), f"MultiHeadSelfAttention: expected (B, L, d_model), got {out.shape}"
    out.sum().backward()
    print("MultiHeadSelfAttention: shapes OK, gradients flow.")
    print("All attention modules OK.")