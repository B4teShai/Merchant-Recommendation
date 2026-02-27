# -*- coding: utf-8 -*-
"""
PyTorch NN layers and helpers (replacement for TF NNLayers).
Provides: FC (LinearBlock), Bias, BatchNorm, activations, dropout, L1/L2 regularization,
SelfAttentionBlock, LightSelfAttentionBlock. No global parameter state; all state lives in nn.Modules.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default for leaky-style activations (avoid mutable global)
DEFAULT_LEAKY = 0.1


def get_activation(
    name: str,
    leaky: float = DEFAULT_LEAKY,
) -> Optional[Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]]:
    """
    Return activation module or callable for the given name.
    Supports: relu, sigmoid, tanh, softmax, leakyRelu, twoWayLeakyRelu6,
    -1relu (minus_one_relu), relu6, relu3.
    """
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "softmax":
        return lambda x: F.softmax(x, dim=-1)
    if name == "leakyRelu":
        return nn.LeakyReLU(negative_slope=leaky)
    if name == "twoWayLeakyRelu6":
        return _TwoWayLeakyRelu6(leaky=leaky)
    if name == "-1relu" or name == "minus_one_relu":
        return _MinusOneReLU()
    if name == "relu6":
        return nn.ReLU6()
    if name == "relu3":
        return _ReLU3()
    return None


class _TwoWayLeakyRelu6(nn.Module):
    """temMask * (6 + leaky*(data-6)) + (1-temMask) * max(leaky*data, data)."""

    def __init__(self, leaky: float = DEFAULT_LEAKY):
        super().__init__()
        self.leaky = leaky

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x > 6.0).to(x.dtype)
        return mask * (6.0 + self.leaky * (x - 6.0)) + (1.0 - mask) * torch.maximum(
            self.leaky * x, x
        )


class _MinusOneReLU(nn.Module):
    """max(-1, x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=-1.0)


class _ReLU3(nn.Module):
    """clamp to [0, 3]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0.0, max=3.0)


class Bias(nn.Module):
    """Add a learnable bias of shape (dim,) to the last dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class FC(nn.Module):
    """
    Fully connected block: Linear -> optional BatchNorm -> optional activation -> optional Dropout.
    Replaces TF FC with per-module parameters; no global param dict.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bias: bool = True,
        activation: Optional[str] = None,
        use_bn: bool = False,
        dropout: float = 0.0,
        initializer: str = "xavier",
        leaky: float = DEFAULT_LEAKY,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=use_bias and not use_bn)
        self.use_bn = use_bn
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else None
        self.activation_fn = get_activation(activation, leaky=leaky) if activation else None
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self._reset_parameters(initializer)

    def _reset_parameters(self, initializer: str) -> None:
        if initializer == "xavier":
            nn.init.xavier_uniform_(self.linear.weight)
        elif initializer == "zeros":
            nn.init.zeros_(self.linear.weight)
        elif initializer == "ones":
            nn.init.ones_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.bn is not None:
            # BatchNorm1d expects (N, C) or (N, C, L); flatten batch*seq for 3D input
            if x.dim() == 3:
                b, s, c = x.shape
                x = self.bn(x.view(b * s, c)).view(b, s, c)
            else:
                x = self.bn(x)
        if self.activation_fn is not None:
            if callable(self.activation_fn) and not isinstance(
                self.activation_fn, nn.Module
            ):
                x = self.activation_fn(x)
            else:
                x = self.activation_fn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


def regularize(
    parameters_or_modules: Union[
        List[torch.Tensor], List[nn.Parameter], nn.Module, List[nn.Module]
    ],
    method: str = "L2",
) -> torch.Tensor:
    """
    Return a scalar regularization loss (sum of abs for L1, sum of squares for L2).
    Pass a list of parameters, a single module (uses .parameters()), or a list of modules.
    """
    if isinstance(parameters_or_modules, nn.Module):
        params = list(parameters_or_modules.parameters())
    elif isinstance(parameters_or_modules, list) and len(parameters_or_modules) > 0:
        if isinstance(parameters_or_modules[0], nn.Module):
            params = []
            for m in parameters_or_modules:
                params.extend(m.parameters())
        else:
            params = list(parameters_or_modules)
    else:
        params = list(parameters_or_modules)
    if not params:
        return torch.tensor(0.0)
    out: Optional[torch.Tensor] = None
    for p in params:
        if not p.requires_grad:
            continue
        if method == "L1":
            term = p.abs().sum()
        elif method == "L2":
            term = (p ** 2).sum()
        else:
            raise ValueError("method must be 'L1' or 'L2'")
        out = term if out is None else out + term
    if out is None:
        return torch.tensor(0.0, device=params[0].device)
    return out


class SelfAttentionBlock(nn.Module):
    """
    Self-attention over a list of local representations (each [batch, inp_dim]).
    Uses one set of Q, K, V projections created once. Output: list of tensors,
    each out[i] = attention_output[i] + localReps[i] (residual).
    """

    def __init__(self, inp_dim: int, num_heads: int):
        super().__init__()
        assert inp_dim % num_heads == 0, "inp_dim must be divisible by num_heads"
        self.inp_dim = inp_dim
        self.num_heads = num_heads
        self.d = inp_dim // num_heads
        self.W_Q = nn.Linear(inp_dim, inp_dim)
        self.W_K = nn.Linear(inp_dim, inp_dim)
        self.W_V = nn.Linear(inp_dim, inp_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in (self.W_Q, self.W_K, self.W_V):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self, local_reps: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        local_reps: list of tensors, each [batch, inp_dim], length N.
        Returns: list of N tensors, each [batch, inp_dim], with residual add.
        """
        if not local_reps:
            return []
        number = len(local_reps)
        # Stack to [B, N, D]
        x = torch.stack(local_reps, dim=1)
        B, N, D = x.shape
        rsp = x.reshape(-1, D)
        q = self.W_Q(rsp).reshape(B, N, 1, self.num_heads, self.d)
        k = self.W_K(rsp).reshape(B, 1, N, self.num_heads, self.d)
        v = self.W_V(rsp).reshape(B, 1, N, self.num_heads, self.d)
        scores = (q * k).sum(dim=-1) / math.sqrt(self.d)
        att = F.softmax(scores, dim=2)
        attval = (att.unsqueeze(-1) * v).sum(dim=2).reshape(B, N, D)
        return [attval[:, i] + local_reps[i] for i in range(number)]


class LightSelfAttentionBlock(nn.Module):
    """
    Light self-attention: single Q projection (K=Q), V from raw reps.
    Same interface as SelfAttentionBlock: list of [batch, inp_dim] -> list with residual add.
    """

    def __init__(self, inp_dim: int, num_heads: int):
        super().__init__()
        assert inp_dim % num_heads == 0, "inp_dim must be divisible by num_heads"
        self.inp_dim = inp_dim
        self.num_heads = num_heads
        self.d = inp_dim // num_heads
        self.W_Q = nn.Linear(inp_dim, inp_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_Q.weight)
        if self.W_Q.bias is not None:
            nn.init.zeros_(self.W_Q.bias)

    def forward(
        self, local_reps: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        local_reps: list of tensors, each [batch, inp_dim], length N.
        Returns: list of N tensors, each [batch, inp_dim], with residual add.
        """
        if not local_reps:
            return []
        number = len(local_reps)
        x = torch.stack(local_reps, dim=1)
        B, N, D = x.shape
        rsp = x.reshape(-1, D)
        tem = self.W_Q(rsp)
        q = tem.reshape(B, N, 1, self.num_heads, self.d)
        k = tem.reshape(B, 1, N, self.num_heads, self.d)
        v = x.reshape(B, 1, N, self.num_heads, self.d)
        scores = (q * k).sum(dim=-1) / math.sqrt(self.d)
        att = F.softmax(scores, dim=2)
        attval = (att.unsqueeze(-1) * v).sum(dim=2).reshape(B, N, D)
        return [attval[:, i] + local_reps[i] for i in range(number)]


__all__ = [
    "FC",
    "Bias",
    "get_activation",
    "regularize",
    "SelfAttentionBlock",
    "LightSelfAttentionBlock",
    "DEFAULT_LEAKY",
]
