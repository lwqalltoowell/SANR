# network.py
import numpy as np
import torch
import torch.nn as nn


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """
    Orthogonal initialization (CleanRL PPO style).

    std here is a weight scaling (gain), NOT Gaussian policy std.
    """
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class _NHiddenMLP(nn.Module):
    """
    Generic N-hidden-layer MLP backbone.

    Each hidden block: Linear -> optional LayerNorm -> ReLU
    Shape: in_dim -> hidden_size -> ... -> hidden_size (repeat num_layer times)

    Exposes out_dim for downstream heads.
    """
    def __init__(self, in_dim: int, hidden_size: int, num_layer: int, use_layernorm: bool = True, use_affine: bool = True):
        super().__init__()
        assert num_layer >= 1, "num_layer must be >= 1"

        layers = []
        d = in_dim
        for _ in range(num_layer):
            layers.append(layer_init(nn.Linear(d, hidden_size), std=np.sqrt(2)))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_size,elementwise_affine=use_affine))
            layers.append(nn.ReLU())
            d = hidden_size

        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_with_last_preact(self, x: torch.Tensor):
        """
        Return backbone output together with the last hidden Linear output before
        any optional LayerNorm/ReLU.
        """
        last_preact = None
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                last_preact = x

        if last_preact is None:
            raise RuntimeError("backbone must contain at least one Linear layer")
        return x, last_preact


class Critic(nn.Module):
    """
    State value function V(s).

    Style (streaming-like):
      backbone: _NHiddenMLP
      critic_head: Identity (stable hook/monitor point)
      value head: Linear -> 1

    Compatible with PPOAgent.get_value / get_action_and_value.
    """
    def __init__(self, obs_dim: int, hidden_size: int = 64, num_layer: int = 2, use_layernorm: bool = True, use_affine: bool = True):
        super().__init__()
        self.backbone = _NHiddenMLP(obs_dim, hidden_size, num_layer, use_layernorm=use_layernorm, use_affine=use_affine)

        # Stable attach point for feature-rank monitoring
        self.critic_head = nn.Identity()

        # Output head (CleanRL: last layer gain=1.0)
        self.linear_value = layer_init(nn.Linear(self.backbone.out_dim, 1), std=1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the representation to monitor (post-backbone, post-critic_head).
        Input:  [B, obs_dim] or [obs_dim]
        Output: [B, hidden]  or [hidden]
        """
        single = (x.dim() == 1)
        if single:
            x = x.unsqueeze(0)

        feats = self.backbone(x)
        feats = self.critic_head(feats)

        if single:
            feats = feats.squeeze(0)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  [B, obs_dim] or [obs_dim]
        Output: [B, 1]       or [1]
        """
        single = (x.dim() == 1)
        if single:
            x = x.unsqueeze(0)

        feats = self.backbone(x)
        feats = self.critic_head(feats)
        v = self.linear_value(feats)

        if single:
            v = v.squeeze(0)
        return v


class ActorMean(nn.Module):
    """
    Gaussian policy mean network μ(s).

    Note:
      - This module ONLY outputs mean.
      - Log-std is handled in PPOAgent via a state-independent nn.Parameter,
        matching CleanRL PPO.

    Style (streaming-like):
      backbone: _NHiddenMLP
      actor_head: Identity (stable hook/monitor point)
      mean head: Linear -> action_dim (CleanRL: last layer gain=0.01)
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64, num_layer: int = 2, use_layernorm: bool = True, use_affine: bool = True):
        super().__init__()
        self.backbone = _NHiddenMLP(obs_dim, hidden_size, num_layer, use_layernorm=use_layernorm, use_affine=use_affine)

        # Stable attach point for feature-rank monitoring
        self.actor_head = nn.Identity()

        # Output head (CleanRL: small init to start near-zero mean actions)
        self.linear_mu = layer_init(nn.Linear(self.backbone.out_dim, action_dim), std=0.01)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the representation to monitor (post-backbone, post-actor_head).
        Input:  [B, obs_dim] or [obs_dim]
        Output: [B, hidden]  or [hidden]
        """
        single = (x.dim() == 1)
        if single:
            x = x.unsqueeze(0)

        feats = self.backbone(x)
        feats = self.actor_head(feats)

        if single:
            feats = feats.squeeze(0)
        return feats

    def forward_features_with_preact(self, x: torch.Tensor):
        """
        Returns monitored features together with the last hidden pre-activation.
        Input:  [B, obs_dim] or [obs_dim]
        Output: ([B, hidden], [B, hidden]) or ([hidden], [hidden])
        """
        single = (x.dim() == 1)
        if single:
            x = x.unsqueeze(0)

        feats, preact_last = self.backbone.forward_with_last_preact(x)
        feats = self.actor_head(feats)

        if single:
            feats = feats.squeeze(0)
            preact_last = preact_last.squeeze(0)
        return feats, preact_last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  [B, obs_dim] or [obs_dim]
        Output: [B, action_dim] or [action_dim]
        """
        single = (x.dim() == 1)
        if single:
            x = x.unsqueeze(0)

        feats = self.backbone(x)
        feats = self.actor_head(feats)
        mu = self.linear_mu(feats)

        if single:
            mu = mu.squeeze(0)
        return mu

    def forward_with_preact(self, x: torch.Tensor):
        """
        Input:  [B, obs_dim] or [obs_dim]
        Output: ([B, action_dim], [B, hidden]) or ([action_dim], [hidden])
        """
        feats, preact_last = self.forward_features_with_preact(x)
        mu = self.linear_mu(feats)
        return mu, preact_last
