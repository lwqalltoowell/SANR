# monitors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn


@torch.no_grad()
def explained_var_rank(feats: torch.Tensor, var_threshold: float = 0.99) -> int:
    """
    Explained-variance rank via singular values.

    feats: [K, D] (any extra dims will be flattened)
    returns: smallest k s.t. cumulative explained variance >= var_threshold
    """
    if feats.ndim != 2:
        feats = feats.view(feats.shape[0], -1)
    K, D = feats.shape
    if K < 2:
        return 0

    x = feats - feats.mean(dim=0, keepdim=True)

    # S: [min(K,D)]
    S = torch.linalg.svdvals(x)
    variances = S ** 2
    total = variances.sum()
    if total <= 0:
        return 0

    cum = torch.cumsum(variances, dim=0) / total
    k = int((cum >= var_threshold).nonzero(as_tuple=False)[0].item() + 1)
    return k


@torch.no_grad()
def module_param_grad_norms(module: nn.Module) -> tuple[float, float]:
    """
    L2 norms of parameters and grads for a module.

    Returns:
        (param_norm, grad_norm)
    """
    p2 = 0.0
    g2 = 0.0
    for p in module.parameters():
        if p.data is not None:
            p2 += p.data.float().norm(2).item() ** 2
        if p.grad is not None:
            g2 += p.grad.data.float().norm(2).item() ** 2
    return (p2 ** 0.5), (g2 ** 0.5)


@torch.no_grad()
def params_norms(params) -> tuple[float, float]:
    """
    L2 norms of a given parameter iterable and their grads.
    Returns (param_norm, grad_norm).
    """
    p2 = 0.0
    g2 = 0.0
    for p in params:
        if p is None:
            continue
        p2 += p.data.float().norm(2).item() ** 2
        if p.grad is not None:
            g2 += p.grad.data.float().norm(2).item() ** 2
    return (p2 ** 0.5), (g2 ** 0.5)


def _collect_group_params(
    root: nn.Module,
    prefix: str,
    cls: type[nn.Module],
):
    """
    Collect parameters that belong to submodules of type `cls`
    whose module path starts with `prefix`.

    Example:
        prefix="backbone", cls=nn.Linear  -> all Linear params under root.backbone.*
        prefix="linear_mu", cls=nn.Linear -> head params
    """
    out = []

    for name, m in root.named_modules():
        if name == "":
            continue
        if not name.startswith(prefix):
            continue
        if not isinstance(m, cls):
            continue
        for p in m.parameters(recurse=False):
            out.append(p)
    return out


@dataclass
class GroupNormMonitor:
    """
    Monitor param/grad norms by groups for a module.

    Groups are defined as (prefix, module_type) selectors.
    - "total" is always computed as the full module norm.
    - other groups are computed by collecting params from matching submodules.

    Note: Grad norms are meaningful only after backward() of the last minibatch.
    """
    module: nn.Module
    tag: str
    every_iters: int = 1
    groups: Dict[str, Tuple[str, type[nn.Module]]] = None

    def __post_init__(self):
        if self.groups is None:
            self.groups = {}

        self._group_params: Dict[str, List[nn.Parameter]] = {}
        for gname, (prefix, cls) in self.groups.items():
            self._group_params[gname] = _collect_group_params(self.module, prefix, cls)

    def maybe_log(self, iteration: int) -> Optional[Dict[str, float]]:
        if self.every_iters <= 0:
            return None
        if iteration % self.every_iters != 0:
            return None

        metrics: Dict[str, float] = {}

        p_total, g_total = module_param_grad_norms(self.module)
        metrics[f"{self.tag}/param_norm"] = float(p_total)

        for gname, params in self._group_params.items():
            p, g = params_norms(params)
            metrics[f"{self.tag}/param_norm_{gname}"] = float(p)

        return metrics


@dataclass
class FeatureRankMonitor:
    """
    Monitor representation rank of a module using its forward_features() output.

    This is batch-PPO-friendly:
      - No hooks, no streaming buffer.
      - Each time, sample K observations from rollout batch and compute rank once.

    Requirements:
      - module must have method forward_features(x)->Tensor [B,H]
    """
    module: nn.Module
    tag: str
    every_iters: int = 10
    sample_size: int = 1024
    var_threshold: float = 0.99

    def maybe_log(self, iteration: int, obs_batch: torch.Tensor) -> Optional[Dict[str, float]]:
        if self.every_iters <= 0:
            return None
        if iteration % self.every_iters != 0:
            return None

        if not hasattr(self.module, "forward_features"):
            raise AttributeError(
                f"{type(self.module).__name__} has no forward_features(). "
                "Please expose a stable feature node in network.py."
            )

        B = obs_batch.shape[0]
        K = min(int(self.sample_size), B)
        if K < 2:
            return None

        idx = torch.randint(0, B, (K,), device=obs_batch.device)
        obs_sample = obs_batch[idx]

        feats = self.module.forward_features(obs_sample).float()  # [K,H]
        r = explained_var_rank(feats, self.var_threshold)

        return {f"{self.tag}/feature_rank": float(r)}


@dataclass
class PolicyStdMonitor:
    """
    Monitor state-independent policy std from actor_logstd parameter.
    """
    actor_logstd: torch.Tensor
    tag: str = "Monitor/Policy"
    every_iters: int = 1

    def maybe_log(self, iteration: int) -> Optional[Dict[str, float]]:
        if self.every_iters <= 0:
            return None
        if iteration % self.every_iters != 0:
            return None

        std = torch.exp(self.actor_logstd.detach())
        return {
            f"{self.tag}/std_mean": float(std.mean().item()),
            f"{self.tag}/std_min": float(std.min().item()),
            f"{self.tag}/std_max": float(std.max().item()),
        }



@dataclass
class DormantNeuronMonitor:
    """
    ReDo-style dormant neuron monitor for MLP backbones.

    What it measures:
      1) For each tracked activation layer output h in [B, H],
         compute per-unit score:
             a_j = mean_b |h[b, j]|
      2) Normalize within layer:
             s_j = a_j / (mean(a) + eps)
      3) Mark dormant if:
             s_j <= tau

    Designed for your PPO setting:
      - module is ActorMean or Critic
      - obs_batch is flattened rollout observations [B, obs_dim]
      - default logging focuses on GLOBAL dormant ratio only

    Notes:
      - Works for ReLU and LeakyReLU because it uses activation magnitude,
        not exact zeros, as the primary criterion.
      - zero-ratio is optional auxiliary info and is only meaningful for ReLU-like sparsity.
    """
    module: nn.Module
    tag: str
    every_iters: int = 1
    sample_size: int = 1024
    tau: float = 0.025
    eps: float = 1e-8
    log_per_layer: bool = False
    log_zero_ratio: bool = False
    aux_tau: float | None = None
    activation_types: Tuple[type[nn.Module], ...] = (nn.ReLU, nn.LeakyReLU)

    def _collect_target_layers(self) -> List[Tuple[str, nn.Module]]:
        """
        Collect activation modules under `module.backbone` only.
        Keeps logging focused on hidden units instead of heads.
        """
        if not hasattr(self.module, "backbone"):
            raise AttributeError(
                f"{type(self.module).__name__} has no attribute `backbone`; "
                "DormantNeuronMonitor expects actor/critic with a backbone."
            )

        targets: List[Tuple[str, nn.Module]] = []
        for name, m in self.module.backbone.named_modules():
            if isinstance(m, self.activation_types):
                targets.append((name, m))
        return targets

    @torch.no_grad()
    def maybe_log(self, iteration: int, obs_batch: torch.Tensor) -> Optional[Dict[str, float]]:
        if self.every_iters <= 0:
            return None
        if iteration % self.every_iters != 0:
            return None

        B = obs_batch.shape[0]
        if B < 2:
            return None

        K = min(int(self.sample_size), B) if self.sample_size is not None else B
        if K < B:
            idx = torch.randint(0, B, (K,), device=obs_batch.device)
            obs_sample = obs_batch[idx]
        else:
            obs_sample = obs_batch

        targets = self._collect_target_layers()
        if len(targets) == 0:
            return {
                f"{self.tag}/Dormant/global_ratio": 0.0,
                f"{self.tag}/Dormant/num_layers": 0.0,
                f"{self.tag}/Dormant/num_units": 0.0,
            }

        activations: Dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module, inputs, output):
                activations[layer_name] = output.detach()
            return hook

        for layer_name, layer in targets:
            hooks.append(layer.register_forward_hook(make_hook(layer_name)))

        was_training = self.module.training
        self.module.eval()
        try:
            _ = self.module(obs_sample)
        finally:
            for h in hooks:
                h.remove()
            if was_training:
                self.module.train()

        metrics: Dict[str, float] = {}

        total_units = 0
        total_dormant = 0
        total_dormant_aux = 0
        total_zero_elements = 0
        total_elements = 0

        layer_ratios = []

        for layer_idx, (layer_name, layer_module) in enumerate(targets):
            if layer_name not in activations:
                continue

            h = activations[layer_name].float()

            # Expect MLP hidden activations in shape [B, H].
            # If some extra dims sneak in, flatten feature dims.
            if h.ndim == 1:
                h = h.unsqueeze(0)
            elif h.ndim != 2:
                h = h.view(h.shape[0], -1)

            # Per-unit average activation magnitude
            score = h.abs().mean(dim=0)  # [H]

            # Layer-normalized score
            norm_score = score / (score.mean() + self.eps)

            dormant_mask = norm_score <= self.tau
            dormant_ratio = dormant_mask.float().mean().item()

            H = score.numel()
            num_dormant = int(dormant_mask.sum().item())

            total_units += H
            total_dormant += num_dormant

            if self.aux_tau is not None:
                num_dormant_aux = int((norm_score <= self.aux_tau).sum().item())
                total_dormant_aux += num_dormant_aux
            layer_ratios.append(dormant_ratio)

            if self.log_zero_ratio:
                zero_ratio = (h.abs() <= self.eps).float().mean().item()
                total_zero_elements += int((h.abs() <= self.eps).sum().item())
                total_elements += h.numel()
            else:
                zero_ratio = None

            if self.log_per_layer:
                metrics[f"{self.tag}/Dormant/layer_{layer_idx}_ratio"] = float(dormant_ratio)
                metrics[f"{self.tag}/Dormant/layer_{layer_idx}_mean_score"] = float(norm_score.mean().item())
                metrics[f"{self.tag}/Dormant/layer_{layer_idx}_min_score"] = float(norm_score.min().item())
                metrics[f"{self.tag}/Dormant/layer_{layer_idx}_num_units"] = float(H)
                if zero_ratio is not None:
                    metrics[f"{self.tag}/Dormant/layer_{layer_idx}_zero_ratio"] = float(zero_ratio)

        global_ratio = (total_dormant / total_units) if total_units > 0 else 0.0
        layer_mean_ratio = (sum(layer_ratios) / len(layer_ratios)) if len(layer_ratios) > 0 else 0.0

        metrics[f"{self.tag}/Dormant/global_ratio"] = float(global_ratio)
        # metrics[f"{self.tag}/Dormant/layer_mean_ratio"] = float(layer_mean_ratio)
        # metrics[f"{self.tag}/Dormant/num_layers"] = float(len(layer_ratios))
        # metrics[f"{self.tag}/Dormant/num_units"] = float(total_units)
        # metrics[f"{self.tag}/Dormant/tau"] = float(self.tau)

        if self.aux_tau is not None:
            aux_global_ratio = (total_dormant_aux / total_units) if total_units > 0 else 0.0
            metrics[f"{self.tag}/Dormant/redo_ratio"] = float(aux_global_ratio)

        if self.log_zero_ratio:
            global_zero_ratio = (total_zero_elements / total_elements) if total_elements > 0 else 0.0
            metrics[f"{self.tag}/Dormant/global_zero_ratio"] = float(global_zero_ratio)

        return metrics


@dataclass
class GradientDormantMonitor:
    """
    Gradient-magnitude based dormant neuron monitor (GraMa-style), critic-only.

    Reads linear.weight.grad from backbone Linear layers after an external
    backward pass. Does NOT trigger its own backward — caller is responsible
    for providing valid gradients in the window between backward() and zero_grad().

    Criterion mirrors GradientReDoResetter in plasticity.py:
        grad_mag  = weight.grad.abs().mean(dim=1)   # per output neuron
        norm_score = grad_mag / (mean(grad_mag) + eps)
        dormant   = norm_score <= tau
    """
    module: nn.Module
    tag: str
    every_iters: int = 1
    tau: float = 0.05
    eps: float = 1e-9

    def log_from_grads(self, iteration: int) -> Optional[Dict[str, float]]:
        if self.every_iters <= 0 or iteration % self.every_iters != 0:
            return None
        if not hasattr(self.module, "backbone"):
            return None

        total, dormant = 0, 0
        for m in self.module.backbone.modules():
            if not isinstance(m, nn.Linear):
                continue
            if m.weight.grad is None:
                continue
            grad_mag = m.weight.grad.float().abs().mean(dim=1)  # [out_features]
            norm_score = grad_mag / (grad_mag.mean() + self.eps)
            total += grad_mag.numel()
            dormant += int((norm_score <= self.tau).sum().item())

        if total == 0:
            return None
        return {f"ReGraMa/{self.tag}/grad_dormant_ratio": dormant / total}
