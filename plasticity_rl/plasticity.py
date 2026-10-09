# plasticity.py
import numpy as np
import torch
import torch.nn as nn
from math import sqrt
from torch.optim.optimizer import Optimizer


class RegenerativeRegularizer:
    """
    L2-to-reference regularizer that pulls parameters toward a saved snapshot.
    """

    def __init__(self, args):
        self.coef = float(getattr(args, "regen_coef", 0.0))
        self.reference_params = {}

    @torch.no_grad()
    def capture_reference(self, named_parameters):
        self.reference_params = {
            name: param.detach().clone()
            for name, param in named_parameters
            if param.requires_grad
        }

    @torch.no_grad()
    def load_reference(self, reference_state):
        if reference_state is None:
            self.reference_params = {}
            return

        self.reference_params = {
            name: param.detach().clone()
            for name, param in reference_state.items()
        }

    def state_dict(self):
        return {
            name: param.detach().clone()
            for name, param in self.reference_params.items()
        }

    def loss(self, named_parameters):
        named_parameters = list(named_parameters)
        device = named_parameters[0][1].device if named_parameters else torch.device("cpu")

        if self.coef <= 0.0 or not self.reference_params:
            zero = torch.tensor(0.0, device=device)
            return zero, {
                "regen_loss": zero,
                "regen_penalty": zero,
            }

        penalty = None

        for name, param in named_parameters:
            if not param.requires_grad:
                continue
            if name not in self.reference_params:
                continue

            ref = self.reference_params[name].to(device=param.device, dtype=param.dtype)
            term = (param - ref).pow(2).sum()
            penalty = term if penalty is None else penalty + term

        if penalty is None:
            zero = torch.tensor(0.0, device=device)
            return zero, {
                "regen_loss": zero,
                "regen_penalty": zero,
            }

        regen_loss = self.coef * penalty
        return regen_loss, {
            "regen_loss": regen_loss,
            "regen_penalty": penalty,
        }


class PFORegularizer:
    """
    Policy Feature Optimization (PFO-last) on actor pre-activations.
    """

    def __init__(self, args):
        self.coef = float(getattr(args, "pfo_coef", 0.0))

    def loss(self, preact_new: torch.Tensor, preact_old: torch.Tensor):
        device = preact_new.device

        if self.coef <= 0.0:
            zero = torch.tensor(0.0, device=device)
            return zero, {
                "pfo_loss": zero,
                "pfo_penalty": zero,
            }

        preact_new = preact_new.view(preact_new.shape[0], -1)
        preact_old = preact_old.view(preact_old.shape[0], -1)
        penalty = (preact_new - preact_old).pow(2).sum(dim=-1).mean()
        pfo_loss = self.coef * penalty
        return pfo_loss, {
            "pfo_loss": pfo_loss,
            "pfo_penalty": penalty,
        }


class SoftShrinkPerturber:
    """
    Post-optimizer-step Soft Shrink-and-Perturb with fresh init samples.
    """

    def __init__(self, args, actor_mean: nn.Module, critic: nn.Module):
        self.beta = float(getattr(args, "snp_beta", 0.0))
        if self.beta < 0.0 or self.beta >= 1.0:
            raise ValueError("snp_beta must be in [0, 1).")

        self.seed = int(getattr(args, "seed", 0)) + 10000
        self.targets = []
        self.generators = {}

        if self.beta > 0.0:
            self._register_backbone(actor_mean.backbone)
            self._register_linear(actor_mean.linear_mu, gain=0.01)
            self._register_backbone(critic.backbone)
            self._register_linear(critic.linear_value, gain=1.0)
            print(
                "[SNP Enabled] "
                f"beta={self.beta}, targets={len(self.targets)}, seed={self.seed}"
            )

    @torch.no_grad()
    def _register_backbone(self, module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                self._register_linear(m, gain=np.sqrt(2))
            elif isinstance(m, nn.LayerNorm):
                self._register_layernorm(m)

    @torch.no_grad()
    def _register_linear(self, layer: nn.Linear, gain: float):
        self.targets.append({
            "param": layer.weight,
            "kind": "orthogonal",
            "gain": float(gain),
        })
        if layer.bias is not None:
            self.targets.append({
                "param": layer.bias,
                "kind": "constant",
                "value": 0.0,
            })

    @torch.no_grad()
    def _register_layernorm(self, layer: nn.LayerNorm):
        if not layer.elementwise_affine:
            return
        self.targets.append({
            "param": layer.weight,
            "kind": "constant",
            "value": 1.0,
        })
        self.targets.append({
            "param": layer.bias,
            "kind": "constant",
            "value": 0.0,
        })

    def _generator_for(self, device: torch.device):
        key = str(device)
        if key not in self.generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            self.generators[key] = generator
        return self.generators[key]

    @torch.no_grad()
    def _fresh_sample(self, target: dict):
        param = target["param"]
        sample = torch.empty_like(param)

        if target["kind"] == "orthogonal":
            nn.init.orthogonal_(
                sample,
                gain=target["gain"],
                generator=self._generator_for(param.device),
            )
        elif target["kind"] == "constant":
            sample.fill_(target["value"])
        else:
            raise ValueError(f"Unknown SNP target kind: {target['kind']}")

        return sample

    @torch.no_grad()
    def after_optimizer_step(self):
        if self.beta <= 0.0 or not self.targets:
            return

        alpha = 1.0 - self.beta
        for target in self.targets:
            param = target["param"]
            sample = self._fresh_sample(target)
            param.mul_(alpha).add_(sample, alpha=self.beta)


class SoftNormAnchoring:
    """
    Post-optimizer-step Soft Norm Anchoring for Linear layers.
    """

    def __init__(self, args, modules):
        if not hasattr(args, "sna_lambda"):
            print("[Warning] args.sna_lambda not found. Soft Norm Anchoring (SNA) is disabled.")

        self.lambda_base = float(getattr(args, "sna_lambda", 0.0))
        self.targets = []

        if self.lambda_base > 0.0:
            for module in modules:
                self._register_targets(module)
            print(f"[SNA Enabled] Registered {len(self.targets)} Linear layers for Soft Norm Anchoring.")

    @torch.no_grad()
    def _register_targets(self, module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                init_norm = torch.norm(m.weight, p="fro").item()
                self.targets.append({
                    "layer": m,
                    "init_norm": init_norm,
                })

    @torch.no_grad()
    def after_optimizer_step(self, lr_frac: float):
        current_lambda = self.lambda_base * lr_frac
        if current_lambda <= 0.0 or not self.targets:
            return

        for target in self.targets:
            layer = target["layer"]
            init_norm = target["init_norm"]
            curr_norm = torch.norm(layer.weight, p="fro").item()

            if curr_norm > 1e-8:
                c_l = (current_lambda * init_norm + (1.0 - current_lambda) * curr_norm) / curr_norm
                layer.weight.mul_(c_l)
                if layer.bias is not None:
                    layer.bias.mul_(c_l)


class ReDoResetter:
    """
    Periodically reset dormant hidden units in Linear -> ReLU backbones.

    If a hidden block includes LayerNorm as Linear -> LayerNorm -> ReLU, the
    matching LayerNorm entries are reset as well.
    """

    def __init__(self, args):
        self.use_redo = bool(getattr(args, "use_redo", False))
        self.frequency = int(getattr(args, "redo_frequency", 1))
        self.tau = float(getattr(args, "redo_tau", 0.025))
        self.batch_size = int(getattr(args, "redo_batch_size", 1024))
        self.eps = float(getattr(args, "redo_eps", 1e-9))

        if self.use_redo:
            print(
                "[ReDo Enabled] "
                f"frequency={self.frequency}, tau={self.tau}, batch_size={self.batch_size}"
            )

    def _layers(self, module: nn.Module):
        if not hasattr(module, "backbone") or not hasattr(module.backbone, "net"):
            raise AttributeError(
                f"{type(module).__name__} must expose backbone.net for ReDo."
            )

        blocks = []
        layers = list(module.backbone.net)
        i = 0
        while i < len(layers):
            linear = layers[i]
            if not isinstance(linear, nn.Linear):
                raise ValueError(
                    "ReDo expects backbone.net blocks to start with Linear."
                )

            i += 1
            layer_norm = None
            if i < len(layers) and isinstance(layers[i], nn.LayerNorm):
                layer_norm = layers[i]
                i += 1

            if i >= len(layers) or not isinstance(layers[i], nn.ReLU):
                raise ValueError(
                    "ReDo expects each backbone block to be Linear -> ReLU "
                    "or Linear -> LayerNorm -> ReLU."
                )
            relu = layers[i]
            i += 1

            blocks.append((linear, layer_norm, relu))
        return blocks

    @torch.no_grad()
    def _zero_optimizer_state_rows(self, optimizer, param: nn.Parameter, rows: torch.Tensor):
        state = optimizer.state.get(param)
        if not state:
            return
        for value in state.values():
            if torch.is_tensor(value) and value.shape == param.shape:
                value.index_fill_(0, rows, 0.0)

    @torch.no_grad()
    def _zero_optimizer_state_cols(self, optimizer, param: nn.Parameter, cols: torch.Tensor):
        state = optimizer.state.get(param)
        if not state:
            return
        for value in state.values():
            if torch.is_tensor(value) and value.shape == param.shape and value.ndim >= 2:
                value.index_fill_(1, cols, 0.0)

    @torch.no_grad()
    def _zero_optimizer_state_indices(self, optimizer, param: nn.Parameter, inds: torch.Tensor):
        state = optimizer.state.get(param)
        if not state:
            return
        for value in state.values():
            if torch.is_tensor(value) and value.shape == param.shape and value.ndim >= 1:
                value.index_fill_(0, inds, 0.0)

    @torch.no_grad()
    def _reset_layer(
        self,
        optimizer,
        linear: nn.Linear,
        layer_norm,
        next_linear: nn.Linear,
        dormant_inds: torch.Tensor,
    ):
        if dormant_inds.numel() == 0:
            return

        dormant_inds = dormant_inds.to(device=linear.weight.device, dtype=torch.long)

        reset_weight = torch.empty_like(linear.weight)
        nn.init.kaiming_uniform_(reset_weight, a=np.sqrt(5))
        linear.weight[dormant_inds] = reset_weight[dormant_inds]
        self._zero_optimizer_state_rows(optimizer, linear.weight, dormant_inds)

        if linear.bias is not None:
            linear.bias[dormant_inds] = 0.0
            self._zero_optimizer_state_indices(optimizer, linear.bias, dormant_inds)

        if layer_norm is not None and layer_norm.elementwise_affine:
            layer_norm.weight[dormant_inds] = 1.0
            layer_norm.bias[dormant_inds] = 0.0
            self._zero_optimizer_state_indices(optimizer, layer_norm.weight, dormant_inds)
            self._zero_optimizer_state_indices(optimizer, layer_norm.bias, dormant_inds)

        next_linear.weight[:, dormant_inds] = 0.0
        self._zero_optimizer_state_cols(optimizer, next_linear.weight, dormant_inds)

    @torch.no_grad()
    def _apply_to_module(
        self,
        optimizer,
        module: nn.Module,
        output_head: nn.Linear,
        obs_batch: torch.Tensor,
        tag: str,
    ) -> dict:
        blocks = self._layers(module)
        if not blocks:
            return {}

        B = obs_batch.shape[0]
        if B < 2:
            return {}

        K = min(self.batch_size, B) if self.batch_size > 0 else B
        if K < B:
            idx = torch.randint(0, B, (K,), device=obs_batch.device)
            obs_sample = obs_batch[idx]
        else:
            obs_sample = obs_batch

        activations = {}
        hooks = []

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                activations[layer_idx] = output.detach()
            return hook

        for layer_idx, (_, _, relu) in enumerate(blocks):
            hooks.append(relu.register_forward_hook(make_hook(layer_idx)))

        was_training = module.training
        module.eval()
        try:
            _ = module(obs_sample)
        finally:
            for hook in hooks:
                hook.remove()
            if was_training:
                module.train()

        total_units = 0
        total_reset = 0

        for layer_idx, (linear, layer_norm, _) in enumerate(blocks):
            h = activations.get(layer_idx)
            if h is None:
                continue
            if h.ndim == 1:
                h = h.unsqueeze(0)
            elif h.ndim != 2:
                h = h.view(h.shape[0], -1)

            score = h.float().abs().mean(dim=0)
            norm_score = score / (score.mean() + self.eps)
            dormant_mask = norm_score <= self.tau
            dormant_inds = dormant_mask.nonzero(as_tuple=False).flatten()

            total_units += score.numel()
            total_reset += int(dormant_inds.numel())

            if dormant_inds.numel() == 0:
                continue

            if layer_idx + 1 < len(blocks):
                next_linear = blocks[layer_idx + 1][0]
            else:
                next_linear = output_head

            self._reset_layer(optimizer, linear, layer_norm, next_linear, dormant_inds)

        return {
            f"ReDo/{tag}/reset_fraction": 100.0 * total_reset / max(1, total_units),
            f"ReDo/{tag}/reset_count": float(total_reset),
        }

    @torch.no_grad()
    def maybe_apply(
        self,
        iteration: int,
        optimizer,
        actor_mean: nn.Module,
        actor_output_head: nn.Linear,
        critic: nn.Module,
        critic_output_head: nn.Linear,
        obs_batch: torch.Tensor,
    ) -> dict:
        if not self.use_redo:
            return {}
        if self.frequency <= 0:
            return {}
        if iteration % self.frequency != 0:
            return {}

        metrics = {}
        actor_metrics = self._apply_to_module(
            optimizer=optimizer,
            module=actor_mean,
            output_head=actor_output_head,
            obs_batch=obs_batch,
            tag="Actor",
        )
        if actor_metrics:
            metrics.update(actor_metrics)

        critic_metrics = self._apply_to_module(
            optimizer=optimizer,
            module=critic,
            output_head=critic_output_head,
            obs_batch=obs_batch,
            tag="Critic",
        )
        if critic_metrics:
            metrics.update(critic_metrics)

        return metrics


class GradientReDoResetter(ReDoResetter):
    """
    ReGraMa: gradient-magnitude based dormant reset, critic-only.

    判据换成权重梯度幅值；重置逻辑完全复用 ReDoResetter._reset_layer。
    必须在一次 critic backward 之后、且不跟随 optimizer.step() 的前提下调用 apply()。
    """

    def __init__(self, args):
        self.use_regrama = bool(getattr(args, "use_regrama", False))
        self.frequency = int(getattr(args, "regrama_frequency", 1))
        self.tau = float(getattr(args, "regrama_tau", 0.05))
        self.eps = 1e-9
        if self.use_regrama:
            print(f"[ReGraMa Enabled] frequency={self.frequency}, tau={self.tau} (critic-only)")

    def should_trigger(self, iteration: int) -> bool:
        return self.use_regrama and self.frequency > 0 and iteration % self.frequency == 0

    @torch.no_grad()
    def apply(self, optimizer, critic: nn.Module, critic_output_head: nn.Linear) -> dict:
        blocks = self._layers(critic)  # 复用父类，处理 Linear / Linear-LN-ReLU
        total, reset = 0, 0
        for layer_idx, (linear, layer_norm, _) in enumerate(blocks):
            if linear.weight.grad is None:
                continue
            grad_mag = linear.weight.grad.abs().mean(dim=1)  # [out_features]
            norm_score = grad_mag / (grad_mag.mean() + self.eps)
            dormant_inds = (norm_score <= self.tau).nonzero(as_tuple=False).flatten()
            total += grad_mag.numel()
            reset += int(dormant_inds.numel())
            if dormant_inds.numel() == 0:
                continue
            if layer_idx + 1 < len(blocks):
                next_linear = blocks[layer_idx + 1][0]
            else:
                next_linear = critic_output_head
            self._reset_layer(optimizer, linear, layer_norm, next_linear, dormant_inds)
        return {
            "ReGraMa/Critic/reset_fraction": 100.0 * reset / max(1, total),
            "ReGraMa/Critic/reset_count": float(reset),
        }


class AdamGnT(Optimizer):
    """
    Adam variant with per-parameter-element step counters for Generate-and-Test.

    Standard Adam keeps one scalar step per tensor. CBP needs to reset optimizer
    state for selected neuron rows/columns without disturbing the rest of the
    tensor, so step is stored with the same shape as each parameter.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _coerce_step(self, state, param):
        step = state.get("step")
        if torch.is_tensor(step) and step.shape == param.shape:
            return step

        if torch.is_tensor(step) and step.numel() == 1:
            step_value = float(step.item())
        elif isinstance(step, (int, float)):
            step_value = float(step)
        else:
            step_value = 0.0

        state["step"] = torch.full_like(param, step_value)
        return state["step"]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("AdamGnT does not support sparse gradients.")

                grad = param.grad
                state = self.state[param]

                if len(state) == 0:
                    state["step"] = torch.zeros_like(param)
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                else:
                    self._coerce_step(state, param)
                    if "exp_avg" not in state or state["exp_avg"].shape != param.shape:
                        state["exp_avg"] = torch.zeros_like(param)
                    if "exp_avg_sq" not in state or state["exp_avg_sq"].shape != param.shape:
                        state["exp_avg_sq"] = torch.zeros_like(param)

                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                step.add_(1)
                if weight_decay != 0.0:
                    grad = grad.add(param, alpha=weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)
                beta1_tensor = torch.full_like(step, beta1)
                beta2_tensor = torch.full_like(step, beta2)
                bias_correction1 = 1.0 - beta1_tensor.pow(step)
                bias_correction2 = 1.0 - beta2_tensor.pow(step)
                step_size = lr * bias_correction2.sqrt() / bias_correction1.clamp_min(1e-16)

                param.add_(exp_avg.div(denom).mul(step_size), alpha=-1.0)

        return loss


class GnT:
    """
    Generate-and-Test for a feed-forward Linear/ReLU MLP plus output head.

    linear_layers contains all hidden Linear layers followed by the output head.
    features contains post-activation hidden outputs in the same hidden order.
    """

    def __init__(
        self,
        linear_layers,
        opt,
        tag: str,
        hidden_activation: str = "relu",
        replacement_rate: float = 1e-4,
        decay_rate: float = 0.99,
        maturity_threshold: int = 10000,
        device: torch.device | str = "cpu",
    ):
        if len(linear_layers) < 2:
            raise ValueError("GnT requires at least one hidden Linear and one output Linear.")

        self.linear_layers = list(linear_layers)
        self.opt = opt
        self.tag = tag
        self.replacement_rate = float(replacement_rate)
        self.decay_rate = float(decay_rate)
        self.maturity_threshold = int(maturity_threshold)
        self.device = torch.device(device)
        self.num_hidden = len(self.linear_layers) - 1

        hidden_sizes = [layer.out_features for layer in self.linear_layers[:-1]]
        self.util = [torch.zeros(size, device=self.device) for size in hidden_sizes]
        self.mean_act = [torch.zeros(size, device=self.device) for size in hidden_sizes]
        self.ages = [torch.zeros(size, device=self.device) for size in hidden_sizes]
        self.accum = [0.0 for _ in hidden_sizes]

        if hidden_activation == "selu":
            self.bounds = [sqrt(3.0 / layer.in_features) for layer in self.linear_layers[:-1]]
        else:
            act = hidden_activation if hidden_activation not in ("swish", "elu") else "relu"
            gain = nn.init.calculate_gain(act)
            self.bounds = [gain * sqrt(3.0 / layer.in_features) for layer in self.linear_layers[:-1]]

    def _bias_correction(self, ages: torch.Tensor) -> torch.Tensor:
        decay = torch.full_like(ages, self.decay_rate)
        return (1.0 - decay.pow(ages)).clamp_min(1e-16)

    @torch.no_grad()
    def _update_utility(self, layer_idx: int, features: torch.Tensor) -> torch.Tensor:
        features = features.detach()
        if features.ndim == 1:
            features = features.unsqueeze(0)
        elif features.ndim != 2:
            features = features.view(features.shape[0], -1)

        expected = self.linear_layers[layer_idx].out_features
        if features.shape[1] != expected:
            raise ValueError(
                f"{self.tag} CBP activation width mismatch at hidden layer {layer_idx}: "
                f"got {features.shape[1]}, expected {expected}."
            )

        self.ages[layer_idx].add_(1)
        bc = self._bias_correction(self.ages[layer_idx])

        self.mean_act[layer_idx].mul_(self.decay_rate).add_(
            features.mean(dim=0), alpha=1.0 - self.decay_rate
        )

        next_linear = self.linear_layers[layer_idx + 1]
        out_weight_mag = next_linear.weight.detach().abs().mean(dim=0)
        new_util = out_weight_mag * features.abs().mean(dim=0)
        self.util[layer_idx].mul_(self.decay_rate).add_(new_util, alpha=1.0 - self.decay_rate)
        return self.util[layer_idx] / bc

    @torch.no_grad()
    def gen_and_test(self, features: list[torch.Tensor]) -> int:
        if len(features) != self.num_hidden:
            raise ValueError(
                f"{self.tag} CBP expected {self.num_hidden} activation tensors, got {len(features)}."
            )

        total_replaced = 0
        for layer_idx in range(self.num_hidden):
            if features[layer_idx] is None:
                raise ValueError(f"{self.tag} CBP missing activation for hidden layer {layer_idx}.")

            bc_util = self._update_utility(layer_idx, features[layer_idx])
            eligible = torch.where(self.ages[layer_idx] > self.maturity_threshold)[0]
            if eligible.numel() == 0:
                continue

            self.accum[layer_idx] += self.replacement_rate * eligible.numel()
            n_replace = min(int(self.accum[layer_idx]), eligible.numel())
            self.accum[layer_idx] -= n_replace
            if n_replace <= 0:
                continue

            idx_in_eligible = torch.topk(-bc_util[eligible], n_replace).indices
            to_replace = eligible[idx_in_eligible]
            self._replace(layer_idx, to_replace)
            total_replaced += int(to_replace.numel())

        return total_replaced

    @torch.no_grad()
    def _replace(self, layer_idx: int, to_replace: torch.Tensor):
        cur = self.linear_layers[layer_idx]
        nxt = self.linear_layers[layer_idx + 1]
        to_replace = to_replace.to(device=cur.weight.device, dtype=torch.long)
        n_replace = int(to_replace.numel())
        if n_replace == 0:
            return

        if nxt.bias is not None:
            mean_act_bc = self.mean_act[layer_idx][to_replace] / self._bias_correction(
                self.ages[layer_idx][to_replace]
            )
            nxt.bias.data.add_((nxt.weight.data[:, to_replace] * mean_act_bc).sum(dim=1))

        nxt.weight.data[:, to_replace] = 0.0
        cur.weight.data[to_replace, :] = torch.empty(
            n_replace,
            cur.in_features,
            device=cur.weight.device,
            dtype=cur.weight.dtype,
        ).uniform_(-self.bounds[layer_idx], self.bounds[layer_idx])
        if cur.bias is not None:
            cur.bias.data[to_replace] = 0.0

        self.ages[layer_idx][to_replace] = 0.0
        self.util[layer_idx][to_replace] = 0.0
        self.mean_act[layer_idx][to_replace] = 0.0
        self._reset_optimizer_state(layer_idx, to_replace)

    @torch.no_grad()
    def _reset_optimizer_state(self, layer_idx: int, to_replace: torch.Tensor):
        cur = self.linear_layers[layer_idx]
        nxt = self.linear_layers[layer_idx + 1]

        self._zero_state_rows(cur.weight, to_replace)
        if cur.bias is not None:
            self._zero_state_indices(cur.bias, to_replace)
        self._zero_state_cols(nxt.weight, to_replace)

    @torch.no_grad()
    def _zero_state_rows(self, param: nn.Parameter, rows: torch.Tensor):
        state = self.opt.state.get(param)
        if not state:
            return
        for key in ("exp_avg", "exp_avg_sq", "step"):
            value = state.get(key)
            if torch.is_tensor(value) and value.shape == param.shape and value.ndim >= 1:
                value.index_fill_(0, rows, 0.0)

    @torch.no_grad()
    def _zero_state_cols(self, param: nn.Parameter, cols: torch.Tensor):
        state = self.opt.state.get(param)
        if not state:
            return
        for key in ("exp_avg", "exp_avg_sq", "step"):
            value = state.get(key)
            if torch.is_tensor(value) and value.shape == param.shape and value.ndim >= 2:
                value.index_fill_(1, cols, 0.0)

    @torch.no_grad()
    def _zero_state_indices(self, param: nn.Parameter, inds: torch.Tensor):
        state = self.opt.state.get(param)
        if not state:
            return
        for key in ("exp_avg", "exp_avg_sq", "step"):
            value = state.get(key)
            if torch.is_tensor(value) and value.shape == param.shape and value.ndim >= 1:
                value.index_fill_(0, inds, 0.0)


class ContinualBackprop:
    """
    CBP manager for the current ActorMean/Critic modules.

    It supports the repository's default Linear -> ReLU backbones. LayerNorm is
    deliberately rejected for v1 to keep GnT indexing and reset semantics clear.
    """

    def __init__(self, args, actor_mean: nn.Module, critic: nn.Module, optimizer, device: torch.device):
        self.use_cbp = bool(getattr(args, "use_cbp", False))
        self.maturity_threshold = int(getattr(args, "cbp_maturity_threshold", 10000))
        self.hooks = []
        self.actor_activations = {}
        self.critic_activations = {}
        self.actor_gnt = None
        self.critic_gnt = None

        if not self.use_cbp:
            return

        if bool(getattr(args, "use_layernorm", False)):
            raise ValueError("CBP v1 supports only --use_layernorm=false.")
        if not isinstance(optimizer, AdamGnT):
            raise TypeError("CBP requires AdamGnT optimizer.")

        actor_linear_layers, actor_relu_layers = self._collect_layers(
            actor_mean, actor_mean.linear_mu, "Actor"
        )
        critic_linear_layers, critic_relu_layers = self._collect_layers(
            critic, critic.linear_value, "Critic"
        )

        self.actor_gnt = GnT(
            actor_linear_layers,
            opt=optimizer,
            tag="Actor",
            replacement_rate=1e-4,
            decay_rate=0.99,
            maturity_threshold=self.maturity_threshold,
            device=device,
        )
        self.critic_gnt = GnT(
            critic_linear_layers,
            opt=optimizer,
            tag="Critic",
            replacement_rate=1e-4,
            decay_rate=0.99,
            maturity_threshold=self.maturity_threshold,
            device=device,
        )

        self._register_hooks(actor_relu_layers, self.actor_activations)
        self._register_hooks(critic_relu_layers, self.critic_activations)
        print(f"[CBP Enabled] replacement_rate=1e-4, maturity_threshold={self.maturity_threshold}, decay_rate=0.99")

    def _collect_layers(self, module: nn.Module, output_head: nn.Linear, tag: str):
        if not hasattr(module, "backbone") or not hasattr(module.backbone, "net"):
            raise AttributeError(f"{tag} module must expose backbone.net for CBP.")

        layers = list(module.backbone.net)
        linear_layers = []
        relu_layers = []
        i = 0
        while i < len(layers):
            linear = layers[i]
            if not isinstance(linear, nn.Linear):
                raise ValueError(f"{tag} CBP expects each hidden block to start with Linear.")
            i += 1

            if i < len(layers) and isinstance(layers[i], nn.LayerNorm):
                raise ValueError("CBP v1 supports only Linear -> ReLU backbones, not LayerNorm.")

            if i >= len(layers) or not isinstance(layers[i], nn.ReLU):
                raise ValueError(f"{tag} CBP expects hidden blocks to be Linear -> ReLU.")
            relu = layers[i]
            i += 1

            linear_layers.append(linear)
            relu_layers.append(relu)

        linear_layers.append(output_head)
        return linear_layers, relu_layers

    def _register_hooks(self, relu_layers, activation_store: dict):
        for layer_idx, relu in enumerate(relu_layers):
            def hook(_module, _inputs, output, idx=layer_idx, store=activation_store):
                store[idx] = output.detach()
            self.hooks.append(relu.register_forward_hook(hook))

    def after_optimizer_step(self) -> dict[str, float]:
        if not self.use_cbp:
            return {}

        actor_features = [self.actor_activations.get(i) for i in range(self.actor_gnt.num_hidden)]
        critic_features = [self.critic_activations.get(i) for i in range(self.critic_gnt.num_hidden)]
        actor_replaced = self.actor_gnt.gen_and_test(actor_features)
        critic_replaced = self.critic_gnt.gen_and_test(critic_features)

        return {
            "CBP/Actor/replaced": float(actor_replaced),
            "CBP/Critic/replaced": float(critic_replaced),
            "CBP/total_replaced": float(actor_replaced + critic_replaced),
        }

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
