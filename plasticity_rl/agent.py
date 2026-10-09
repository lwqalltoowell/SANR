# agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from gymnasium import spaces

from .network import Critic, ActorMean, layer_init
from .monitors import (
    DormantNeuronMonitor,
    FeatureRankMonitor,
    GradientDormantMonitor,
    GroupNormMonitor,
    PolicyStdMonitor,
)
from .plasticity import (
    PFORegularizer,
    RegenerativeRegularizer,
    SoftShrinkPerturber,
    SoftNormAnchoring,
    ReDoResetter,
    GradientReDoResetter,
    AdamGnT,
    ContinualBackprop,
)


class PPOAgent(nn.Module):
    """
    PPO agent that owns the networks, optimizer, rollout storage and update logic.

    Monitors are intentionally kept external (monitors.py) to keep PPO logic clean.
    """

    def __init__(self, observation_space, action_space, args, device: torch.device):
        super().__init__()
        assert isinstance(action_space, spaces.Box), "Only continuous action spaces are supported."

        self.args = args
        self.device = device

        # ----- infer dimensions -----
        obs_shape = observation_space.shape          # e.g. (obs_dim,)
        act_shape = action_space.shape               # e.g. (act_dim,)
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.obs_dim = int(np.prod(obs_shape))
        self.action_dim = int(np.prod(act_shape))

        # ----- build networks -----
        hidden_size = args.hidden_size
        num_layer = args.num_layer

        self.critic = Critic(self.obs_dim, hidden_size=hidden_size, num_layer=num_layer, use_layernorm=args.use_layernorm, use_affine=args.use_affine)
        self.actor_mean = ActorMean(self.obs_dim, self.action_dim,
                                    hidden_size=hidden_size, num_layer=num_layer, use_layernorm=args.use_layernorm, use_affine=args.use_affine)

        # state-independent log std as in CleanRL
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))

        # move all modules/parameters to device
        self.to(self.device)

        #----- optimizer -----
        weight_decay = float(getattr(args, "weight_decay", 0))
        if bool(getattr(args, "use_cbp", False)):
            self.optimizer = AdamGnT(
                self.parameters(),
                lr=args.lr,
                eps=1e-5,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = optim.Adam(
                self.parameters(),
                lr=args.lr,
                eps=1e-5,
                weight_decay=weight_decay,
            )


        # ----- rollout storage -----
        num_steps = args.num_steps
        num_envs = args.num_envs

        # [T, N, *shape] – same layout as in CleanRL
        self.obs = torch.zeros((num_steps, num_envs) + obs_shape, device=self.device)
        self.actions = torch.zeros((num_steps, num_envs) + act_shape, device=self.device)
        self.logprobs = torch.zeros(num_steps, num_envs, device=self.device)
        self.rewards = torch.zeros(num_steps, num_envs, device=self.device)
        self.dones = torch.zeros(num_steps, num_envs, device=self.device)
        self.values = torch.zeros(num_steps, num_envs, device=self.device)
        self.actor_preacts = torch.zeros(
            (num_steps, num_envs, self.actor_mean.backbone.out_dim),
            device=self.device,
        )

        # will be filled after GAE
        self.advantages = torch.zeros(num_steps, num_envs, device=self.device)
        self.returns = torch.zeros(num_steps, num_envs, device=self.device)

        # ----- monitors -----
        # Rank monitors (use rollout b_obs as input)
        rank_every = getattr(args, "rank_every_iters", 1)
        rank_bs = getattr(args, "rank_batch_size", 1024)
        rank_thr = getattr(args, "rank_var_threshold", 0.99)

        self.actor_rank_monitor = FeatureRankMonitor(
            module=self.actor_mean,
            tag="Actor",
            every_iters=rank_every,
            sample_size=rank_bs,
            var_threshold=rank_thr,
        )
        self.critic_rank_monitor = FeatureRankMonitor(
            module=self.critic,
            tag="Critic",
            every_iters=rank_every,
            sample_size=rank_bs,
            var_threshold=rank_thr,
        )

        # Norm monitors (param/grad)
        norm_every = getattr(args, "norm_every_iters", 1)
        self.actor_norm_monitor = GroupNormMonitor(
            module=self.actor_mean,
            tag="Actor",
            every_iters=norm_every,
            groups={
                # backbone 的 Linear（SNA 作用的对象）
                "backbone_linear": ("backbone", nn.Linear),
            },
        )

        self.critic_norm_monitor = GroupNormMonitor(
            module=self.critic,
            tag="Critic",
            every_iters=norm_every,
            groups={
                "backbone_linear": ("backbone", nn.Linear),
            },
        )

        # Policy std monitor (state-independent std)
        std_every = getattr(args, "std_every_iters", 1)
        self.policy_std_monitor = PolicyStdMonitor(
            actor_logstd=self.actor_logstd,
            tag="Policy",
            every_iters=std_every,
        )



        # Dormant neuron monitors (ReDo-style)
        dormant_every = getattr(args, "dormant_every_iters", 1)
        dormant_bs = getattr(args, "dormant_batch_size", 1024)
        dormant_tau = getattr(args, "dormant_tau", 0.1)
        redo_tau = getattr(args, "redo_tau", 0.025)
        dormant_eps = getattr(args, "dormant_epsilon", 1e-8)

        self.actor_dormant_monitor = DormantNeuronMonitor(
            module=self.actor_mean,
            tag="Actor",
            every_iters=dormant_every,
            sample_size=dormant_bs,
            tau=dormant_tau,
            aux_tau=redo_tau,
            eps=dormant_eps,
            log_per_layer=False,
            log_zero_ratio=False,
        )

        self.critic_dormant_monitor = DormantNeuronMonitor(
            module=self.critic,
            tag="Critic",
            every_iters=dormant_every,
            sample_size=dormant_bs,
            tau=dormant_tau,
            aux_tau=redo_tau,
            eps=dormant_eps,
            log_per_layer=False,
            log_zero_ratio=False,
        )



        # Gradient-based dormant monitor (GraMa-style, critic-only)
        regrama_tau = getattr(args, "regrama_tau", 0.05)
        regrama_frequency = getattr(args, "regrama_frequency", 1)
        self.critic_grad_dormant_monitor = GradientDormantMonitor(
            module=self.critic,
            tag="Critic",
            every_iters=regrama_frequency,
            tau=regrama_tau,
            eps=dormant_eps,
        )

        # For clean looping
        self._monitors_with_obs = [
            self.actor_rank_monitor,
            self.critic_rank_monitor,
            self.actor_dormant_monitor,
            self.critic_dormant_monitor,
        ]
        self._monitors_no_obs = [
            self.actor_norm_monitor,
            self.critic_norm_monitor,
            self.policy_std_monitor,
        ]

        # ----- plasticity methods -----
        self.pfo = PFORegularizer(args)
        self.regen = RegenerativeRegularizer(args)
        self.snp = SoftShrinkPerturber(args, self.actor_mean, self.critic)
        self.sna = SoftNormAnchoring(
            args,
            modules=[self.actor_mean.backbone, self.critic.backbone],
        )
        self.redo = ReDoResetter(args)
        self.regrama = GradientReDoResetter(args)
        self.cbp = ContinualBackprop(args, self.actor_mean, self.critic, self.optimizer, self.device)

    @torch.no_grad()
    def capture_regen_reference(self):
        self.regen.capture_reference(self.named_parameters())

    @torch.no_grad()
    def reset_actor_head(self):
        """
        Reinitialize the actor mean head (linear_mu) and clear only its Adam state.

        This is used by regime-switch experiments to isolate whether plasticity loss
        mainly lives in the actor output head rather than in the backbone or log_std.
        """
        layer_init(self.actor_mean.linear_mu, std=0.01, bias_const=0.0)

        for param in self.actor_mean.linear_mu.parameters():
            if param in self.optimizer.state:
                self.optimizer.state[param].clear()

    # ------------------------------------------------------------------
    # basic helpers
    # ------------------------------------------------------------------
    def _flatten_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Flatten obs to [batch, obs_dim]. Works for [batch, *obs_shape].
        """
        return obs.view(obs.shape[0], -1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute V(s).

        obs: [batch, *obs_shape]
        returns: [batch] (squeezed)
        """
        obs_flat = self._flatten_obs(obs)
        value = self.critic(obs_flat)
        return value.squeeze(-1)


    def get_action_and_value(self, obs: torch.Tensor, action=None, return_preact: bool = False):
        """
        Return (action, logprob, entropy, value) for given obs.
        """
        obs_flat = self._flatten_obs(obs)
        if return_preact:
            action_mean, actor_preact = self.actor_mean.forward_with_preact(obs_flat)
        else:
            action_mean = self.actor_mean(obs_flat)
            actor_preact = None

        action_logstd = self.actor_logstd.expand_as(action_mean)

        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        logprob = dist.log_prob(action).sum(-1)   # [batch]
        entropy = dist.entropy().sum(-1)          # [batch]
        value = self.critic(obs_flat).squeeze(-1) # [batch]

        if return_preact:
            return action, logprob, entropy, value, actor_preact
        return action, logprob, entropy, value


    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False, return_preact: bool = False):
        """
        用于 Rollout 或推理：采样动作并返回 (action, logprob, value)。
        """
        if deterministic:
            obs_flat = self._flatten_obs(obs)
            if return_preact:
                action, actor_preact = self.actor_mean.forward_with_preact(obs_flat)
            else:
                action = self.actor_mean(obs_flat)
                actor_preact = None
            value = self.critic(obs_flat).squeeze(-1)

            logprob = torch.zeros(action.shape[0], device=self.device)
            if return_preact:
                return action, logprob, value, actor_preact
            return action, logprob, value
        else:
            out = self.get_action_and_value(obs, action=None, return_preact=return_preact)
            if return_preact:
                action, logprob, _, value, actor_preact = out
                return action, logprob, value, actor_preact
            action, logprob, _, value = out
            return action, logprob, value

    def store_step(
        self,
        step: int,
        obs: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        actor_preacts: torch.Tensor = None,
    ):
        """
        Store one environment step into the rollout buffer.
        """
        self.obs[step].copy_(obs)
        self.actions[step].copy_(actions)
        self.logprobs[step].copy_(logprobs)
        self.rewards[step].copy_(rewards)
        self.dones[step].copy_(dones)
        self.values[step].copy_(values)
        if actor_preacts is not None:
            self.actor_preacts[step].copy_(actor_preacts)

    # ------------------------------------------------------------------
    # learning rate schedule
    # ------------------------------------------------------------------
    def update_learning_rate(self, iteration: int, num_iterations: int) -> float:
        """
        Linearly anneal the learning rate if args.anneal_lr is True.
        Returns the current learning rate.
        """
        if getattr(self.args, "anneal_lr", False):
            frac = 1.0 - (iteration - 1.0) / float(num_iterations)
            lr_now = frac * self.args.lr
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr_now
        else:
            lr_now = self.optimizer.param_groups[0]["lr"]
        return lr_now

    # ------------------------------------------------------------------
    # GAE & returns
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_returns_and_advantages(self, next_obs: torch.Tensor, next_done: torch.Tensor):
        """
        Compute GAE(λ) advantages and returns, in-place.
        """
        num_steps, num_envs = self.rewards.shape

        next_value = self.get_value(next_obs)  # [N]
        lastgaelam = torch.zeros(num_envs, device=self.device, dtype=torch.float32)

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done      # [N]
                nextvalues = next_value                # [N]
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]  # [N]
                nextvalues = self.values[t + 1]            # [N]

            delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
            lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
            self.advantages[t] = lastgaelam

        self.returns = self.advantages + self.values

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def update(self, iteration: int, num_iterations: int) -> dict:
        """
        Run one PPO update over the collected rollout.
        """
        # 1) maybe anneal LR
        lr_now = self.update_learning_rate(iteration, num_iterations)

        # SNA follows the same decay fraction as the learning rate.
        lr_frac = lr_now / self.args.lr

        num_steps, num_envs = self.rewards.shape
        batch_size = self.args.batch_size
        minibatch_size = self.args.minibatch_size

        # flatten the batch
        b_obs = self.obs.reshape((num_steps * num_envs, -1))          # [B,O]
        b_actions = self.actions.reshape((num_steps * num_envs, -1))  # [B,A]
        b_logprobs = self.logprobs.reshape(-1)                        # [B]
        b_advantages = self.advantages.reshape(-1)                    # [B]
        b_returns = self.returns.reshape(-1)                          # [B]
        b_values = self.values.reshape(-1)                            # [B]
        b_actor_preacts = self.actor_preacts.reshape((num_steps * num_envs, -1))

        # indices for shuffling
        b_inds = np.arange(batch_size)

        # stats to log from last minibatch
        pg_loss = v_loss = entropy_loss = None
        pfo_loss = pfo_penalty = None
        regen_loss = regen_penalty = None
        redo_metrics = {}
        cbp_metrics = {}

        for epoch in range(self.args.update_epochs):
            np.random.shuffle(b_inds)
            epoch_kl_sum = None
            epoch_kl_samples = 0
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, actor_preact_new = self.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds], return_preact=True
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                if self.args.target_kl is not None:
                    with torch.no_grad():
                        minibatch_kl = ((ratio - 1) - logratio).mean()
                        minibatch_samples = len(mb_inds)
                        weighted_kl = minibatch_kl * minibatch_samples
                        epoch_kl_sum = (
                            weighted_kl
                            if epoch_kl_sum is None
                            else epoch_kl_sum + weighted_kl
                        )
                        epoch_kl_samples += minibatch_samples

                mb_advantages = b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - self.args.clip_coef, 1.0 + self.args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.args.ent_coef * entropy_loss + self.args.vf_coef * v_loss
                ppo_loss = float(loss.item()) if loss is not None else 0.0

                pfo_loss, pfo_metrics = self.pfo.loss(
                    actor_preact_new,
                    b_actor_preacts[mb_inds],
                )
                pfo_penalty = pfo_metrics["pfo_penalty"]

                regen_loss, regen_metrics = self.regen.loss(self.named_parameters())
                regen_penalty = regen_metrics["regen_penalty"]

                loss = loss + pfo_loss + regen_loss

                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)


                self.optimizer.step()

                step_cbp_metrics = self.cbp.after_optimizer_step()
                if step_cbp_metrics:
                    cbp_metrics = step_cbp_metrics
                self.snp.after_optimizer_step()
                self.sna.after_optimizer_step(lr_frac)


            # early stop if KL too large
            if epoch_kl_sum is not None and epoch_kl_samples > 0:
                epoch_approx_kl = epoch_kl_sum / epoch_kl_samples
                if epoch_approx_kl > self.args.target_kl:
                    break

        redo_metrics = self.redo.maybe_apply(
            iteration=iteration,
            optimizer=self.optimizer,
            actor_mean=self.actor_mean,
            actor_output_head=self.actor_mean.linear_mu,
            critic=self.critic,
            critic_output_head=self.critic.linear_value,
            obs_batch=b_obs,
        )

        # Shared diagnostic backward: critic value loss on full b_obs batch.
        # Supplies gradients for ReGraMa reset and GradientDormantMonitor.
        # Resets are applied before zero_grad so weights are not overwritten by step.
        regrama_metrics = {}
        grad_dormant_metrics = {}
        _grad_every = self.critic_grad_dormant_monitor.every_iters
        need_diag_backward = (
            self.regrama.should_trigger(iteration) or
            (_grad_every > 0 and iteration % _grad_every == 0)
        )
        if need_diag_backward:
            self.optimizer.zero_grad()
            newvalue_full = self.get_value(b_obs)
            v_for_reset = ((newvalue_full - b_returns) ** 2).mean()
            v_for_reset.backward()
            if self.regrama.should_trigger(iteration):
                regrama_metrics = self.regrama.apply(
                    optimizer=self.optimizer,
                    critic=self.critic,
                    critic_output_head=self.critic.linear_value,
                )
            out = self.critic_grad_dormant_monitor.log_from_grads(iteration)
            if out:
                grad_dormant_metrics = out
            self.optimizer.zero_grad()

        metrics = {
            # 暂时不记录一些非关键变量，精简日志输出，避免过度冗杂
            #"Train/lr": float(lr_now),
            # "Train/Loss/value_loss": float(v_loss.item() if v_loss is not None else 0.0),
            # "Train/Loss/policy_loss": float(pg_loss.item() if pg_loss is not None else 0.0),
            "Train/Loss/ppo_loss": ppo_loss,
            "Train/Loss/pfo_loss": float(pfo_loss.item() if pfo_loss is not None else 0.0),
            "Train/Loss/pfo_penalty": float(pfo_penalty.item() if pfo_penalty is not None else 0.0),
            "Train/Loss/regen_loss": float(regen_loss.item() if regen_loss is not None else 0.0),
            "Train/Loss/regen_penalty": float(regen_penalty.item() if regen_penalty is not None else 0.0),
            "Actor/entropy": float(entropy_loss.item() if entropy_loss is not None else 0.0),
        }
        metrics.update(redo_metrics)
        metrics.update(regrama_metrics)
        metrics.update(cbp_metrics)
        metrics.update(grad_dormant_metrics)

        # ------------------------------------------------------------------
        # monitors
        # ------------------------------------------------------------------
        for mon in self._monitors_with_obs:
            out = mon.maybe_log(iteration, b_obs)
            if out:
                metrics.update(out)

        for mon in self._monitors_no_obs:
            out = mon.maybe_log(iteration)
            if out:
                metrics.update(out)

        return metrics




    # ------------------------------------------------------------------
    # Save & Load (Enhanced for Experiments)
    # ------------------------------------------------------------------
    def save_model(self, path: str):
        """
        保存模型的所有状态。
        """
        checkpoint = {
            "actor_mean_state_dict": self.actor_mean.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_logstd": self.actor_logstd,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "regen_reference_state": self.regen.state_dict(),
            "args": self.args,
        }
        torch.save(checkpoint, path)
        print(f"==> Model and optimizer state saved to {path}")

    def load_model(self, path: str, load_optimizer: bool = True):
        """
        加载模型参数。

        Args:
            path: 模型文件路径。
            load_optimizer: 是否加载优化器状态。
                            True:  热启动 (Warm Start)，保留历史动量估计。
                            False: 冷启动 (Cold Start)，仅加载权重，优化器重新初始化。
        """
        # 使用 map_location 确保跨设备（GPU/CPU）加载的兼容性
        checkpoint = torch.load(path, map_location=self.device)

        # 1. 始终加载权重参数 (Weights)
        self.actor_mean.load_state_dict(checkpoint["actor_mean_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])

        # 加载独立的 actor_logstd
        with torch.no_grad():
            self.actor_logstd.copy_(checkpoint["actor_logstd"])

        load_status = "Weights only"

        if "regen_reference_state" in checkpoint:
            self.regen.load_reference(checkpoint["regen_reference_state"])

        # 2. 选择性加载优化器状态 (Optimizer State)
        if load_optimizer:
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                load_status = "Weights + Optimizer (Warm Start)"
            else:
                print("Warning: No optimizer state found in checkpoint. Loading weights only.")
        else:
            # 如果不加载优化器，建议根据当前的 args 重新初始化一次优化器，以防万一
            # 注意：在 __init__ 中已经初始化过了，这里只需保持原样即可
            load_status = "Weights only (Cold Start/Inference)"

        print(f"==> Successfully loaded: [{load_status}] from {path}")
