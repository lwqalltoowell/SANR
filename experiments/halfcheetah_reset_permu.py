# halfcheetah_reset_permu.py
import os
import argparse
import time
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import wandb

from plasticity_rl.agent import PPOAgent
from plasticity_rl.wrapper import SaveRawReward, HalfCheetahCyclicBlockPermutationWrapper


def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in ("true", "1", "yes", "y", "t"):
            return True
        if v.lower() in ("false", "0", "no", "n", "f"):
            return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_parser():
    parser = argparse.ArgumentParser(description="PPO Plasticity Test: HalfCheetah cyclic block permutation (HalfCheetah-v5)")

    parser.add_argument("--total_steps", type=int, default=6_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--anneal_lr", type=str2bool, default=False)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)

        # ---SNA & regularization settings---
    parser.add_argument("--sna_lambda", type=float, default=0, help="SNA lambda (baseline: 0.0, higher: more SNA effect)")

    # ---LayerNorm and weight decay settings---
    parser.add_argument("--use_layernorm", type=str2bool, default=False, help="Whether to use LayerNorm in the model.")
    parser.add_argument("--use_affine", type=str2bool, default=True,
                        help="Whether to use affine transform in the model.")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="Adam L2 weight decay for Linear weights only.")



    #---ReDo settings---
    parser.add_argument("--use_redo", type=str2bool, default=False, help="Whether to periodically reset dormant backbone units with ReDo.")
    parser.add_argument("--redo_frequency", type=int, default=1, help="Rollout iterations between ReDo applications.")
    parser.add_argument("--redo_tau", type=float, default=0.025, help="Layer-normalized dormant-unit threshold for ReDo.")

    #---ReGraMa settings---
    parser.add_argument("--use_regrama", type=str2bool, default=False, help="Whether to periodically reset critic units by gradient magnitude (ReGraMa).")
    parser.add_argument("--regrama_tau", type=float, default=0.025, help="Layer-normalized gradient-magnitude dormant threshold for ReGraMa.")
    parser.add_argument("--regrama_frequency", type=int, default=1, help="Rollout iterations between ReGraMa applications.")


    #----Regenerative Regularization settings---
    parser.add_argument("--regen_coef", type=float, default=0.0, help="Regenerative Regularization coefficient")

    #----PFO settings---
    parser.add_argument("--pfo_coef", type=float, default=0.0, help="PFO-last coefficient on actor last hidden pre-activations.")


    #----Soft Shrink-and-Perturb settings-----
    parser.add_argument("--snp_beta", type=float, default=0.0, help="Soft Shrink-and-Perturb beta. 0 disables SNP.")



    #----Continual Backprop settings---
    parser.add_argument("--use_cbp", type=str2bool, default=False, help="Whether to enable Continual Backprop (CBP).")
    parser.add_argument("--cbp_maturity_threshold", type=int, default=10000, help="Maturity threshold for CBP.")

    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=2048)
    parser.add_argument("--num_minibatches", type=int, default=16)
    parser.add_argument("--update_epochs", type=int, default=10)

    parser.add_argument("--norm_adv", type=str2bool, default=True)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--clip_vloss", type=str2bool, default=True)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=None)

    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layer", type=int, default=2)

    #-----Switch settings-----
    parser.add_argument("--switch_interval", type=int, default=2_000_000)

    parser.add_argument(
        "--inject_exploration_on_switch",
        type=str2bool,
        default=True,
        help="Whether to reset actor_logstd (and clear its optimizer state) after each observation permutation switch.",
    )
    parser.add_argument(
        "--reset_actor_head_on_switch",
        type=str2bool,
        default=False,
        help="Whether to reinitialize actor mean head linear_mu (and clear only its optimizer state) after each observation permutation switch.",
    )
    parser.add_argument(
        "--reset_agent_on_switch",
        type=str2bool,
        default=False,
        help="Whether to fully reinitialize the PPO agent after each observation permutation switch.",
    )

    parser.add_argument("--env_name", type=str, default="HalfCheetah-v5")
    parser.add_argument("--wandb_project_name", type=str, default="SANR-HalfCheetah")
    parser.add_argument("--group_name", type=str, default="Layernorm+L2")
    parser.add_argument("--seed", type=int, default=0, help="控制网络初始化和动作采样的种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch_deterministic", type=str2bool, default=False)
    parser.add_argument("--use_async_vecenv", type=str2bool, default=True)
    parser.add_argument("--shared_memory", type=str2bool, default=True)
    parser.add_argument("--track", type=str2bool, default=False)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--model_save_dir", type=str, default="./model")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--torch_threads", type=int, default=1)

    return parser


def parse_args():
    args = build_parser().parse_args()

    args.batch_size = args.num_envs * args.num_steps
    assert args.batch_size % args.num_minibatches == 0, "batch_size error"
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_steps // args.batch_size

    if args.experiment_name is None:
        args.experiment_name = (
            f"{args.env_name}_"
            f"seed{args.seed}_"
            f"h{args.hidden_size}"
        )

    if args.sna_lambda > 0.0:
        args.experiment_name += f"_SNA{args.sna_lambda}"

    if args.use_redo:
        args.experiment_name += f"_ReDo{args.redo_frequency}_{args.redo_tau}"

    if args.weight_decay > 0.0:
        args.experiment_name += f"_WD{args.weight_decay}"

    if args.regen_coef > 0.0:
        args.experiment_name += f"_Regen{args.regen_coef}"

    if args.pfo_coef > 0.0:
        args.experiment_name += f"_PFO{args.pfo_coef}"

    if args.snp_beta > 0.0:
        args.experiment_name += f"_SNP{args.snp_beta}"


    if args.use_cbp:
        args.experiment_name += f"_CBP{args.cbp_maturity_threshold}"

    if not args.inject_exploration_on_switch:
        args.experiment_name += f"_Nostd"

    if args.use_regrama:
        args.experiment_name += f"_ReGraMa{args.regrama_tau}_F{args.regrama_frequency}"

    return args


def set_seed(seed: int, torch_deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def ensure_dir(path: str):
    if path and (not os.path.exists(path)):
        os.makedirs(path, exist_ok=True)


def make_env(env_name: str, idx: int, seed: int):
    def thunk():
        env = gym.make(env_name)
        env = gym.wrappers.FlattenObservation(env)
        env = HalfCheetahCyclicBlockPermutationWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.DtypeObservation(env, dtype=np.float32)
        env = SaveRawReward(env)

        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


def make_vec_envs(args):
    env_fns = [make_env(args.env_name, i, args.seed) for i in range(args.num_envs)]

    if args.num_envs == 1:
        envs = gym.vector.SyncVectorEnv(env_fns)
    else:
        if args.use_async_vecenv:
            envs = gym.vector.AsyncVectorEnv(env_fns, shared_memory=args.shared_memory)
        else:
            envs = gym.vector.SyncVectorEnv(env_fns)

    envs = gym.wrappers.vector.NormalizeObservation(envs, epsilon=1e-8)
    envs = gym.wrappers.vector.NormalizeReward(envs, gamma=args.gamma)
    return envs


def reset_obs_norm_stats(vec_env):
    current = vec_env
    while current is not None:
        if isinstance(current, gym.wrappers.vector.NormalizeObservation):
            current.obs_rms.mean.fill(0.0)
            current.obs_rms.var.fill(1.0)
            current.obs_rms.count = 1e-4
            print("[Norm Reset] NormalizeObservation.obs_rms reset (mean=0,var=1,count=1e-4).")
            return True
        current = getattr(current, "env", None)

    print("[Norm Reset] WARNING: NormalizeObservation wrapper not found; no reset performed.")
    return False


def reset_reward_norm_stats(vec_env):
    current = vec_env
    while current is not None:
        if isinstance(current, gym.wrappers.vector.NormalizeReward):
            current.return_rms.mean.fill(0.0)
            current.return_rms.var.fill(1.0)
            current.return_rms.count = 1e-4
            current.accumulated_reward.fill(0.0)
            print("[Norm Reset] NormalizeReward.return_rms reset (mean=0,var=1,count=1e-4).")
            return True
        current = getattr(current, "env", None)

    print("[Norm Reset] WARNING: NormalizeReward wrapper not found; no reset performed.")
    return False


def train(args):
    torch.set_num_threads(args.torch_threads)
    set_seed(args.seed, args.torch_deterministic)
    device = torch.device(args.device)
    ensure_dir(args.model_save_dir)

    if args.track:
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=args.experiment_name,
            group=args.group_name,
            config=vars(args),
        )
    else:
        run = None

    envs = make_vec_envs(args)
    agent = PPOAgent(envs.single_observation_space, envs.single_action_space, args, device)
    agent.capture_regen_reference()

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.float32)

    ep_raw_returns = np.zeros(args.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int32)
    ep_return_buffer = deque(maxlen=40)

    current_perm_seed = 0
    next_switch_step = args.switch_interval


    pending_switch = False
    pending_perm_seed = current_perm_seed
    done_seen = np.zeros(args.num_envs, dtype=np.bool_)
    regime_idx = 0

    print(f"\n[Init] Initial cyclic permutation schedule index: {current_perm_seed} ...")
    envs.unwrapped.call("set_permutation", perm_seed=current_perm_seed)

    for iteration in range(1, args.num_iterations + 1):
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_tensor = next_obs
            dones_tensor = next_done

            with torch.no_grad():
                if args.pfo_coef > 0.0:
                    actions, logprobs, values, actor_preacts = agent.act(
                        obs_tensor,
                        deterministic=False,
                        return_preact=True,
                    )
                else:
                    actions, logprobs, values = agent.act(obs_tensor, deterministic=False)
                    actor_preacts = None

            actions_np = actions.cpu().numpy()
            next_obs_np, rewards_np, terminations, truncations, infos = envs.step(actions_np)

            rewards_tensor = torch.as_tensor(rewards_np, device=device, dtype=torch.float32)
            done_np = np.logical_or(terminations, truncations)
            next_done = torch.as_tensor(done_np, device=device, dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)

            agent.store_step(
                step,
                obs_tensor,
                actions,
                logprobs,
                rewards_tensor,
                dones_tensor,
                values,
                actor_preacts=actor_preacts,
            )

            switched_this_step = False

            if (not pending_switch) and (global_step >= next_switch_step):
                pending_switch = True
                pending_perm_seed = current_perm_seed + 1
                done_seen[:] = False
                print(
                    f"\n[Switch Armed] step={global_step} | "
                    f"Will switch to permutation schedule index: {pending_perm_seed}"
                )

            if pending_switch:
                done_seen |= done_np
                if done_seen.all():
                    print("\n" + "=" * 70)
                    print("[Plasticity Test] Switching HalfCheetah cyclic block observation schedule")
                    print(
                        f"Global Step: {global_step} | "
                        f"Schedule Index: {current_perm_seed} -> {pending_perm_seed}"
                    )
                    print("=" * 70 + "\n")

                    envs.unwrapped.call("set_permutation", perm_seed=pending_perm_seed)

                    did_obs_reset = reset_obs_norm_stats(envs)
                    did_reward_reset = reset_reward_norm_stats(envs)
                    did_entropy_reset = 0.0
                    did_actor_head_reset = 0.0
                    did_agent_reset = 0.0

                    if args.reset_agent_on_switch:
                        agent_reset_seed = args.seed + regime_idx + 1
                        torch.manual_seed(agent_reset_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(agent_reset_seed)
                        agent = PPOAgent(envs.single_observation_space, envs.single_action_space, args, device)
                        agent.capture_regen_reference()
                        did_agent_reset = 1.0
                        print("[Agent Reset] Full agent reinitialized. Supersedes sub-resets.")
                    else:
                        if args.inject_exploration_on_switch:
                            with torch.no_grad():
                                agent.actor_logstd.fill_(0.0)
                                if agent.actor_logstd in agent.optimizer.state:
                                    agent.optimizer.state[agent.actor_logstd].clear()
                            did_entropy_reset = 1.0
                            print("[Exploration Reset] Actor log_std reset.")
                        else:
                            print("[Exploration Reset] Skipped.")

                        if args.reset_actor_head_on_switch:
                            agent.reset_actor_head()
                            did_actor_head_reset = 1.0
                            print("[Head Reset] Actor head reinitialized.")
                        else:
                            print("[Head Reset] Skipped.")

                    current_perm_seed = pending_perm_seed
                    pending_switch = False
                    done_seen[:] = False
                    next_switch_step += args.switch_interval
                    regime_idx += 1
                    switched_this_step = True

            if isinstance(infos, dict):
                raw_rewards = infos.get("raw_reward", rewards_np)
            else:
                raw_rewards = np.array([info.get("raw_reward", np.nan) for info in infos])

            ep_raw_returns += raw_rewards
            ep_lengths += 1

            for i, d in enumerate(done_np):
                if d:
                    ep_return = float(ep_raw_returns[i])

                    if args.track:
                        wandb.log({"Reward/episodic_return": ep_return}, step=global_step)

                    ep_return_buffer.append(ep_return)
                    if len(ep_return_buffer) >= 10 and args.track:
                        wandb.log({"Reward/avg_return": float(np.mean(ep_return_buffer))}, step=global_step)

                    ep_raw_returns[i] = 0.0
                    ep_lengths[i] = 0

            if switched_this_step:
                ep_return_buffer.clear()

        agent.compute_returns_and_advantages(next_obs, next_done)
        metrics = agent.update(iteration, args.num_iterations)
        metrics["Speed/SPS"] = int(global_step / (time.time() - start_time))

        if args.track:
            wandb.log(metrics, step=global_step)

    envs.close()
    if run is not None:
        run.finish()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    train(parse_args())
