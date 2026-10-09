# continual_dmc_train.py
import argparse
import os
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import wandb

from plasticity_rl.agent import PPOAgent
from plasticity_rl.wrapper import ContinualDMC


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
    parser = argparse.ArgumentParser(
        description="PPO Plasticity Test: Continual DeepMind Control task sequence"
    )

    parser.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help=(
            "Transition budget for the complete continual task sequence. "
            "Defaults to 1,000,000 transitions per task and is rounded down "
            "so every task receives the same number of complete PPO batches."
        ),
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--anneal_lr", type=str2bool, default=False)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)

    # Plasticity methods shared with reset_permu entry points.
    parser.add_argument("--sna_lambda", type=float, default=0.0)
    parser.add_argument("--snp_beta", type=float, default=0.0)
    parser.add_argument("--use_layernorm", type=str2bool, default=False)
    parser.add_argument("--use_affine", type=str2bool, default=True)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--use_redo", type=str2bool, default=False)
    parser.add_argument("--redo_frequency", type=int, default=5)
    parser.add_argument("--redo_tau", type=float, default=0.025)

    parser.add_argument("--use_regrama", type=str2bool, default=False)
    parser.add_argument("--regrama_tau", type=float, default=0.025)
    parser.add_argument("--regrama_frequency", type=int, default=1)

    parser.add_argument("--regen_coef", type=float, default=0.0)
    parser.add_argument("--pfo_coef", type=float, default=0.0)

    parser.add_argument("--use_cbp", type=str2bool, default=False)
    parser.add_argument("--cbp_maturity_threshold", type=int, default=10_000)

    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=2048)
    parser.add_argument("--num_minibatches", type=int, default=32)
    parser.add_argument("--update_epochs", type=int, default=10)

    parser.add_argument("--norm_adv", type=str2bool, default=True)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--clip_vloss", type=str2bool, default=True)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=None)

    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layer", type=int, default=2)

    parser.add_argument(
        "--env_family",
        type=str,
        default="walker",
        choices=list(ContinualDMC.ENV_NAME_DICT),
    )
    parser.add_argument(
        "--reset_agent_on_switch",
        type=str2bool,
        default=False,
        help="Reinitialize the complete PPO agent at each task switch (Fresh/Oracle).",
    )

    parser.add_argument("--capture_video", type=str2bool, default=False)
    parser.add_argument("--run_name_for_video", type=str, default=None)
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=None,
        help=(
            "Override the W&B project name. By default it is derived from "
            "--env_family, for example SANR-ContinualDMC-Walker."
        ),
    )
    parser.add_argument("--group_name", type=str, default="baseline")
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
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
    parser = build_parser()
    args = parser.parse_args()

    positive_int_args = (
        "num_envs",
        "num_steps",
        "num_minibatches",
        "update_epochs",
        "hidden_size",
        "num_layer",
    )
    for name in positive_int_args:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")

    args.batch_size = args.num_envs * args.num_steps
    if args.batch_size % args.num_minibatches != 0:
        parser.error(
            "num_envs * num_steps must be divisible by num_minibatches "
            f"(got batch_size={args.batch_size})"
        )
    args.minibatch_size = args.batch_size // args.num_minibatches

    args.num_tasks = len(ContinualDMC.ENV_NAME_DICT[args.env_family])
    if args.wandb_project_name is None:
        family_name = args.env_family.capitalize()
        args.wandb_project_name = f"SANR-ContinualDMC-{family_name}"

    if args.total_steps is None:
        args.total_steps = 1_000_000 * args.num_tasks
    elif args.total_steps <= 0:
        parser.error("--total_steps must be positive")

    args.num_iterations_per_task = (
        args.total_steps // (args.num_tasks * args.batch_size)
    )
    if args.num_iterations_per_task < 1:
        parser.error(
            "total_steps must contain at least one complete PPO batch per "
            f"task (num_tasks={args.num_tasks}, batch_size={args.batch_size})"
        )

    args.actual_task_timesteps = args.num_iterations_per_task * args.batch_size
    args.num_iterations = args.num_tasks * args.num_iterations_per_task
    actual_total_steps = args.actual_task_timesteps * args.num_tasks

    if actual_total_steps != args.total_steps:
        print(
            f"[Budget] total_steps={args.total_steps} cannot be divided "
            f"into equal complete PPO batches across {args.num_tasks} tasks; "
            f"the run will use {actual_total_steps} transitions "
            f"({args.actual_task_timesteps} per task)."
        )

    if args.experiment_name is None:
        args.experiment_name = (
            f"{args.env_family}_seed{args.seed}_h{args.hidden_size}"
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
    if args.use_regrama:
        args.experiment_name += (
            f"_ReGraMa{args.regrama_tau}_F{args.regrama_frequency}"
        )

    return args


def set_seed(seed: int, torch_deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def make_continual_dmc_envs(args):
    return ContinualDMC(
        env_id=args.env_family,
        num_envs=args.num_envs,
        gamma=args.gamma,
        capture_video=args.capture_video,
        run_name=args.run_name_for_video,
        seed=args.seed,
        use_async=args.use_async_vecenv,
        shared_memory=args.shared_memory,
    )


def extract_final_episode_stats(infos: dict, done: np.ndarray):
    """Return (env_index, raw_return, length) from SAME_STEP vector info."""
    if not isinstance(infos, dict) or "final_info" not in infos:
        return []

    final_info = infos["final_info"]
    final_mask = np.asarray(infos.get("_final_info", done), dtype=bool)
    episodes = []

    # Gymnasium >= 1.0 batches nested final-info dictionaries.
    if isinstance(final_info, dict):
        episode = final_info.get("episode")
        if not isinstance(episode, dict) or "r" not in episode or "l" not in episode:
            return episodes

        episode_mask = np.asarray(
            final_info.get("_episode", final_mask),
            dtype=bool,
        )
        returns = np.asarray(episode["r"])
        lengths = np.asarray(episode["l"])
        for idx in np.flatnonzero(done & final_mask & episode_mask):
            episodes.append((int(idx), float(returns[idx]), int(lengths[idx])))
        return episodes

    # Compatibility with older Gymnasium object-array/list final_info format.
    for idx, info in enumerate(final_info):
        if not done[idx] or not final_mask[idx] or not info:
            continue
        episode = info.get("episode")
        if episode is not None:
            episodes.append(
                (idx, float(np.asarray(episode["r"])), int(np.asarray(episode["l"])))
            )
    return episodes


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

    envs = make_continual_dmc_envs(args)
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "Only continuous action spaces are supported."
    )

    agent = PPOAgent(
        envs.single_observation_space,
        envs.single_action_space,
        args,
        device,
    )
    agent.capture_regen_reference()

    print(f"Observation space shape: {envs.single_observation_space.shape}")
    print(f"Action space shape: {envs.single_action_space.shape}")
    print(f"PPO batch size: {args.batch_size}")

    global_step = 0
    start_time = time.time()
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.float32)

    ep_return_buffer = deque(maxlen=40)

    print(f"\n[Init] Continual DMC family: {args.env_family}")
    print(f"[Init] Number of tasks: {envs.get_num_tasks()}")
    print(f"[Init] Task list: {envs.get_task_list()}")
    print(f"[Init] Starting task: {envs.get_current_task()}\n")

    for task_seq_idx in range(args.num_tasks):
        current_task_idx = envs.get_current_task_idx()
        current_task_name = envs.get_current_task()

        print("\n" + "=" * 80)
        print(f"[Task Start] task_seq_idx={task_seq_idx + 1}/{args.num_tasks}")
        print(f"[Task Start] current_task_idx={current_task_idx}")
        print(f"[Task Start] current_task_name={current_task_name}")
        print("=" * 80 + "\n")

        for local_iter in range(1, args.num_iterations_per_task + 1):
            iteration = task_seq_idx * args.num_iterations_per_task + local_iter

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
                        actions, logprobs, values = agent.act(
                            obs_tensor,
                            deterministic=False,
                        )
                        actor_preacts = None

                next_obs_np, rewards_np, terminations, truncations, infos = envs.step(
                    actions.cpu().numpy()
                )
                done_np = np.logical_or(terminations, truncations)

                rewards_tensor = torch.as_tensor(
                    rewards_np,
                    device=device,
                    dtype=torch.float32,
                )
                next_done = torch.as_tensor(
                    done_np,
                    device=device,
                    dtype=torch.float32,
                )
                next_obs = torch.as_tensor(
                    next_obs_np,
                    device=device,
                    dtype=torch.float32,
                )

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

                for _, ep_return, _ in extract_final_episode_stats(
                    infos,
                    done_np,
                ):
                    ep_return_buffer.append(ep_return)
                    if args.track:
                        log_dict = {
                            "Reward/episodic_return": ep_return,
                        }
                        if len(ep_return_buffer) >= 10:
                            log_dict["Reward/avg_return"] = float(
                                np.mean(ep_return_buffer)
                            )
                        wandb.log(log_dict, step=global_step)

            agent.compute_returns_and_advantages(next_obs, next_done)
            metrics = agent.update(
                iteration=iteration,
                num_iterations=args.num_iterations,
            )
            metrics["Speed/SPS"] = int(global_step / (time.time() - start_time))

            if args.track:
                wandb.log(metrics, step=global_step)

        if task_seq_idx >= args.num_tasks - 1:
            continue

        previous_task_name = current_task_name
        print("\n" + "=" * 80)
        print("[Plasticity Test] Switching Continual DMC task")
        print(f"Global Step: {global_step}")
        print(f"Task: {previous_task_name} -> next")
        print("=" * 80 + "\n")

        envs.switch()
        next_obs_np, _ = envs.reset(seed=args.seed)
        next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
        next_done = torch.zeros(args.num_envs, device=device, dtype=torch.float32)

        # The closed task may end mid-episode; do not mix its return with the
        # new task's rolling metric.
        ep_return_buffer.clear()

        if args.reset_agent_on_switch:
            agent_reset_seed = args.seed + task_seq_idx + 1
            torch.manual_seed(agent_reset_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(agent_reset_seed)
            agent = PPOAgent(
                envs.single_observation_space,
                envs.single_action_space,
                args,
                device,
            )
            agent.capture_regen_reference()
            print(f"[Agent Reset] Fresh agent initialized with seed={agent_reset_seed}.")

    if args.save_model:
        model_path = os.path.join(args.model_save_dir, f"{args.experiment_name}.pt")
        agent.save_model(model_path)

    envs.close()
    if run is not None:
        run.finish()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    train(parse_args())
