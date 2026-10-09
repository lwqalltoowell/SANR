# SANR

Research code for SANR (Soft Norm Anchoring with ReDo) and comparison methods for studying neural network plasticity in non-stationary reinforcement learning with Proximal Policy Optimization (PPO).

The experiments cover cyclic observation permutations in MuJoCo Hopper, Walker2d, and HalfCheetah, and sequential DeepMind Control tasks for Walker, Dog, and Quadruped. The implementation includes PPO baselines, LayerNorm with L2 regularization, L2-Init, soft shrink-and-perturb, Soft Norm Anchoring (SNA), SNA with ReDo (SANR), ReDo, Continual Backpropagation (CBP), and ReGraMa.

## Installation

Use Python 3.10 or later in a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Repository layout

```text
plasticity_rl/       PPO, plasticity methods, networks, monitors, and wrappers
experiments/         Training entry points
configs/            Task and method configuration
run_experiments.py  Batch experiment scheduler
```

## Example usage

Run these commands from the repository root. W&B tracking is disabled by default.

Train PPO on Hopper with cyclic observation permutations:

```bash
python -m experiments.hopper_reset_permu
```

Train with SNA on the continual DMC Walker sequence:

```bash
python -m experiments.continual_dmc_train --env_family walker --sna_lambda 0.001 --use_layernorm true
```

Run the configured SANR experiments on Hopper, or preview the commands:

```bash
python run_experiments.py --task hopper --method SANR
python run_experiments.py --task hopper --method SANR --dry-run
```

Tasks and method settings are defined in `configs/experiments.yaml`. Script paths in this configuration are relative to the repository root. Use `--help` on a training entry point or the scheduler to see the available options.

To use W&B, log in with your own account and enable `--track true`. You can select a team with `--wandb_entity YOUR_TEAM` and a project with `--wandb_project_name YOUR_PROJECT`.
