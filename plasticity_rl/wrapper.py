import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Any, List


# =========================================================
# wrappers
# =========================================================
class SaveRawReward(gym.Wrapper):
    """
    Save the *pre-normalization/pre-clipping* reward into info["raw_reward"].
    IMPORTANT: Place this wrapper BEFORE NormalizeReward / TransformReward.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)  # ensure mutable
        info["raw_reward"] = float(reward)
        return obs, reward, terminated, truncated, info


class ContinualDMC:
    """Sequential DeepMind Control task family backed by a vector environment."""

    ENV_NAME_DICT = {
        "dog": [
            "dm_control/dog-stand-v0",
            "dm_control/dog-walk-v0",
            "dm_control/dog-run-v0",
            "dm_control/dog-trot-v0",
        ],
        "walker": [
            "dm_control/walker-stand-v0",
            "dm_control/walker-walk-v0",
            "dm_control/walker-run-v0",
            "dm_control/walker-stand-v0",
            "dm_control/walker-walk-v0",
            "dm_control/walker-run-v0",

        ],
        "quadruped": [
            "dm_control/quadruped-walk-v0",
            "dm_control/quadruped-run-v0",
            "dm_control/quadruped-walk-v0",
        ],
    }

    def __init__(
        self,
        env_id: str,
        num_envs: int = 4,
        gamma: float = 0.99,
        capture_video: bool = False,
        run_name: Optional[str] = None,
        seed: Optional[int] = None,
        use_async: bool = True,
        shared_memory: bool = True,
    ):
        if env_id not in self.ENV_NAME_DICT:
            raise ValueError(
                f"env_id must be one of {list(self.ENV_NAME_DICT)}, got {env_id}"
            )

        self.env_id = env_id
        self.num_envs = num_envs
        self.gamma = gamma
        self.capture_video = capture_video
        self.run_name = run_name
        self.seed = seed
        self.use_async = use_async
        self.shared_memory = shared_memory

        self.env_name_list = self.ENV_NAME_DICT[env_id]
        self.num_tasks = len(self.env_name_list)
        self.current_task_idx = 0
        self.current_env_name = self.env_name_list[self.current_task_idx]

        self.envs = self._build_env()
        self._update_spaces()

    def _make_env(self, env_name: str, idx: int):
        capture_video = self.capture_video
        run_name = self.run_name
        gamma = self.gamma

        def thunk():
            import shimmy  # noqa: F401  # Register dm_control environments.

            if capture_video and idx == 0:
                env = gym.make(env_name, render_mode="rgb_array")
                if run_name:
                    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_name)

            # Match the public C-CHAIN DMC preprocessing chain.
            env = gym.wrappers.FlattenObservation(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env,
                lambda obs: np.clip(obs, -10, 10),
                env.observation_space,
            )
            env = SaveRawReward(env)
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(
                env,
                lambda reward: np.clip(reward, -10, 10),
            )
            return env

        return thunk

    def _build_env(self):
        env_fns = [
            self._make_env(self.current_env_name, idx)
            for idx in range(self.num_envs)
        ]
        vector_kwargs = {
            "autoreset_mode": gym.vector.AutoresetMode.SAME_STEP,
        }

        if self.use_async and self.num_envs > 1:
            return gym.vector.AsyncVectorEnv(
                env_fns,
                shared_memory=self.shared_memory,
                **vector_kwargs,
            )
        return gym.vector.SyncVectorEnv(env_fns, **vector_kwargs)

    def _update_spaces(self):
        self.observation_space = self.envs.single_observation_space
        self.action_space = self.envs.single_action_space
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

    def switch(self):
        if self.current_task_idx >= self.num_tasks - 1:
            raise RuntimeError("Cannot switch past the final continual DMC task.")

        self.current_task_idx += 1
        self.current_env_name = self.env_name_list[self.current_task_idx]
        self.envs.close()
        self.envs = self._build_env()
        self._update_spaces()

        print(
            f"Switched to task {self.current_task_idx + 1}/{self.num_tasks}: "
            f"{self.current_env_name}"
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        reset_seed = seed if seed is not None else self.seed
        return self.envs.reset(seed=reset_seed, options=options)

    def step(
        self,
        action: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        return self.envs.step(action)

    def get_current_task(self) -> str:
        return self.current_env_name

    def get_current_task_idx(self) -> int:
        return self.current_task_idx

    def get_num_tasks(self) -> int:
        return self.num_tasks

    def get_task_list(self) -> List[str]:
        return self.env_name_list.copy()

    def close(self):
        if hasattr(self, "envs"):
            self.envs.close()



class FloorFrictionWrapper(gym.Wrapper):
    """
    Generic floor-friction wrapper for MuJoCo locomotion tasks.

    It edits the sliding-friction term of a floor-like geom. Resolution is:
    1) exact or alias name match for common floor/ground geoms
    2) a unique plane geom in the MuJoCo model
    3) otherwise raise an error instead of silently touching the wrong geom
    """
    DEFAULT_FLOOR_GEOM_NAMES = (
        "floor",
        "ground",
        "plane",
        "terrain",
        "world",
        "groundplane",
    )

    def __init__(self, env, floor_geom_name: str = "floor", floor_geom_names=None):
        super().__init__(env)
        self.floor_geom_name = floor_geom_name
        self.floor_geom_names = self._build_floor_name_candidates(floor_geom_name, floor_geom_names)
        self.floor_geom_id = None
        self.floor_geom_label = None

    def _build_floor_name_candidates(self, primary_name, extra_names):
        names = []
        for name in (primary_name, *(extra_names or ()), *self.DEFAULT_FLOOR_GEOM_NAMES):
            if name is None:
                continue
            normalized = str(name).strip().lower()
            if normalized and normalized not in names:
                names.append(normalized)
        return tuple(names)

    def _resolve_floor_geom(self):
        model = self.unwrapped.model
        named_matches = []

        for i in range(model.ngeom):
            geom_name = (model.geom(i).name or "").strip()
            if geom_name.lower() in self.floor_geom_names:
                named_matches.append(i)

        if len(named_matches) == 1:
            return named_matches[0], "name"
        if len(named_matches) > 1:
            raise RuntimeError(
                f"Multiple floor geom name matches found: "
                f"{[(idx, model.geom(idx).name) for idx in named_matches]}"
            )

        plane_matches = [i for i in range(model.ngeom) if int(model.geom_type[i]) == 0]
        if len(plane_matches) == 1:
            return plane_matches[0], "plane"
        if len(plane_matches) > 1:
            raise RuntimeError(
                f"Multiple plane geoms found; floor target is ambiguous: "
                f"{[(idx, model.geom(idx).name) for idx in plane_matches]}"
            )

        raise RuntimeError(
            "Failed to locate a floor geom. "
            f"Tried names={self.floor_geom_names} and plane-geom fallback."
        )

    def _find_floor_id(self):
        self.floor_geom_id, source = self._resolve_floor_geom()
        geom_name = self.unwrapped.model.geom(self.floor_geom_id).name or f"geom_{self.floor_geom_id}"
        self.floor_geom_label = f"{geom_name} (id={self.floor_geom_id}, source={source})"
        print(f"[FloorFrictionWrapper] Target geom resolved: {self.floor_geom_label}")

    def get_target_geom_info(self):
        if self.floor_geom_id is None:
            self._find_floor_id()

        model = self.unwrapped.model
        body_id = int(model.geom_bodyid[self.floor_geom_id])
        body_name = model.body(body_id).name or f"body_{body_id}"
        geom_name = model.geom(self.floor_geom_id).name or f"geom_{self.floor_geom_id}"
        return {
            "geom_id": int(self.floor_geom_id),
            "geom_name": geom_name,
            "body_name": body_name,
            "geom_type": int(model.geom_type[self.floor_geom_id]),
        }

    def set_friction(self, friction_coef):
        if self.floor_geom_id is None:
            self._find_floor_id()

        self.unwrapped.model.geom_friction[self.floor_geom_id, 0] = friction_coef

        return self.get_friction()

    def get_friction(self):
        if self.floor_geom_id is None:
            self._find_floor_id()
        return self.unwrapped.model.geom_friction[self.floor_geom_id, 0].copy()







class HopperCyclicBlockPermutationWrapper(gym.ObservationWrapper):
    """
    Deterministic Hopper-v5 permutation wrapper with a 5-step cyclic schedule.

    The flat 11-d observation is treated as five semantically paired blocks:
      - [rootz, rootz_dot]
      - [rooty, rooty_dot]
      - [thigh, thigh_dot]
      - [leg, leg_dot]
      - [foot, foot_dot]

    The forward-velocity term rootx_dot (index 5) stays fixed. Each schedule
    step performs a cyclic block shift, so the permutation does not return to
    the initial arrangement until five distinct regimes have been visited.
    """

    BLOCK_NAMES = ("rootz", "rooty", "thigh", "leg", "foot")
    BLOCK_IDXS = (
        (0, 6),
        (1, 7),
        (2, 8),
        (3, 9),
        (4, 10),
    )
    FIXED_IDX = 5

    def __init__(self, env):
        super().__init__(env)
        if len(self.observation_space.shape) != 1 or self.observation_space.shape[0] != 11:
            raise ValueError(
                "HopperCyclicBlockPermutationWrapper expects a flat Hopper-v5 observation of shape (11,)."
            )
        self.obs_dim = self.observation_space.shape[0]
        self.perm = np.arange(self.obs_dim, dtype=np.int64)
        self.schedule_index = 0
        self.block_order = list(self.BLOCK_NAMES)

    def set_permutation(self, perm_seed):
        self.schedule_index = int(perm_seed)
        shift = self.schedule_index % len(self.BLOCK_IDXS)
        rotated_block_ids = [
            (dst_block_idx + shift) % len(self.BLOCK_IDXS)
            for dst_block_idx in range(len(self.BLOCK_IDXS))
        ]

        perm = np.empty(self.obs_dim, dtype=np.int64)
        block_order = []
        for dst_block_idx, src_block_idx in enumerate(rotated_block_ids):
            dst_pos_idx, dst_vel_idx = self.BLOCK_IDXS[dst_block_idx]
            src_pos_idx, src_vel_idx = self.BLOCK_IDXS[src_block_idx]
            perm[dst_pos_idx] = src_pos_idx
            perm[dst_vel_idx] = src_vel_idx
            block_order.append(self.BLOCK_NAMES[src_block_idx])

        perm[self.FIXED_IDX] = self.FIXED_IDX
        self.perm = perm
        self.block_order = block_order

        print(
            f"[Plasticity] Hopper cyclic block schedule idx={self.schedule_index} "
            f"shift={shift} order={self.block_order}"
        )
        print(f"[Plasticity] Cyclic perm indices: {self.perm}")

    def observation(self, observation):
        return observation[self.perm]


class Walker2dCyclicBlockPermutationWrapper(gym.ObservationWrapper):
    """
    Deterministic Walker2d-v5 permutation wrapper with an 8-step cyclic schedule.

    The flat 17-d observation is treated as eight semantically paired blocks:
      - [rootz, rootz_dot]
      - [rooty, rooty_dot]
      - [thigh, thigh_dot]
      - [leg, leg_dot]
      - [foot, foot_dot]
      - [thigh_left, thigh_left_dot]
      - [leg_left, leg_left_dot]
      - [foot_left, foot_left_dot]

    The forward-velocity term rootx_dot (index 8) stays fixed. Each schedule
    step performs a cyclic block shift, so the permutation does not return to
    the initial arrangement until eight distinct regimes have been visited.
    """

    BLOCK_NAMES = (
        "rootz",
        "rooty",
        "thigh",
        "leg",
        "foot",
        "thigh_left",
        "leg_left",
        "foot_left",
    )
    BLOCK_IDXS = (
        (0, 9),
        (1, 10),
        (2, 11),
        (3, 12),
        (4, 13),
        (5, 14),
        (6, 15),
        (7, 16),
    )
    FIXED_IDX = 8

    def __init__(self, env):
        super().__init__(env)
        if len(self.observation_space.shape) != 1 or self.observation_space.shape[0] != 17:
            raise ValueError(
                "Walker2dCyclicBlockPermutationWrapper expects a flat Walker2d-v5 observation of shape (17,)."
            )
        self.obs_dim = self.observation_space.shape[0]
        self.perm = np.arange(self.obs_dim, dtype=np.int64)
        self.schedule_index = 0
        self.block_order = list(self.BLOCK_NAMES)

    def set_permutation(self, perm_seed):
        self.schedule_index = int(perm_seed)
        shift = self.schedule_index % len(self.BLOCK_IDXS)
        rotated_block_ids = [
            (dst_block_idx + shift) % len(self.BLOCK_IDXS)
            for dst_block_idx in range(len(self.BLOCK_IDXS))
        ]

        perm = np.empty(self.obs_dim, dtype=np.int64)
        block_order = []
        for dst_block_idx, src_block_idx in enumerate(rotated_block_ids):
            dst_pos_idx, dst_vel_idx = self.BLOCK_IDXS[dst_block_idx]
            src_pos_idx, src_vel_idx = self.BLOCK_IDXS[src_block_idx]
            perm[dst_pos_idx] = src_pos_idx
            perm[dst_vel_idx] = src_vel_idx
            block_order.append(self.BLOCK_NAMES[src_block_idx])

        perm[self.FIXED_IDX] = self.FIXED_IDX
        self.perm = perm
        self.block_order = block_order

        print(
            f"[Plasticity] Walker2d cyclic block schedule idx={self.schedule_index} "
            f"shift={shift} order={self.block_order}"
        )
        print(f"[Plasticity] Cyclic perm indices: {self.perm}")

    def observation(self, observation):
        return observation[self.perm]


class HalfCheetahCyclicBlockPermutationWrapper(gym.ObservationWrapper):
    """
    Deterministic HalfCheetah-v5 permutation wrapper with an 8-step cyclic schedule.

    The flat 17-d observation is treated as eight semantically paired blocks:
      - [rootz, rootz_dot]
      - [rooty, rooty_dot]
      - [bthigh, bthigh_dot]
      - [bshin, bshin_dot]
      - [bfoot, bfoot_dot]
      - [fthigh, fthigh_dot]
      - [fshin, fshin_dot]
      - [ffoot, ffoot_dot]

    The forward-velocity term rootx_dot (index 8) stays fixed. Each schedule
    step performs a cyclic block shift, so the permutation does not return to
    the initial arrangement until eight distinct regimes have been visited.
    """

    BLOCK_NAMES = (
        "rootz",
        "rooty",
        "bthigh",
        "bshin",
        "bfoot",
        "fthigh",
        "fshin",
        "ffoot",
    )
    BLOCK_IDXS = (
        (0, 9),
        (1, 10),
        (2, 11),
        (3, 12),
        (4, 13),
        (5, 14),
        (6, 15),
        (7, 16),
    )
    FIXED_IDX = 8

    def __init__(self, env):
        super().__init__(env)
        if len(self.observation_space.shape) != 1 or self.observation_space.shape[0] != 17:
            raise ValueError(
                "HalfCheetahCyclicBlockPermutationWrapper expects a flat HalfCheetah-v5 observation of shape (17,)."
            )
        self.obs_dim = self.observation_space.shape[0]
        self.perm = np.arange(self.obs_dim, dtype=np.int64)
        self.schedule_index = 0
        self.block_order = list(self.BLOCK_NAMES)

    def set_permutation(self, perm_seed):
        self.schedule_index = int(perm_seed)
        shift = self.schedule_index % len(self.BLOCK_IDXS)
        rotated_block_ids = [
            (dst_block_idx + shift) % len(self.BLOCK_IDXS)
            for dst_block_idx in range(len(self.BLOCK_IDXS))
        ]

        perm = np.empty(self.obs_dim, dtype=np.int64)
        block_order = []
        for dst_block_idx, src_block_idx in enumerate(rotated_block_ids):
            dst_pos_idx, dst_vel_idx = self.BLOCK_IDXS[dst_block_idx]
            src_pos_idx, src_vel_idx = self.BLOCK_IDXS[src_block_idx]
            perm[dst_pos_idx] = src_pos_idx
            perm[dst_vel_idx] = src_vel_idx
            block_order.append(self.BLOCK_NAMES[src_block_idx])

        perm[self.FIXED_IDX] = self.FIXED_IDX
        self.perm = perm
        self.block_order = block_order

        print(
            f"[Plasticity] HalfCheetah cyclic block schedule idx={self.schedule_index} "
            f"shift={shift} order={self.block_order}"
        )
        print(f"[Plasticity] Cyclic perm indices: {self.perm}")

    def observation(self, observation):
        return observation[self.perm]
