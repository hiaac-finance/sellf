from typing import Any, Dict, List, Optional, Tuple, Type, Union

from tqdm import tqdm
import gym
import numpy as np
import torch
import torch as th

from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from agents.buffers import RolloutBuffer, ReplayMemory
from lending_experiment.agents.policy import Agent
from stable_baselines3.common.logger import Logger


class OnPolicyAlgorithm:
    def __init__(
        self,
        env: Union[gym.Env, VecEnv],
        learning_rate: float,
        n_steps: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.2,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "cpu",
    ):
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.env = env
        self.device = device
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        if hasattr(env, "num_envs"):
            self.n_envs = env.num_envs
        else:
            self.n_envs = 1
        self.learning_rate = learning_rate
        self._last_obs = None
        self._last_episode_starts = None
        self.seed = seed
        self.policy_kwargs = policy_kwargs

    def set_random_seed(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        th.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

    def _setup_model(self) -> None:

        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        self.memory = ReplayMemory(
            10000,
            self.observation_space,
            self.action_space,
            device=self.device,
        )

        self.policy = Agent(
            observation_space=self.observation_space,
            action_space=self.action_space,
            learning_rate=self.learning_rate,
            use_predictor=self.policy_kwargs.get("use_predictor", False),
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> None:
        if self._last_obs is None:
            self._last_obs = env.reset()
            self._last_episode_starts = np.ones((env.num_envs,), dtype=bool)

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.eval()

        # first, predict for everyone in the pool
        env.env_method("update_models")

        n_steps = 0
        rollout_buffer.reset()

        while n_steps < n_rollout_steps:

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs, _ = self.policy.get_action_and_value(
                    obs_tensor
                )
                label_pred = self.policy.get_label(obs_tensor)
            actions = actions.cpu().numpy()
            label_pred = label_pred.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

            idx = env.get_attr("idx")[0]
            data = env.get_attr("data")[0]
            label = np.array([data["label"][idx]]).astype(np.float32)
            group_idx = data["group"][idx]
            group = np.zeros(2, dtype=np.float32)
            group[group_idx] = 1


            imputation = label if actions[0] == 1 else label_pred

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.get_value(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            delta = torch.tensor(np.array([env.get_attr("delta")[0]])).float()
            delta_obs = torch.tensor(np.array([env.get_attr("delta_obs")])).float()
            delta_delta = torch.tensor(np.array([env.get_attr("delta_delta")])).float()
            delta_pred = torch.tensor(np.array([env.get_attr("delta_pred")])).float()
            rollout_buffer.add(
                self._last_obs,
                actions,
                label,
                label_pred,
                imputation,
                group,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                delta,
                delta_obs,
                delta_delta,
                delta_pred,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.get_value(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self,
        total_timesteps: int,
    ) -> "OnPolicyAlgorithm":
        pbar = tqdm(total=total_timesteps)
        self.num_timesteps = 0
        while self.num_timesteps < total_timesteps:

            self.collect_rollouts(
                self.env, self.rollout_buffer, n_rollout_steps=self.n_steps
            )
            self.logger.dump(step=self.num_timesteps)
            self.train()
            pbar.update(self.n_steps)

        # callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

    def set_logger(self, logger: Logger) -> None:
        """
        Setter for for logger object.

        .. warning::

          When passing a custom logger object,
          this will overwrite ``tensorboard_log`` and ``verbose`` settings
          passed to the constructor.
        """
        self._logger = logger
        # User defined logger
        self._custom_logger = True

    @property
    def logger(self) -> Logger:
        return self._logger

    def save(self, path: str) -> None:
        """
        Save the model to a file.
        """
        torch.save(self.policy.state_dict(), path + ".pth")

    def load(self, path: str) -> None:
        """
        Load the model from a file.
        """
        self.policy.load_state_dict(torch.load(path + ".pth", map_location=self.device))
        self.policy.eval()
