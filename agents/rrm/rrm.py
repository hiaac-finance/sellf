import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Set, NamedTuple, Generator
from tqdm import tqdm

import gym
from gym import spaces
import numpy as np
import torch
import torch as th

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

from .policy import ActorPolicy

from stable_baselines3.common.buffers import BaseBuffer
from config import Config

class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    y_obs: th.Tensor
    group: th.Tensor

class RRMRolloutBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        num_groups: int = 2,
    ):
        super(RRMRolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.buffer_size = buffer_size
        self.device = device
        self.n_envs = n_envs
        self.obs_shape = observation_space.shape
        self.num_groups = num_groups
        self.reset()
    
    
    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.y_obs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.group = np.zeros((self.buffer_size, self.n_envs, self.num_groups), dtype = np.int64)
        self.pos = 0
        self.full = False
    
    def add(
        self,
        obs: np.ndarray,
        y_obs: np.ndarray,
        g: np.ndarray,
    ) -> None:
        self.observations[self.pos] = np.array(obs).copy()
        self.y_obs[self.pos] = np.array(y_obs).copy()
        self.group[self.pos] = g
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def get(
        self, 
        batch_size: int
    ) -> Generator[RolloutBufferSamples, None, None]:
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        start_idx = 0
        while start_idx < len(indices):
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.y_obs[batch_inds],
            self.group[batch_inds],
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

class RRM(BaseAlgorithm):
    """
    Class that implements Repeated Risk Minimization
    """
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        n_steps: int = 1024,
        batch_size: int = 64,
        n_epochs: int = 10,
        beta: float = 0.,
        policy_base: Type[BasePolicy] = ActorCriticPolicy,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
        **kwargs,
    ):
        super(RRM, self).__init__(
            policy=policy,
            env=env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.beta = beta
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        if _init_setup_model:
            self._setup_model()
        

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        if self.env is None:
            self.rollout_buffer = None
        else:
            self.rollout_buffer = RRMRolloutBuffer(
                buffer_size=self.n_steps,
                observation_space=self.observation_space,
                action_space=self.action_space,
                device=self.device,
                n_envs=self.env.num_envs,
                num_groups=2,
            )

        
        self.policy = ActorPolicy(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )

        self.policy = self.policy.to(self.device)

    def collect_rollouts(self,
        env: VecEnv,
        rollout_buffer: RRMRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions = self.predict(obs_tensor)[0]

            #actions = actions.cpu().numpy()

            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
           
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs
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
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            y_obs = np.array([
                env.get_attr("state")[0].y_obs
            ]).astype(np.float32)
            g = np.array(env.get_attr("state")[0].group).astype(np.int32)
            rollout_buffer.add(self._last_obs, y_obs, g)
            # ----------------------------------------------------------------
            self._last_obs = new_obs


        return True

    def train(self) -> None:
        """
        Update policy using few iterations of gradient descent.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range

        criterion = torch.nn.CrossEntropyLoss(reduction="none")

        for epoch in range(self.n_epochs):

            actions=[]
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                g0_idx = rollout_data.group[:, :, 0]
                g1_idx = rollout_data.group[:, :, 1]
                pred = self.policy(rollout_data.observations)
                y_obs = rollout_data.y_obs.view(-1).long()
                loss = criterion(pred, y_obs)
                loss_g0 = loss[g0_idx].mean()
                loss_g1 = loss[g1_idx].mean()
                loss = loss.mean() + self.beta * (loss_g0 - loss_g1)**2

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()


                ## Clip grad norm
                #th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                actions.append(
                    pred.argmax(dim=1).view(-1, 1).detach().cpu().numpy()
                )

                #if torch.isnan(self.policy.mlp_extractor.shared_net[0].weight).any():
                #    import pdb; pdb.set_trace()

        actions = np.concatenate(actions, axis=0)

        self.logger.record("train/loss", loss.item())
        self.logger.record("train/actions", actions.mean())
        self.logger.dump()
        self._n_updates += self.n_epochs

        #self.logger.record("train/loss", loss.item())
        #self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        #self.logger.record("train/cumulative_reward", self.rollout_buffer.rewards.sum().item())


    def learn(
        self,
        total_timesteps: int,
    ) -> "RRM":
        self._setup_learn(
            total_timesteps=total_timesteps, eval_env=None,
        )
        self.num_timesteps = 0
        pbar = tqdm(total = total_timesteps + 1)
        while self.num_timesteps < total_timesteps:
            old_timesteps = self.num_timesteps
            self._last_obs = self.env.reset()
            self.collect_rollouts(self.env, self.rollout_buffer, n_rollout_steps=self.n_steps)
            self.train()
            pbar.update(self.num_timesteps - old_timesteps)
        pbar.close()
        return self


    def _get_torch_save_params(self) -> Tuple[List[str], List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []