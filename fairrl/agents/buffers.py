import warnings
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Union,
    Callable,
    NamedTuple,
    Tuple,
)

from enum import Enum

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common import callbacks, vec_env


try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    labels: th.Tensor
    preds: th.Tensor
    imputations: th.Tensor
    groups: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    deltas: th.Tensor
    delta_obs: th.Tensor
    delta_preds: th.Tensor
    delta_deltas: th.Tensor
    prob_action_all: th.Tensor


class ReplayMemorySamples(NamedTuple):
    observations: th.Tensor
    labels: th.Tensor
    groups: th.Tensor


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)
        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> RolloutBufferSamples:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default
        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array).to(self.device)
        return th.as_tensor(array).to(self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(
        reward: np.ndarray, env: Optional[VecNormalize] = None
    ) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.
    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):

        super(RolloutBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = (
            None,
            None,
            None,
            None,
        )
        self.returns, self.episode_starts, self.values, self.log_probs = (
            None,
            None,
            None,
            None,
        )
        self.deltas = None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:

        self.observations = np.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.labels = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.imputations = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.preds = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.groups = np.zeros((self.buffer_size, self.n_envs, 2), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.deltas = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.delta_obs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.delta_deltas = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.delta_preds = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.prob_action_all = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def compute_returns_and_advantage(
        self, last_values: th.Tensor, dones: np.ndarray
    ) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.
        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.
        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))
        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.
        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        label: np.ndarray,
        pred: np.ndarray,
        imputation: np.ndarray,
        group: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        deltas: th.Tensor,
        delta_obs: th.Tensor,
        delta_delta: th.Tensor,
        delta_preds: th.Tensor,
        prob_action_all: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.labels[self.pos] = np.array(label).copy()
        self.preds[self.pos] = np.array(pred).copy()
        self.imputations[self.pos] = np.array(imputation).copy()
        self.groups[self.pos] = np.array(group).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.deltas[self.pos] = deltas.copy()
        self.delta_obs[self.pos] = delta_obs.copy()
        self.delta_deltas[self.pos] = delta_delta.copy()
        self.delta_preds[self.pos] = delta_preds.copy()
        self.prob_action_all[self.pos] = prob_action_all.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "labels",
                "preds",
                "imputations",
                "groups",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "deltas",
                "delta_obs",
                "delta_deltas",
                "delta_preds",
                "prob_action_all",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.labels[batch_inds],
            self.preds[batch_inds],
            self.imputations[batch_inds],
            self.groups[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.deltas[batch_inds].flatten(),
            self.delta_obs[batch_inds].flatten(),
            self.delta_deltas[batch_inds].flatten(),
            self.delta_preds[batch_inds].flatten(),
            self.prob_action_all[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


class ReplayMemory(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
    ):
        super(ReplayMemory, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )
        self.n_envs = n_envs
        self.observation_space = observation_space
        self.buffer_size = buffer_size
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((0,) + self.obs_shape, dtype=np.float32)
        self.labels = np.zeros((0, 1), dtype=np.float32)
        self.groups = np.zeros((0, 2), dtype=np.float32)
        self.pos = 0

    def add(self, obs: np.ndarray, label: np.ndarray, group : np.ndarray) -> None:
        # this function should concatenate the old and the new data, and
        # sample the new data to the original size

        if obs.shape[1] == 1:
            obs = obs.squeeze(axis=1)
        if group.shape[1] == 1:
            group = group.squeeze(axis=1)

        self.observations = np.concatenate((self.observations, obs), axis=0)
        self.labels = np.concatenate((self.labels, label), axis=0)
        self.groups = np.concatenate((self.groups, group), axis=0)

        if self.observations.shape[0] > self.buffer_size:
            indices = np.random.choice(
                np.arange(self.observations.shape[0]),
                size=self.buffer_size,
                replace=True,
            )
            self.observations = self.observations[indices]
            self.labels = self.labels[indices]
            self.groups = self.groups[indices]

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[ReplayMemorySamples, None, None]:
        # Sample random indices
        indices = np.random.permutation(self.observations.shape[0])
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "labels",
                "groups",
            ]

            # for tensor in _tensor_names:
            #    self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < (self.observations.shape[0] - batch_size):
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayMemorySamples:
        data = (
            self.observations[batch_inds],
            self.labels[batch_inds],
            self.groups[batch_inds],
        )
        return ReplayMemorySamples(*tuple(map(self.to_torch, data)))
