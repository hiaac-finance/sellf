import random
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import BaseBuffer

from agents.ppo.sb3.type_aliases import (
    RolloutBufferSamples,
)


class Individual:
    """
    Used to track the information for each 'individual' in the population. 
    """
    def __init__(self, idx, init_score, group) -> None:
        self.idx = idx
        self.group = group
        self.t_steps = []
        self.num_t_steps = 0
        self.c_scores = [init_score]
        self.q_values = []
        self.actions_pi0 = []
        self.next_x_pi0 = []
        self.gx_pi0 = []
        self.v_ps = None
        self.gx = None

    def add(self, t_step, next_c_score, qvals, action_pi0, next_x_pi0):
        self.t_steps.append(t_step)
        self.num_t_steps += 1
        self.c_scores.append(next_c_score)
        self.q_values.append(qvals)
        self.actions_pi0.append(action_pi0.item())
        self.next_x_pi0.append(next_x_pi0.item())

    def compute_q_mc(self, gx, gx_pi0, q_advs, v_ps, actions, save_vals=False):
        """
        Computes the reward-to-go for the individual for estimates of V_do(pi) and V_do(pi^ps)
        """
        q_tg = 0

        v_tg = 0.0
        for i in reversed(range(len(self.t_steps))):
            q_tg += gx[self.t_steps[i]]
            q_advs[self.t_steps[i]] = q_tg

            v_tg += gx_pi0[self.t_steps[i]]
            v_ps[self.t_steps[i]] = v_tg

        if save_vals:
            # for debugging
            self.gx = gx[self.t_steps].tolist()
            self.gx_pi0 = gx_pi0[self.t_steps].tolist()
            self.actions = actions[self.t_steps].tolist()
            self.q_advs = q_advs[self.t_steps].tolist()
            self.v_ps = v_ps[self.t_steps].tolist()

    def __repr__(self) -> str:
        return f'Individual(idx={self.idx}, group={self.group})'
    

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
        num_groups=2,
        group_size=100,
        init_dists=None,
        qual_change_func=None,
    ):
        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert qual_change_func is not None, "qual_change_func must be provided to RolloutBuffer"

        self.qual_change_func = qual_change_func

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.deltas = None
        self.generator_ready = False

        self.num_groups = num_groups
        self.group_size = group_size
        self.init_dists = init_dists
        self.g_loc_start = len(init_dists[0]) # number of credit score buckets

        self.cscore_to_init_qual_gain = np.zeros(len(init_dists[0]), dtype=np.float32)
        for i in range(len(init_dists[0])):
            self.cscore_to_init_qual_gain[i] = self.qual_change_func(0, i)

        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.deltas = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.delta_deltas = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        self.long_delta = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # -----------------------------------------------------------------------
        self.q_values = np.zeros((self.buffer_size, self.n_envs, 2), dtype=np.float32)
        self.q_advs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.gx = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.next_x = np.zeros((self.buffer_size, self.n_envs), dtype=np.uintc)
        self.curr_x = np.zeros((self.buffer_size, self.n_envs), dtype=np.uintc)
        self.q_log_prob = np.zeros((self.buffer_size, self.n_envs,2), dtype=np.float32)
        self.gx_pi0 = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.v_ps = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.expected_v_base = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.benefit_deltas = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        self._reset_c_buckets()

        self.pos_to_id = np.zeros((self.buffer_size, self.n_envs), dtype=np.uintc)
        # -------------------------------------------------------------------------------
        super(RolloutBuffer, self).reset()

    def _reset_c_buckets(self):
        """
        Initialize the credit score buckets for each group based on the initial distributions.
        Each bucket will contain individuals with the same credit score and group.
        """
        self.c_buckets = []
        start = 0
        for g in range(self.num_groups):
            g_idxs = [i for i in range(start,(g + 1) * self.group_size)]
            start = (g + 1) * self.group_size

            c_buckets = {}
            num_buckets = len(self.init_dists[g])

            g_start = 0
            for cscore in range(num_buckets):
                bucket_size = round(self.init_dists[g][cscore] * self.group_size)
                c_buckets[cscore] = {}
                for idx in g_idxs[g_start:g_start + bucket_size]:
                    c_buckets[cscore][idx] = Individual(idx, cscore, g)
                g_start += bucket_size

            assert g_start == self.group_size
            self.c_buckets.append(c_buckets)
        self._compute_x_to_expectected_v_base()

    def _compute_x_to_expectected_v_base(self):
        """
        Compute the expected value of the baseline policy pi_0 for each group and credit score bucket.
        Assumes credits scores will eventually be uniformly distributed
        """
        num_buckets = len(self.init_dists[0])
        self.x_to_expectected_v_base = np.zeros((self.num_groups,num_buckets))
        for g in range(self.num_groups):
            for i in range(num_buckets):
                for j in range(num_buckets):
                    self.x_to_expectected_v_base[g,i] += self.qual_change_func(i, j)
        self.x_to_expectected_v_base = self.x_to_expectected_v_base / num_buckets

    def _sample_and_update_individual(self, action_pi0, next_x_pi0):
        """
        Sample an individual from the current credit score bucket and update its attributes.
        The individual is then removed from current bucket and added to the next bucket
        according their next credit score.
        """
        cscore = self.curr_x[self.pos].item()
        next_cscore = self.next_x[self.pos].item()
        group = 0 if self.observations[self.pos][0][self.g_loc_start] >= 0.01 else 1

        try:
            idx = random.choice(list(self.c_buckets[group][cscore].keys()))
        except:
            breakpoint()
        self.pos_to_id[self.pos] = idx
        individual = self.c_buckets[group][cscore][idx]
        del self.c_buckets[group][cscore][idx]

        individual.add(self.pos, next_cscore, self.q_values[self.pos].flatten(), action_pi0, next_x_pi0)
        self.c_buckets[group][next_cscore][idx] = individual

    def _compute_q_advs(self, save_vals=True):
        decomps = {}

        # Iterate through all individuals in the credit score buckets and compute their q-values.
        # The results will then be used to compute the empirical estimates for eqns 1, 3, and 4
        # as well as DPE, IPE, and SPE
        for g, group in enumerate(self.c_buckets):
            decomps[g] = {}
            x0_total = 0
            x0_counts = np.zeros(self.g_loc_start, dtype=np.uintc)
            x0_vpi = np.zeros(self.g_loc_start, dtype=np.float32)
            x0_vps = np.zeros(self.g_loc_start, dtype=np.float32)
            x0_vpi0 = np.zeros(self.g_loc_start, dtype=np.float32)

            for cscore in group.keys():
                
                for idx in group[cscore].keys():
                    individual = group[cscore][idx]
                    if individual.t_steps:
                        individual.compute_q_mc(self.gx[:,0], self.gx_pi0[:,0], self.q_advs[:,0], self.v_ps[:,0], self.actions[:,0,0].astype(int), save_vals=save_vals)
                        x0_counts[individual.c_scores[0]] += 1
                        x0_total += 1

                        x0_vpi[individual.c_scores[0]] += individual.q_advs[0]
                        x0_vps[individual.c_scores[0]] += individual.v_ps[0]
                        x0_vpi0[individual.c_scores[0]] += self.x_to_expectected_v_base[individual.group, individual.c_scores[0]]
            
            x0_vpi = np.divide(x0_vpi, x0_counts, out=np.zeros_like(x0_vpi), where=x0_counts!=0)
            x0_vps = np.divide(x0_vps, x0_counts, out=np.zeros_like(x0_vps), where=x0_counts!=0)
            x0_vpi0 = np.divide(x0_vpi0, x0_counts, out=np.zeros_like(x0_vpi0), where=x0_counts!=0)

            prob_x0 = x0_counts / x0_total
            
            g_de = np.sum((x0_vpi - x0_vps) * prob_x0)
            g_ie = np.sum((x0_vps - x0_vpi0) * prob_x0)
            g_se = np.sum(x0_vpi0 * prob_x0)
            g_c = np.sum(x0_vpi * prob_x0)

            g_init_gain = np.sum(self.cscore_to_init_qual_gain * prob_x0)
            decomps[g] = {'g_c': g_c, 'g_de': g_de, 'g_ie': g_ie, 'g_se': g_se, 
                           'g_init_gain': g_init_gain, 'avg_vx0': np.sum(x0_vpi * prob_x0),
                           'x0_vps': np.sum(x0_vps * prob_x0), 'x0_vpi0': np.sum(x0_vpi0 * prob_x0)}

        decomps['final'] = {}
        decomps['final']['c_pi_theta'] = decomps[0]['g_c'] - decomps[1]['g_c']
        decomps['final']['de']= decomps[0]['g_de'] - decomps[1]['g_de']
        decomps['final']['ie']= decomps[0]['g_ie'] - decomps[1]['g_ie']
        decomps['final']['se']= decomps[0]['g_se'] - decomps[1]['g_se']

        self.decomps = decomps

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
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

        self._compute_q_advs()

        last_gae_lam = 0
        
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

            group = 0 if self.observations[step][0][self.g_loc_start] >= 0.01 else 1

            self.expected_v_base[step] = self.x_to_expectected_v_base[group, self.curr_x[step].item()]

        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        deltas: th.Tensor,
        delta_deltas: th.Tensor,
        # ------------------------------- add new vars -----------------------
        long_delta: th.Tensor,
        gx: np.ndarray,
        q_log_prob: th.Tensor,
        curr_x: int,
        next_x: int,
        gx_pi0: np.ndarray,
        action_pi0: int,
        next_x_pi0: int,
        benefit_delta: float,
        # ------------------------------------------------------------------------
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
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.deltas[self.pos] = deltas.clone().cpu().numpy()
        self.delta_deltas[self.pos] = delta_deltas.clone().cpu().numpy()
        # ------------------------------- add one variable -----------------------
        self.long_delta[self.pos] = long_delta.clone().cpu().numpy()
        self.gx[self.pos] = np.array(gx).copy()
        self.q_log_prob[self.pos] = q_log_prob.clone().cpu().numpy().flatten()
        self.curr_x[self.pos] = np.array(curr_x).copy()
        self.next_x[self.pos] = np.array(next_x).copy()
        self.gx_pi0[self.pos] = np.array(gx_pi0).copy()
        self.benefit_deltas[self.pos] = np.array(benefit_delta).copy()

        self._sample_and_update_individual(action_pi0, next_x_pi0)
        # -------------------------------------------------------------------------
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "deltas",
                "delta_deltas",
                "long_delta",
                # -------------------------------------------------------------------------
                "q_log_prob",
                "q_advs",
                "v_ps",
                "benefit_deltas",
                "expected_v_base",
                # -------------------------------------------------------------------------
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

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.deltas[batch_inds].flatten(),
            self.delta_deltas[batch_inds].flatten(),
            self.long_delta[batch_inds].flatten(),
            # --------------------------------------------------------------
            self.q_log_prob[batch_inds],
            self.q_advs[batch_inds].flatten(),
            self.v_ps[batch_inds].flatten(),
            self.benefit_deltas[batch_inds].flatten(),
            self.expected_v_base[batch_inds].flatten(),
            # --------------------------------------------------------------
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
    

class DummyEvalBuffer:
    """
        Used for evaluation purposes when we don't want to use the full rollout buffer.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape,
        num_groups=2,
        n_envs=1,
        group_size=100,
        init_dists=None,
        qual_change_func=None,
    ):

        self.obs_shape = obs_shape
        self.observations = None,
        self.actions = None,
        self.n_envs = n_envs
        self.buffer_size = buffer_size
        self.pos = 0
        self.num_groups = num_groups
        self.group_size = group_size
        self.init_dists = init_dists
        self.g_loc_start = len(init_dists[0])
        self.action_dim = 1
        self.qual_change_func = qual_change_func

        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        # --------------------------------- add new part --------------------------------
        self.q_values = np.zeros((self.buffer_size, self.n_envs, 2), dtype=np.float32)
        self.q_advs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.q_targs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.gx = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.next_x = np.zeros((self.buffer_size, self.n_envs), dtype=np.uintc)
        self.curr_x = np.zeros((self.buffer_size, self.n_envs), dtype=np.uintc)
        self.gx_pi0 = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.v_ps = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.benefit_deltas = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.c_buckets = []
        start = 0
        for g in range(self.num_groups):
            g_idxs = [i for i in range(start,(g + 1) * self.group_size)]
            start = (g + 1) * self.group_size

            c_buckets = {}
            num_buckets = len(self.init_dists[g])

            g_start = 0
            for cscore in range(num_buckets):
                bucket_size = round(self.init_dists[g][cscore] * self.group_size)
                c_buckets[cscore] = {}
                for idx in g_idxs[g_start:g_start + bucket_size]:
                    c_buckets[cscore][idx] = Individual(idx, cscore, g)
                g_start += bucket_size

            assert g_start == self.group_size
            self.c_buckets.append(c_buckets)

        # -------------------------------------------------------------------------------

    def _sample_and_update_individual(self, action_pi0, next_x_pi0):
        cscore = self.curr_x[self.pos].item()
        next_cscore = self.next_x[self.pos].item()
        group = 0 if self.observations[self.pos][0][self.g_loc_start] != 0 else 1

        idx = random.choice(list(self.c_buckets[group][cscore].keys()))
        individual = self.c_buckets[group][cscore][idx]
        del self.c_buckets[group][cscore][idx]

        individual.add(self.pos, next_cscore, self.q_values[self.pos].flatten(), action_pi0, next_x_pi0)
        self.c_buckets[group][next_cscore][idx] = individual

    def _compute_q_advs(self, save_vals=False):

        for g, group in enumerate(self.c_buckets):
            for cscore in group.keys():
                for idx in group[cscore].keys():
                    individual = group[cscore][idx]
                    individual.compute_q_mc(self.gx[:,0], self.gx_pi0[:,0], self.q_advs[:,0], self.v_ps[:,0], self.actions[:,0,0].astype(int), save_vals=save_vals)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        # ------------------------------- add new vars -----------------------
        gx: np.ndarray,
        curr_x: int,
        next_x: int,
        gx_pi0: np.ndarray,
        action_pi0: int,
        next_x_pi0: int,
        benefit_delta: float,
        # -------------------------------------------------------------------------
    ) -> None:

        self.observations[self.pos] = np.array(obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        # ------------------------------- add one variable -----------------------
        self.gx[self.pos] = np.array(gx).copy()
        self.curr_x[self.pos] = np.array(curr_x).copy()
        self.next_x[self.pos] = np.array(next_x).copy()
        self.gx_pi0[self.pos] = np.array(gx_pi0).copy()
        self.benefit_deltas[self.pos] = np.array(benefit_delta).copy()

        self._sample_and_update_individual(np.array(action_pi0), np.array(next_x_pi0))
        # -------------------------------------------------------------------------
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True