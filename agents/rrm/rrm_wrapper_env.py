import gym
import numpy as np
import torch
from gym import spaces
from copy import deepcopy
from collections import deque

from environments.distributions import Bernoulli



class RRMEnvWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        reward_fn,
        omega: float = 0.0,
        dist_test: bool = True,
        mu_type: str = "error",
        penalize_mu: bool = False,
        window: int = 50,
        pop_window: int = 300,
        ep_timesteps: int = 500,
        ):
        super(RRMEnvWrapper, self).__init__(env)

        self.omega = omega
        self.mu_type = mu_type
        self.penalize_mu = penalize_mu
        self.window = window
        self.pop_window = pop_window
        self.ep_timesteps = ep_timesteps

        self.y_hist = [
            deque(maxlen = self.window),
            deque(maxlen = self.window),
        ]
        self.y_obs_hist = [
            deque(maxlen = self.window),
            deque(maxlen = self.window),
        ]
        self.a_hist = [
            deque(maxlen = self.window),
            deque(maxlen = self.window),
        ]
        self.timestep = 0
        self.old_bank_cash = 0
        self.population= deque(maxlen=self.pop_window)

        self.dist_est = dist_test
        if self.dist_est:
                self.observation_space = spaces.Box(
                low=np.inf,
                high=np.inf,
                shape=(3 * env.observation_space['applicant_features'].shape[0] + env.state.params.num_groups,),
            )
        else:
            self.observation_space = spaces.Box(
                low=np.inf,
                high=np.inf,
                shape=(env.observation_space['applicant_features'].shape[0] + env.state.params.num_groups,),
                # -------------------------------------------------------------
            )
        self.action_space = spaces.Discrete(n=2)

        self.env = env
        self.reward_fn = reward_fn(omega = omega)


    def process_observation(self, obs) -> np.ndarray:
        credit_score = obs['applicant_features']
        group = obs['group']
        hist0 = self.history[0]
        hist1 = self.history[1]
        norm = sum(hist0 + hist1) + 1.

        if self.dist_est:
            return np.concatenate(
                (credit_score,
                group,
                hist0 / norm,
                hist1 / norm,
                ),
                axis=0
            )
        else:
            return np.concatenate(
                (credit_score,
                group,
                ),
                axis=0
            )

    def reset(self) -> np.ndarray:
        self.y_hist = [
            deque(maxlen = self.window),
            deque(maxlen = self.window),
        ]
        self.y_obs_hist = [
            deque(maxlen = self.window),
            deque(maxlen = self.window),
        ]
        self.a_hist = [
            deque(maxlen = self.window),
            deque(maxlen = self.window),
        ]
        self.timestep = 0
        self.old_bank_cash = 0
        self.population= deque(maxlen=self.pop_window)
        self.history = np.zeros((self.env.state.params.num_groups, self.env.observation_space['applicant_features'].shape[0]))
        self.mu = np.zeros(self.env.state.params.num_groups)
        self.mu_obs = np.zeros(self.env.state.params.num_groups)
        self.delta = 0
        self.delta_obs = 0

        return self.process_observation(self.env.reset())
    

    def step(self, action):
        group_id = int(np.argmax(self.env.state.group))
        self.y_hist[group_id].append(self.env.state.y)
        self.y_obs_hist[group_id].append(self.env.state.y_obs)
        self.a_hist[group_id].append(action)

        # calculate mu and delta
        if self.mu_type == "error":
            self.mu = [
                np.mean(np.array(self.y_hist[0]) != np.array(self.a_hist[0])) if len(self.y_hist[0]) > 0 else 1,
                np.mean(np.array(self.y_hist[1]) != np.array(self.a_hist[1])) if len(self.y_hist[1]) > 0 else 1,
            ]
            self.delta = np.abs(self.mu[0] - self.mu[1])

            self.mu_obs = [
                np.mean(np.array(self.y_obs_hist[0]) != np.array(self.a_hist[0])) if len(self.y_obs_hist[0]) > 0 else 1,
                np.mean(np.array(self.y_obs_hist[1]) != np.array(self.a_hist[1])) if len(self.y_obs_hist[1]) > 0 else 1,
            ]
            self.delta_obs = np.abs(self.mu_obs[0] - self.mu_obs[1])
        elif self.mu_type == "tpr":
            for i in range(2):
                if np.sum(self.y_hist[i]) > 0:
                    self.mu[i] = np.mean([self.a_hist[i][j] for j in range(len(self.y_hist[i])) if self.y_hist[i][j] == 1])
                else:
                    self.mu[i] = 1

                if np.sum(self.y_obs_hist[i]) > 0:
                    self.mu_obs[i] = np.mean([self.a_hist[i][j] for j in range(len(self.y_obs_hist[i])) if self.y_obs_hist[i][j] == 1])
                else:
                    self.mu_obs[i] = 1

            self.delta = np.abs(self.mu[0] - self.mu[1])
            self.delta_obs = np.abs(self.mu_obs[0] - self.mu_obs[1])

        elif self.mu_type == "quali":
            for i in range(2):
                self.mu[i] = np.mean(self.y_hist[i]) if len(self.y_hist[i]) > 0 else 1
                self.mu_obs[i] = np.mean(self.y_obs_hist[i]) if len(self.y_obs_hist[i]) > 0 else 1
            self.delta = np.abs(self.mu[0] - self.mu[1])
            self.delta_obs = np.abs(self.mu_obs[0] - self.mu_obs[1])

        
        self.old_bank_cash = self.env.state.bank_cash

        if len(self.population) == self.pop_window: #if population is full, remove the oldest
            old_id, old_feats, old_default, old_action = self.population.popleft()
            self.history[old_id] -= old_feats

        state_feats = deepcopy(self.env.state.applicant_features)
        state_default = deepcopy(self.env.state.y == 0)
        self.history[group_id] = self.history[group_id] + state_feats
        self.population.append((group_id, state_feats, state_default, action))

        # -------------------------------------------------------------------------
        obs, _, done, info = self.env.step(action)

        r = self.reward_fn(
                old_bank_cash=self.old_bank_cash,
                bank_cash=self.env.state.bank_cash,
                mu=self.mu,
                zeta0=1 if not self.penalize_mu else 1- self.timestep/self.ep_timesteps,
                zeta1=0 if not self.penalize_mu else self.timestep/self.ep_timesteps,
        )

        self.timestep += 1
        if self.timestep >= self.ep_timesteps:
            done = True

        obs = self.process_observation(obs)

        # ---------------------------------------------
        return obs, r, done, info