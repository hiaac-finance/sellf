import gym
import numpy as np
import torch
from gym import spaces
from collections import deque

from stable_baselines3.common.monitor import Monitor as SBMonitor

class Monitor(SBMonitor):
    def __init__(self, env, filename=None, allow_early_resets=True):
        super(Monitor, self).__init__(env, filename, allow_early_resets)

    
    # ugly code to have the pred attribute
    @property
    def pred(self):
        return getattr(self.env, 'pred', 0)

    @pred.setter
    def pred(self, value):
        """
        Set the prediction value in the environment.
        """
        if hasattr(self.env, 'pred'):
            self.env.pred = value
        else:
            raise AttributeError("Environment does not have 'pred' attribute")
        
    @property
    def prob_accept(self):
        return getattr(self.env, 'prob_accept', 1)
    
    @prob_accept.setter
    def prob_accept(self, value):
        """
        Set the probability of acceptance in the environment.
        """
        if hasattr(self.env, 'prob_accept'):
            self.env.prob_accept = value
        else:
            raise AttributeError("Environment does not have 'prob_accept' attribute")

class PPOEnvWrapper(gym.Wrapper):
    def __init__(self,
        env,
        reward_fn,
        ep_timesteps=2000,
        mu_type = "qualification",
        obs_type = "imputation",
        zeta_0 = 1,
        zeta_1 = 0,
    ):
        super(PPOEnvWrapper, self).__init__(env)
        assert mu_type in ["qualification", "accuracy", "tpr"], f"mu_type {mu_type} not supported"
        assert obs_type in ["imputation", "accepted", "full"], f"obs_type {obs_type} not supported"

        self.observation_space = spaces.Box(
            low=np.inf,
            high=np.inf,
            # (7) OHE of credit score + (2) group +    (2) mu of each group
            shape=(env.observation_space['applicant_features'].shape[0] + 2 * env.state.params.num_groups,),
        )

        self.action_space = spaces.Discrete(n=2)

        self.env = env
        self.reward_fn = reward_fn()

        self.timestep = 0
        self.ep_timesteps = ep_timesteps
        self.mu_type = mu_type
        self.obs_type = obs_type
        self.zeta_0 = zeta_0
        self.zeta_1 = zeta_1

        self.delta = np.zeros(1, )
        self.old_bank_cash = 0
        self.delta_delta = 0

        # my addition
        self.window = 100
        self.y_real_hist = [deque(maxlen=self.window) for _ in range(2)]
        self.y_pred_hist = [deque(maxlen=self.window) for _ in range(2)]
        self.pred_hist = [deque(maxlen=self.window) for _ in range(2)]
        self.a_hist = [deque(maxlen=self.window) for _ in range(2)]
        self.pred = 0
        self.mu = np.zeros(2,)
        self.mu_real = np.zeros(2,)
        self.rejection = np.zeros(2,)
        self.error_rejection = np.zeros(2,)
        self.b_term = np.zeros(2,)

    def process_observation(self, obs):
        credit_score = obs['applicant_features']
        group = obs['group']

        return np.concatenate((credit_score, group, self.mu,), axis=0)

    def reset(self):
        self.timestep = 0
        self.delta = np.zeros(1, )
        self.old_bank_cash = 0
        self.delta_delta = 0    
        self.delta_b_term = 0


        # my addiition
        self.y_real_hist = [deque(maxlen=self.window) for _ in range(2)]
        self.y_pred_hist = [deque(maxlen=self.window) for _ in range(2)]
        self.pred_hist = [deque(maxlen=self.window) for _ in range(2)]
        self.a_hist = [deque(maxlen=self.window) for _ in range(2)]
        self.pred = 0
        self.mu = np.ones(2,)
        self.mu_real = np.ones(2,)
        self.rejection = np.zeros(2,)
        self.error_rejection = np.zeros(2,)
        self.b_term = np.zeros(2,)

        return self.process_observation(self.env.reset())
    
    def compute_mu(self):
        for i in range(2):
            y_real = np.array(self.y_real_hist[i])
            y_pred = np.array(self.y_pred_hist[i])
            pred = np.array(self.pred_hist[i])
            action = np.array(self.a_hist[i])


            # calculate first mu_real
            if self.mu_type == "qualification":
                self.mu_real[i] = y_real.mean() if len(y_real) > 0 else 1
            elif self.mu_type == "accuracy":
                self.mu_real[i] = (y_real == action).mean() if len(y_real) > 0 else 1
            elif self.mu_type == "tpr":
                self.mu_real[i] = np.mean(action[y_real == 1]) if y_real.sum() > 0 else 1
            # now, calculate mu
            if self.obs_type == "imputation":
                if self.mu_type == "qualification":
                    self.mu[i] = y_pred.mean() if len(y_pred) > 0 else 1
                elif self.mu_type == "accuracy":
                    self.mu[i] = (y_pred == action).mean() if len(y_pred) > 0 else 1
                elif self.mu_type == "tpr":
                    self.mu[i] = np.mean(action[y_pred == 1]) if y_pred.sum() > 0 else 1
            elif self.obs_type == "accepted":
                if self.mu_type == "qualification":
                    self.mu[i] = y_pred[action == 1].mean() if (action == 1).sum() > 0 else 1
                elif self.mu_type == "accuracy":
                    self.mu[i] = (y_pred[action == 1] == 1).mean() if (action == 1).sum() > 0 else 1
                elif self.mu_type == "tpr":
                    self.mu[i] = 1
            elif self.obs_type == "full":
                self.mu[i] = self.mu_real[i]

            # calculate rejection terms
            self.rejection[i] = np.mean(action == 0) if len(action) > 0 else 0
            error = pred - y_real
            # calculate error in the accepted group
            self.error_rejection[i] = np.mean(error[action == 1]) if (action).sum() > 0 else 0
            if self.mu_type == "qualification" or self.mu_type == "accuracy":
                self.b_term[i] = self.rejection[i] * self.error_rejection[i]
            else:
                if len(y_pred) == 0 or np.mean(y_pred) == 0:
                    self.b_term[i] = 1
                else:
                    self.b_term[i] = 1 - self.rejection[i] * self.error_rejection[i] / np.mean(y_pred)
            


    def step(self, action):
        old_delta = self.delta

        # Update instance variables before we step the environment
        group_id = np.argmax(self.env.state.group)
        self.old_bank_cash = self.env.state.bank_cash
        label = 1 - self.env.state.will_default

        if self.obs_type == "imputation":
            pred = self.pred if action == 0 else label
        else:
            pred = label

        self.y_real_hist[group_id].append(label)
        self.y_pred_hist[group_id].append(pred)
        self.pred_hist[group_id].append(self.pred)
        self.a_hist[group_id].append(action)
        self.compute_mu()

        self.delta = np.abs(self.mu[0] - self.mu[1])
        self.delta_real = np.abs(self.mu_real[0] - self.mu_real[1])
        self.delta_delta = self.delta - old_delta
        self.delta_b_term = np.abs(self.b_term[0] - self.b_term[1])


        obs, _, done, info = self.env.step(action)

        r = self.reward_fn(
            old_bank_cash=self.old_bank_cash,
            bank_cash=self.env.state.bank_cash,
            tpr=self.mu,
            zeta0=self.zeta_0,
            zeta1=self.zeta_1
        )

        self.timestep += 1
        if self.timestep >= self.ep_timesteps:
            done = True

        return self.process_observation(obs), r, done, info
