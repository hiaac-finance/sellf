import gym
import numpy as np
import torch
from gym import spaces
from collections import deque

class PPOEnvWrapper(gym.Wrapper):
    def __init__(self,
        env,
        reward_fn,
        ep_timesteps=2000,
        mu_type = "qualification",
        delta_type="imputation",
        partial_observation=True,
        zeta_0 = 1,
        zeta_1 = 0,
    ):
        super(PPOEnvWrapper, self).__init__(env)
        assert mu_type in ["qualification", "accuracy", "tpr"], f"mu_type {mu_type} not supported"
        assert delta_type in ["imputation", "accepted"], f"delta_type {delta_type} not supported"

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
        self.delta_type = delta_type
        self.partial_observation = partial_observation
        self.zeta_0 = zeta_0
        self.zeta_1 = zeta_1

        self.delta = np.zeros(1, )
        self.old_bank_cash = 0
        self.delta_delta = 0

        # my addition
        self.window = 300
        self.y_real_hist = [deque(maxlen=self.window) for _ in range(2)]
        self.y_pred_hist = [deque(maxlen=self.window) for _ in range(2)]
        self.a_hist = [deque(maxlen=self.window) for _ in range(2)]
        self.pred = 0
        self.mu = np.zeros(2,)
        self.mu_real = np.zeros(2,)

    def process_observation(self, obs):
        credit_score = obs['applicant_features']
        group = obs['group']

        return np.concatenate((credit_score, group, self.mu,), axis=0)

    def reset(self):
        self.timestep = 0
        self.delta = np.zeros(1, )
        self.old_bank_cash = 0
        self.delta_delta = 0    


        # my addiition
        self.y_real_hist = [deque(maxlen=self.window) for _ in range(2)]
        self.y_pred_hist = [deque(maxlen=self.window) for _ in range(2)]
        self.a_hist = [deque(maxlen=self.window) for _ in range(2)]
        self.pred = 0
        self.mu = np.ones(2,)
        self.mu_real = np.ones(2,)

        return self.process_observation(self.env.reset())
    
    def compute_mu(self):
        for i in range(2):
            y_real = np.array(self.y_real_hist[i])
            y_pred = np.array(self.y_pred_hist[i])
            action = np.array(self.a_hist[i])

            # first compute mu_real
            if self.mu_type == "qualification":
                self.mu_real[i] = y_real.mean() if len(y_real) > 0 else 1
            elif self.mu_type == "accuracy":
                self.mu_real[i] = (y_real == action).mean() if len(y_real) > 0 else 1
            elif self.mu_type == "tpr":
                self.mu_real[i] = np.mean(action[y_real == 1]) if y_real.sum() > 0 else 1

            
            # then compute mu
            if self.mu_type == "qualification":
                if self.delta_type == "imputation":
                    self.mu[i] = y_pred.mean() if len(y_pred) > 0 else 1
                elif self.delta_type == "accepted":
                    self.mu[i] = np.mean(y_pred[action == 1]) if action.sum() > 0 else 1
            elif self.mu_type == "accuracy":
                if self.delta_type == "imputation":
                    self.mu[i] = (y_pred == action).mean() if len(y_pred) > 0 else 1
                elif self.delta_type == "accepted":
                    self.mu[i] = np.mean(y_pred[action == 1] == action[action == 1]) if action.sum() > 0 else 1
            elif self.mu_type == "tpr":
                if self.delta_type == "imputation":
                    self.mu[i] = np.mean(action[y_pred == 1]) if y_pred.sum() > 0 else 1
                elif self.delta_type == "accepted":
                    self.mu[i] = 1 # is always 1
        


    def step(self, action):
        old_delta = self.delta

        # Update instance variables before we step the environment
        group_id = np.argmax(self.env.state.group)
        
        self.old_bank_cash = self.env.state.bank_cash

        label = 1 - self.env.state.will_default
        if self.partial_observation:
            pred = self.pred if action == 0 else label
        else:
            pred = label
        self.y_real_hist[group_id].append(1 - label)
        self.y_pred_hist[group_id].append(pred)
        self.a_hist[group_id].append(action)
        self.compute_mu()

        self.delta = np.abs(self.mu[0] - self.mu[1])
        self.delta_real = np.abs(self.mu_real[0] - self.mu_real[1])
        self.delta_delta = self.delta - old_delta


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
