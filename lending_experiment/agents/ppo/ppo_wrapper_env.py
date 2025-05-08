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
        zeta_0 = 1,
        zeta_1 = 0,
    ):
        super(PPOEnvWrapper, self).__init__(env)

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
        self.zeta_0 = zeta_0
        self.zeta_1 = zeta_1

        self.delta = np.zeros(1, )
        self.old_bank_cash = 0
        self.delta_delta = 0

        # my addition
        self.window = 200
        self.y_real_hist = [deque(maxlen=self.window) for _ in range(2)]
        self.y_pred_hist = [deque(maxlen=self.window) for _ in range(2)]
        self.a_hist = [deque(maxlen=self.window) for _ in range(2)]
        self.pred = 0
        self.mu = np.zeros(2,)
        self.mu_real = np.zeros(2,)

    def process_observation(self, obs):
        credit_score = obs['applicant_features']
        group = obs['group']

        return np.concatenate(
            (credit_score,
             group,
             self.mu,
             ),
            axis=0
        )

    
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
        self.mu = np.zeros(2,)
        self.mu_real = np.zeros(2,)

        return self.process_observation(self.env.reset())
    
    def compute_mu(self):
        if self.mu_type == "qualification":
            for i in range(2):
                self.mu_real[i] = np.mean(self.y_real_hist[i]) if len(self.y_real_hist[i]) > 0 else 1
                self.mu[i] = np.mean(self.y_pred_hist[i]) if len(self.y_pred_hist[i]) > 0 else 1
        elif self.mu_type == "accuracy":
            for i in range(2):
                y_real = np.array(self.y_real_hist[i])
                y_pred = np.array(self.y_pred_hist[i])
                action = np.array(self.a_hist[i])
                self.mu_real[i] = np.mean(y_real == action) if len(y_real) > 0 else 1
                self.mu[i] = np.mean(y_pred == action) if len(y_pred) > 0 else 1


    def step(self, action):
        old_delta = self.delta

        # Update instance variables before we step the environment
        group_id = np.argmax(self.env.state.group)
        
        self.old_bank_cash = self.env.state.bank_cash


        # my addition
        pred = self.pred if action == 0 else 1 - self.env.state.will_default
        self.y_real_hist[group_id].append(1 - self.env.state.will_default)
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
