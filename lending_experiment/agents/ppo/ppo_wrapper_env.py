import gym
import numpy as np
import torch
from gym import spaces

from stable_baselines3.common.monitor import Monitor as SBMonitor


class Monitor(SBMonitor):
    def __init__(self, env, filename=None, allow_early_resets=True):
        super(Monitor, self).__init__(env, filename, allow_early_resets)

    # ugly code to have the pred attribute
    @property
    def pred(self):
        return getattr(self.env, "pred", 0)

    @pred.setter
    def pred(self, value):
        """
        Set the prediction value in the environment.
        """
        if hasattr(self.env, "pred"):
            self.env.pred = value
        else:
            raise AttributeError("Environment does not have 'pred' attribute")

    @property
    def prob_accept(self):
        return getattr(self.env, "prob_accept", 1)

    @prob_accept.setter
    def prob_accept(self, value):
        """
        Set the probability of acceptance in the environment.
        """
        if hasattr(self.env, "prob_accept"):
            self.env.prob_accept = value
        else:
            raise AttributeError("Environment does not have 'prob_accept' attribute")

    @property
    def prob_predict(self):
        return getattr(self.env, "prob_predict", 1)

    @prob_predict.setter
    def prob_predict(self, value):
        """
        Set the probability of prediction in the environment.
        """
        if hasattr(self.env, "prob_predict"):
            self.env.prob_predict = value
        else:
            raise AttributeError("Environment does not have 'prob_predict' attribute")


class PPOEnvWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
    ):
        super(PPOEnvWrapper, self).__init__(env)

        self.observation_space = spaces.Box(
            low=np.inf,
            high=np.inf,
            shape=(
                env.observation_space["applicant_features"].shape[0]
                + env.state.params.num_groups,
            ),
        )

        self.action_space = spaces.Discrete(n=2)
        self.env = env
        self.num_applicants = len(self.env.pool)
        self.env.predict_fn = self._predict_fn

    def set_agent(self, agent):
        self.policy = agent.policy

    def _predict_fn(self, applicant):
        obs = {
            "applicant_features": applicant["features"],
            "group": applicant["group"],
        }
        obs = self.process_observation(obs)
        obs = np.array(obs).reshape(1, -1)
        obs = torch.tensor(obs, dtype=torch.float32).to(self.policy.device)
        with torch.no_grad():
            pred = self.policy.get_label(obs).cpu().numpy()[0]
        return pred

    def get_applicant_obs(self, idx):
        """Get the observation for a specific applicant."""
        applicant = self.env.pool[idx]
        obs = {
            "applicant_features": applicant["features"],
            "group": applicant["group"],
        }
        return self.process_observation(obs)

    def process_observation(self, obs):
        credit_score = obs["applicant_features"]
        group = obs["group"]
        return np.concatenate((credit_score, group), axis=0)

    def reset(self):
        return self.process_observation(self.env.reset())

    def step(self, action):
        old_resource = self.env.state.resource
        obs, _, done, info = self.env.step(action)
        r = self.env.state.resource - old_resource
        return self.process_observation(obs), r, done, info
