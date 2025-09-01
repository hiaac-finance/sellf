import gym
import numpy as np
import torch
from gym import spaces


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
                env.observation_space["applicant_features"].shape[0] + env.n_groups,
            ),
        )

        self.action_space = spaces.Discrete(n=2)
        self.env = env
        self.num_applicants = len(self.env.pool)
        self.env.get_label_pred = self._get_label_pred
        self.env.get_action = self._get_action
        self.env.get_action_prob = self._get_action_prob

    def set_agent(self, agent):
        self.policy = agent.policy

    def _get_label_pred(self, features, group):
        obs = {
            "applicant_features": features,
            "group": group,
        }
        obs = self.process_observation(obs)
        obs = np.array(obs).reshape(1, -1)
        obs = torch.tensor(obs, dtype=torch.float32).to(self.policy.device)
        with torch.no_grad():
            pred = self.policy.get_label(obs).cpu().numpy()[0]
        return pred

    def _get_action(self, features, group):
        obs = {
            "applicant_features": features,
            "group": group,
        }
        obs = self.process_observation(obs)
        obs = np.array(obs).reshape(1, -1)
        obs = torch.tensor(obs, dtype=torch.float32).to(self.policy.device)
        with torch.no_grad():
            pred = self.policy.get_action(obs).cpu().numpy()[0]
        return pred

    def _get_action_prob(self, features, group):
        obs = {
            "applicant_features": features,
            "group": group,
        }
        obs = self.process_observation(obs)
        obs = np.array(obs).reshape(1, -1)
        obs = torch.tensor(obs, dtype=torch.float32).to(self.policy.device)
        with torch.no_grad():
            pred = self.policy.get_action_prob(obs).cpu().numpy()[0]
        return pred

    def _get_action_prob_list(self, features, group):
        obs = {
            "applicant_features": features,
            "group": group,
        }
        obs = self.process_observation(obs)
        obs = np.array(obs).reshape(1, -1)
        obs = torch.tensor(obs, dtype=torch.float32).to(self.policy.device)
        pred_list = []
        with torch.no_grad():
            for policy in self.policy_hist:
                pred = policy.get_action_prob(obs).cpu().numpy()[0]
                pred_list.append(pred)
        return pred_list

    def process_observation(self, obs):
        credit_score = obs["applicant_features"]
        group = obs["group"]
        # if group is scalar, transform to array with 2 values
        if np.isscalar(group):
            group_aux = np.zeros(2)
            group_aux[group] = 1
            group = group_aux
        return np.concatenate((credit_score, group), axis=0)

    def reset(self):
        return self.process_observation(self.env.reset())

    def step(self, action):
        old_resource = self.env.resource
        obs, _, done, info = self.env.step(action)
        r = self.env.resource - old_resource
        return self.process_observation(obs), r, done, info
