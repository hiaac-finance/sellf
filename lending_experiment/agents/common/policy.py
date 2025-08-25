from typing import Tuple
import numpy as np
import gym

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        learning_rate: float,
        use_predictor: bool = False,
    ):
        super().__init__()
        self.use_predictor = use_predictor
        self.features_dim = np.array(observation_space.shape).prod()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.features_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.features_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 2), std=0.01),
        )
        self.predictor = nn.Sequential(
            layer_init(nn.Linear(self.features_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 2), std=0.01),
        )

        self.pred_optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=learning_rate
        )
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate,
            eps = 1e-5,
        )


    @property
    def device(self) -> torch.device:
        for param in self.parameters():
            return param.device
        return torch.device("cpu")

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, self.critic(x), probs.log_prob(action), probs.entropy(), 

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action

    def get_label(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_predictor:
            return torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        logits = self.predictor(x)
        probs = Categorical(logits=logits)
        label = probs.sample()
        return label

    def get_action_prob(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        # return only prob of action 1
        return probs.probs[:, 1]

    def get_label_prob(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_predictor:
            return torch.zeros(x.size(0), dtype=torch.float32, device=x.device)
        logits = self.predictor(x)
        probs = Categorical(logits=logits)
        # return only prob of label 1
        return probs.probs[:, 1]
