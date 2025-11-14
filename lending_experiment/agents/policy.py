from typing import Tuple
import numpy as np
import gym

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import lr_scheduler


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
        predictor: str = "linear",
        censor : float = 0.0,
    ):
        super().__init__()
        self.use_predictor = use_predictor
        self.censor = censor
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
        if predictor == "linear":
            self.predictor = nn.Sequential(
                layer_init(nn.Linear(self.features_dim, 1), std=0.01),
            )
        else:
            self.predictor = nn.Sequential(
                layer_init(nn.Linear(self.features_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=0.01),
            )
    

        self.pred_optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=1e-2,
        )
        self.pred_scheduler = lr_scheduler.ExponentialLR(
            self.pred_optimizer, gamma=0.95
        )
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate,
            eps=1e-5,
        )
        self.actor_history = []

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
            if probs.probs[0, 1] < self.censor:
                action = torch.tensor(0).to(self.device)
            else:
                action = probs.sample()
        return (
            action,
            self.critic(x),
            probs.log_prob(action),
            probs.entropy(),
        )

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        if probs.probs[0, 1] < self.censor:
            action = torch.tensor(0).to(self.device)
        return action

    def get_label(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_predictor:
            if x.dim == 1:
                x = x.unsqueeze(0)
            return torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        log_odds = self.predictor(x)
        probs = torch.sigmoid(log_odds)
        p = torch.rand((x.shape[0]), dtype=torch.float32, device=self.device)
        labels = (probs.reshape(-1) > p).float()
        return labels

    def get_action_prob(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        # return only prob of action 1
        return probs.probs[:, 1]

    def get_label_prob(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_predictor:
            return torch.zeros(x.size(0), dtype=torch.float32, device=x.device)
        log_odds = self.predictor(x)
        probs = torch.sigmoid(log_odds)
        return probs

    def save_history(self) -> None:
        """Add current actor to history and keep only the last 10."""
        self.actor_history.append(
            nn.Sequential(
                layer_init(nn.Linear(self.features_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 2), std=0.01),
            ).to(self.device)
        )
        self.actor_history[-1].load_state_dict(self.actor.state_dict())

    def get_action_all_prob(self, x: torch.Tensor) -> torch.Tensor:
        selected_actors = np.random.choice(
            len(self.actor_history), min(len(self.actor_history), 10), replace=False
        )
        rej = torch.ones_like(x[:, 0])
        for i in selected_actors:
            actor = self.actor_history[i]
            logits = actor(x)
            probs = Categorical(logits=logits)
            rej = rej * (1 - probs.probs[:, 1])
        return 1 - rej
