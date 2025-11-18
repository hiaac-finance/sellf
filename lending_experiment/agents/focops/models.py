import numpy as np
import torch.nn as nn
import torch
from torch.distributions import Normal, Categorical


def mlp(input_size, hidden_sizes=(64, 64), activation='tanh'):
    if activation == 'tanh':
        activation = nn.Tanh
    elif activation == 'relu':
        activation = nn.ReLU
    elif activation == 'sigmoid':
        activation = nn.Sigmoid

    layers = []
    sizes = (input_size, ) + hidden_sizes
    for i in range(len(hidden_sizes)):
        layers += [nn.Linear(sizes[i], sizes[i+1]), activation()]
    return nn.Sequential(*layers)



class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), activation='tanh', log_std=-0.5):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.mlp_net = mlp(obs_dim, hidden_sizes, activation)
        self.mean_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.logstd_layer = nn.Parameter(torch.ones(1, act_dim) * log_std)

        self.mean_layer.weight.data.mul_(0.1)
        self.mean_layer.bias.data.mul_(0.0)

    def forward(self, obs):

        out = self.mlp_net(obs)
        mean = self.mean_layer(out)
        if len(mean.size()) == 1:
            mean = mean.view(1, -1)
        logstd = self.logstd_layer.expand_as(mean)
        std = torch.exp(logstd)

        return mean, logstd, std

    def get_act(self, obs, deterministic = False):
        mean, _, std = self.forward(obs)
        if deterministic:
            return mean
        else:
            return torch.normal(mean, std)

    def logprob(self, obs, act):
        mean, _, std = self.forward(obs)
        normal = Normal(mean, std)
        return normal.log_prob(act).sum(-1, keepdim=True), mean, std



class Value(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(64, 64), activation='tanh'):
        super().__init__()

        self.obs_dim = obs_dim

        self.mlp_net = mlp(obs_dim, hidden_sizes, activation)
        self.v_head = nn.Linear(hidden_sizes[-1], 1)

        self.v_head.weight.data.mul_(0.1)
        self.v_head.bias.data.mul_(0.0)

    def forward(self, obs):
        mlp_out = self.mlp_net(obs)
        v_out = self.v_head(mlp_out)
        return v_out

class CategoricalPolicy(nn.Module):
    """
    Policy for discrete actions; here: binary, so logits dim = 2
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), activation='tanh'):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim  # for binary, act_dim = 2

        self.mlp_net = mlp(obs_dim, hidden_sizes, activation)
        self.logits_layer = nn.Linear(hidden_sizes[-1], act_dim)

        self.logits_layer.weight.data.mul_(0.1)
        self.logits_layer.bias.data.mul_(0.0)
        self.device = torch.device("cpu")

    def forward(self, obs):
        """
        Returns logits for each discrete action.
        obs: [B, obs_dim]
        logits: [B, act_dim]
        """
        out = self.mlp_net(obs)
        logits = self.logits_layer(out)
        return logits

    def get_act(self, obs, deterministic=False):
        """
        obs: 1D or 2D tensor
        Returns: integer action (0 or 1 for binary)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        logits = self.forward(obs)
        dist = Categorical(logits=logits)

        if deterministic:
            # Greedy action: argmax over logits
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        return action.squeeze(0)  # return scalar for single state

    def logprob(self, obs, act):
        """
        obs: [B, obs_dim]
        act: [B] (LongTensor) with values in {0, 1} for binary

        Returns:
            logprob: [B, 1]
            logits: [B, act_dim]
        """
        logits = self.forward(obs)
        dist = Categorical(logits=logits)

        if act.dim() == 2 and act.size(-1) == 1:
            # If actions come in as [B, 1], squeeze to [B]
            act = act.squeeze(-1)
        act = act.long()

        logprob = dist.log_prob(act).unsqueeze(-1)  # [B, 1]
        return logprob, logits
    

    def get_label(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim == 1:
            x = x.unsqueeze(0)
        return torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action