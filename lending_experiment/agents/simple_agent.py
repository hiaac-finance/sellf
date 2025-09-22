import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch
import torch as th
import gym
from gym import spaces
from torch.nn import functional as F
import time

from stable_baselines3.common.utils import explained_variance


from agents.on_policy_algorithm import OnPolicyAlgorithm


class SimpleAgent(OnPolicyAlgorithm):
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 1e-5,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        decision_type: str = "random",
        **kwargs,
    ):

        super(SimpleAgent, self).__init__(
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            device=device,
            seed=seed,
        )

        assert decision_type in [
            "random",
            "all_approve",
            "all_deny",
            "accept_g0",
            "accept_g1",
            "min_max",
        ]
        self.decision_type = decision_type
        self._setup_model()

    def learn(
        self,
        total_timesteps: int,
    ) -> "SimpleAgent":
        # no training is necessary for this simple agent
        return self

    def get_action(self, observation):
        if self.decision_type == "random":
            action = self.action_space.sample()
        elif self.decision_type == "all_approve":
            action = 1
        elif self.decision_type == "all_deny":
            action = 0
        elif self.decision_type == "accept_g0":
            group = observation[0, -2].item()
            action = 1 if group == 0 else 0
        elif self.decision_type == "accept_g1":
            group = observation[0, -1].item()
            action = 1 if group == 0 else 0
        elif self.decision_type == "min_max":
            # accept if group is the one with lowest utility
            group = observation[0, -2:].argmax().item()
            utility = self.env.utility_values
            action = 1 if utility[group] == np.min(utility) else 0
        action = th.tensor([action], device=self.device)
        return action
