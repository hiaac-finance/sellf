import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch
import torch as th
import gym
from gym import spaces
from torch.nn import functional as F
import time

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import explained_variance


from lending_experiment.agents.on_policy_algorithm import OnPolicyAlgorithm


class RRM(OnPolicyAlgorithm):
    def __init__(
        self,
        env: Union[gym.Env, VecEnv],
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        beta_0: float = 0.5,
        omega: float = 0.1,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
    ):

        super(RRM, self).__init__(
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            device=device,
            seed=seed,
        )

        if hasattr(env, "utility_method"):
            self.utility_method = env.utility_method
        else:
            self.utility_method = env.get_attr("utility_method")[0]

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.beta_0 = beta_0

        self._setup_model()

    def compute_fair_penalization(
        self, loss: th. Tensor, labels: th.Tensor, groups: th.Tensor
    ) -> th.Tensor:
        g0_idx = (groups[:, 0] == 1).nonzero()
        g1_idx = (groups[:, 1] == 1).nonzero()
        g0_loss = loss[g0_idx]
        g1_loss = loss[g1_idx]

        if g0_idx.shape[0] == 0 or g1_idx.shape[0] == 0:
            delta = torch.Tensor([0.0]).to(loss.device)
        elif self.utility_method == "qualification":
            delta = torch.Tensor([0.0]).to(loss.device)
        elif self.utility_method == "accuracy":
            delta = (g0_loss.mean() - g1_loss.mean()) ** 2
        elif self.utility_method == "tpr":
            g0_loss = g0_loss[labels[g0_idx] == 1]
            g1_loss = g1_loss[labels[g1_idx] == 1]
            if g0_loss.shape[0] == 0 or g1_loss.shape[0] == 0:
                delta = torch.Tensor([0.0]).to(loss.device)
            else:
                delta = (g0_loss.mean() - g1_loss.mean()) ** 2
        return delta

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.train()
        criterion = torch.nn.BCELoss(reduction = "none")
        pred_losses = []

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                labels = rollout_data.labels.reshape(-1)
                probs = self.policy.get_action_prob(rollout_data.observations)

                # calculate loss on accepted population
                labels = labels[actions == 1]
                probs = probs[actions == 1]
                groups = rollout_data.groups[actions == 1]

                loss = criterion(probs, labels)

                fairness_cost = self.compute_fair_penalization(
                    loss, labels, groups
                )
                loss = loss.mean() + self.beta_0 * fairness_cost

                self.policy.optimizer.zero_grad()
                loss.backward()
                self.policy.optimizer.step()

                pred_losses.append(loss.item())

        # Logs
        self.logger.record("train/loss", np.mean(pred_losses))
        self.logger.record(
            "train/accept_rate", np.mean(self.rollout_buffer.actions.flatten())
        )
        self.logger.record(
            "train/pos_rate", np.mean(self.rollout_buffer.labels.flatten())
        )
        self.logger.record("train/reward", self.rollout_buffer.rewards.mean().item())

        # Logs some group-dependent variables
        g0_idx = (self.rollout_buffer.groups[:, 0] == 1).nonzero()
        g1_idx = (self.rollout_buffer.groups[:, 1] == 1).nonzero()

        accept_rate = [
            self.rollout_buffer.actions[g0_idx, 0].mean().item(),
            self.rollout_buffer.actions[g1_idx, 0].mean().item(),
        ]

        accuracy = (
            (self.rollout_buffer.labels[:, 0] == self.rollout_buffer.preds[:, 0])
            .mean()
            .item()
        )

        self.logger.record("train/accept_g0", accept_rate[0])
        self.logger.record("train/accept_g1", accept_rate[1])
        self.logger.record("train/delta", self.rollout_buffer.deltas.mean().item())
        self.logger.record(
            "train/delta_real", self.rollout_buffer.delta_reals.mean().item()
        )
        self.logger.record("train/accuracy", accuracy)
