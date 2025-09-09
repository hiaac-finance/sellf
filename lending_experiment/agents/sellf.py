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


class SELLF(OnPolicyAlgorithm):
    def __init__(
        self,
        env: Union[gym.Env, VecEnv],
        learning_rate: float = 1e-5,
        beta_0: float = 1,
        beta_1: float = 0.5,
        beta_2: float = 0.5,
        beta_3: float = 0.5,
        omega: float = 0.1,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        normalize_advantage: bool = True,
        ent_coef: float = 0.2,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        **kwargs,
    ):

        super(SELLF, self).__init__(
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

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            if hasattr(env, "utility_method"):
                self.utility_method = env.utility_method
            else:
                self.utility_method = env.get_attr("utility_method")[0]

        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.omega = omega
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.predictor_steps = 300

        self._setup_model()

    def train_predictor(self) -> None:
        """Train the predictor model using the accepted data from rollout buffer."""

        pred_criterion = th.nn.BCELoss(reduction="none")

        # add rollout data to memory
        actions = self.rollout_buffer.actions
        obs = self.rollout_buffer.observations[actions[:, 0, 0] == 1]
        labels = self.rollout_buffer.labels[actions[:, 0, 0] == 1]

        self.memory.add(obs=obs, label=labels)

        losses_hist = []
        steps = 0
        for epoch in range(100):
            for i, rollout_data in enumerate(self.memory.get(self.batch_size)):
                preds = self.policy.get_label_prob(rollout_data.observations)
                with th.no_grad():
                    prob_loan = self.policy.get_action_prob(rollout_data.observations)
                    prob_loan = th.clamp(prob_loan, min=0.05, max=0.95)

                pred_loss = pred_criterion(preds, rollout_data.labels)
                pred_loss = pred_loss * (1 / prob_loan)
                pred_loss = pred_loss.sum() / (1 / prob_loan).sum()
                losses_hist.append(pred_loss.item())

                self.policy.pred_optimizer.zero_grad()
                pred_loss.backward()
                self.policy.pred_optimizer.step()

                steps += 1

                if steps >= self.predictor_steps:
                    break
            
            self.policy.pred_scheduler.step()
            if steps >= self.predictor_steps:
                break


        mean_loss = np.mean(losses_hist)
        self.logger.record("train/pred_loss", mean_loss)

        self.predictor_steps = 5

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.train()

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        continue_training = True

        self.train_predictor()

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                _, values, log_prob, entropy = self.policy.get_action_and_value(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                # Advantages shape: (batch_size,)
                advantages = rollout_data.advantages

                # Compute value-thresholding (vt) term as part of Eq. 3 from the paper
                vt_term = th.min(
                    th.zeros(rollout_data.delta_obs.shape[0]).to(self.device),
                    -rollout_data.delta_obs + th.tensor(self.omega / 2, dtype=th.float32),
                )

                # Compute the error constraint
                error_term = th.min(
                    th.zeros(rollout_data.delta_preds.shape[0]).to(self.device),
                    -rollout_data.delta_preds + th.tensor(self.omega / 2, dtype=th.float32),
                )

                # Compute the variance constraint
                #var_term = th.max(
                #    th.zeros(rollout_data.delta_vars.shape[0]).to(self.device),
                #    -rollout_data.delta_vars,
                #)

                # Bring the 3 terms to scale for numerical stability
                advantages = (advantages - th.min(advantages)) / (
                    th.max(advantages) - th.min(advantages) + 1e-8
                )
                vt_term = (vt_term - th.min(vt_term)) / (
                    th.max(vt_term) - th.min(vt_term) + 1e-8
                )
                error_term = (error_term - th.min(error_term)) / (
                    th.max(error_term) - th.min(error_term) + 1e-8
                )
                #var_term = (var_term - th.min(var_term)) / (
                #    th.max(var_term) - th.min(var_term) + 1e-8
                #)

                # Add terms to advantages
                advantages = (
                    self.beta_0 * advantages
                    + self.beta_1 * vt_term
                    + self.beta_2 * error_term
                #    + self.beta_3 * var_term
                )

                # Normalize advantage
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - self.clip_range, 1 + self.clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > self.clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)

                values_pred = rollout_data.old_values + th.clamp(
                    values - rollout_data.old_values, -self.clip_range, self.clip_range
                )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            if not continue_training:
                break

        # self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
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
        self.logger.record("train/delta_obs", self.rollout_buffer.delta_obs.mean().item())
        self.logger.record("train/delta_pred", self.rollout_buffer.delta_preds.mean().item())
        self.logger.record("train/accuracy", accuracy)

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        
        error_rate = self.env.get_attr("error_accepted")[0]
        self.logger.record("train/error_g0", error_rate[0])
        self.logger.record("train/error_g1", error_rate[1])

        error_bound = self.env.get_attr("error_bound")[0]
        self.logger.record("train/error_bound_g0", error_bound[0])
        self.logger.record("train/error_bound_g1", error_bound[1])
        
        chi_divergence = self.env.get_attr("chi_divergence")[0]
        self.logger.record("train/divergence_g0", chi_divergence[0])
        self.logger.record("train/divergence_g1", chi_divergence[1])

