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


from fairrl.agents.on_policy_algorithm import OnPolicyAlgorithm


class SELLF(OnPolicyAlgorithm):
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 1e-5,
        beta_0: float = 1,
        beta_1: float = 0.5,
        beta_2: float = 0.5,
        omega: float = 0.1,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        normalize_advantage: bool = True,
        ent_coef: float = 0.,
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

        self.utility_method = env.utility_method
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.omega = omega
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.predictor_steps = 25
        self.first_iter = True
        self.calculate_error = False

        self._setup_model()

    def train_predictor(self) -> None:
        """Train the predictor model using the accepted data from rollout buffer."""

        pred_criterion = th.nn.BCELoss(reduction="none")

        # add rollout data to memory
        actions = self.rollout_buffer.actions
        obs = self.rollout_buffer.observations[actions[:, 0, 0] == 1]
        labels = self.rollout_buffer.labels[actions[:, 0, 0] == 1]
        groups = self.rollout_buffer.groups[actions[:, 0, 0] == 1]

        self.memory.add(obs=obs, label=labels, group = groups)

        self.policy.save_history()

        losses_hist=[]
        max_weight_g0 = 0
        max_weight_g1 = 0

        steps = 0
        for epoch in range(100):
            for i, rollout_data in enumerate(self.memory.get(self.batch_size)):
                group_0_idx = (rollout_data.groups[:, 0] == 1).nonzero()
                group_1_idx = (rollout_data.groups[:, 1] == 1).nonzero()

                if len(group_0_idx) == 0 or len(group_1_idx) == 0:
                    continue

                preds = self.policy.get_label_prob(rollout_data.observations)
                with th.no_grad():
                    prob_rej = 1 - self.policy.get_action_prob(rollout_data.observations)
                    prob_accept_all = self.policy.get_action_all_prob(rollout_data.observations)
                    weights = prob_rej / (prob_accept_all)

                
                max_weight_g0 = max(max_weight_g0, weights[group_0_idx].max().item())
                max_weight_g1 = max(max_weight_g1, weights[group_1_idx].max().item())

                pred_loss = pred_criterion(preds, rollout_data.labels)
                pred_loss = pred_loss * weights
                pred_loss = pred_loss[group_0_idx].sum() / weights[group_0_idx].sum() + pred_loss[group_1_idx].sum() / weights[group_1_idx].sum()

                losses_hist.append(pred_loss.item())

                self.policy.pred_optimizer.zero_grad()
                pred_loss.backward()
                self.policy.pred_optimizer.step()

                steps += 1

                if steps >= self.predictor_steps:
                    break
            
            self.policy.pred_scheduler.step()
            self.logger.record("pred_lr", self.policy.pred_scheduler.get_last_lr()[0])
            if steps >= self.predictor_steps:
                break
        
        if self.calculate_error:
            # calculate weighted error on accepted
            accepted_error_g0 = []
            accepted_error_g1 = []
            weights_g0 = []
            weights_g1 = []
            for i, rollout_data in enumerate(self.memory.get(self.batch_size)):
                group_0_idx = (rollout_data.groups[:, 0] == 1).nonzero()
                group_1_idx = (rollout_data.groups[:, 1] == 1).nonzero()
            
                preds = self.policy.get_label(rollout_data.observations)
                with th.no_grad():
                    prob_rej = 1 - self.policy.get_action_prob(rollout_data.observations)
                    prob_accept_all = self.policy.get_action_all_prob(rollout_data.observations)
                    weights = prob_rej / (prob_accept_all)
            
                errors = (preds - rollout_data.labels.view(-1))
                accepted_error_g0.append(errors[group_0_idx].cpu().numpy())
                accepted_error_g1.append(errors[group_1_idx].cpu().numpy())
                weights_g0.append(weights[group_0_idx].cpu().numpy())
                weights_g1.append(weights[group_1_idx].cpu().numpy())

                if i >= 150:
                    break


            accepted_error_g0 = np.concatenate(accepted_error_g0)
            accepted_error_g1 = np.concatenate(accepted_error_g1)
            weights_g0 = np.concatenate(weights_g0)
            weights_g1 = np.concatenate(weights_g1) 

            pred_error_g0 = (accepted_error_g0 * weights_g0).sum() / (weights_g0.sum() + 1e-8)
            pred_error_g1 = (accepted_error_g1 * weights_g1).sum() / (weights_g1.sum() + 1e-8)


            self.logger.record("error_accepted_g0", pred_error_g0)
            self.logger.record("error_accepted_g1", pred_error_g1)


        mean_loss = np.mean(losses_hist)
        self.logger.record("pred_loss", mean_loss)
        self.logger.record("max_weight_g0", max_weight_g0)
        self.logger.record("max_weight_g1", max_weight_g1)


    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.train()

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        renyi_divs_g0 = []
        renyi_divs_g1 = []
        continue_training = True

        self.train_predictor()
        if self.first_iter == True:
            self.first_iter = False
            # only train the predictor
            return
        
        # calculate r_i and a[i]_i for each group
        g0_idx = (self.rollout_buffer.groups[:, :, 0] == 1).nonzero()
        g1_idx = (self.rollout_buffer.groups[:, :, 1] == 1).nonzero()

        r0 = 1 - self.rollout_buffer.actions[g0_idx].mean()
        r1 = 1 - self.rollout_buffer.actions[g1_idx].mean()
        aK_0 = self.rollout_buffer.prob_action_all[g0_idx].mean()
        aK_1 = self.rollout_buffer.prob_action_all[g1_idx].mean()

        aK_min_g0 = self.rollout_buffer.prob_action_all[g0_idx].min().item()
        aK_min_g1 = self.rollout_buffer.prob_action_all[g1_idx].min().item()

        if self.utility_method == "tpr":
            tphi_0 = (self.rollout_buffer.imputations[g0_idx]).mean()
            tphi_1 = (self.rollout_buffer.imputations[g1_idx]).mean()
            c0 = aK_0 / (r0 * tphi_0 + 1e-8)
            c1 = aK_1 / (r1 * tphi_1 + 1e-8)
        else:
            c0 = aK_0 / (r0 + 1e-8)
            c1 = aK_1 / (r1 + 1e-8)
        
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

                advantages = (advantages - th.min(advantages)) / (
                    th.max(advantages) - th.min(advantages) + 1e-8
                )
                vt_term = (vt_term - th.min(vt_term)) / (
                    th.max(vt_term) - th.min(vt_term) + 1e-8
                )

                # Add terms to advantages
                advantages = (
                    self.beta_0 * advantages
                    + self.beta_1 * vt_term
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

                # also minimize the Renyi divergence between the current policy and all previous policies
                prob_action = self.policy.get_action_prob(rollout_data.observations)
                group_0_idx = (rollout_data.groups[:, 0] == 1).nonzero()
                group_1_idx = (rollout_data.groups[:, 1] == 1).nonzero()
                prob_all_action = rollout_data.prob_action_all
                renyi_div_g0 = c0 * (1 - prob_action[group_0_idx]) ** 2 / (prob_all_action[group_0_idx] + 1e-8)
                renyi_div_g1 = c1 * (1 - prob_action[group_1_idx]) ** 2 / (prob_all_action[group_1_idx] + 1e-8)
                renyi_div = (renyi_div_g0.mean() + renyi_div_g1.mean()) / (c1 + c0)

                renyi_divs_g0.append(renyi_div_g0.mean().item())
                renyi_divs_g1.append(renyi_div_g1.mean().item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.beta_2 * renyi_div
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
        self.logger.record("entropy_loss", np.mean(entropy_losses))
        self.logger.record("policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("value_loss", np.mean(value_losses))
        self.logger.record("approx_kl", np.mean(approx_kl_divs))
        self.logger.record("clip_fraction", np.mean(clip_fractions))
        self.logger.record("loss", loss.item())
        self.logger.record("renyi_div_g0", np.mean(renyi_divs_g0))
        self.logger.record("renyi_div_g1", np.mean(renyi_divs_g1))
        self.logger.record("explained_variance", explained_var)
        self.logger.record(
            "accept_rate", np.mean(self.rollout_buffer.actions.flatten())
        )
        self.logger.record(
            "pos_rate", np.mean(self.rollout_buffer.labels.flatten())
        )
        self.logger.record("reward", self.rollout_buffer.rewards.mean().item())

        # Logs some group-dependent variables
        accuracy = (
            (self.rollout_buffer.labels[:, 0] == self.rollout_buffer.preds[:, 0])
            .mean()
            .item()
        )

        self.logger.record("accept_g0", 1 - r0)
        self.logger.record("accept_g1", 1 - r1)
        self.logger.record("delta", self.rollout_buffer.deltas.mean().item())
        self.logger.record("delta_obs", self.rollout_buffer.delta_obs.mean().item())
        self.logger.record("accuracy", accuracy)

        error_rejected = self.env.error_rejected
        self.logger.record("error_rejected_g0", error_rejected[0])
        self.logger.record("error_rejected_g1", error_rejected[1])

        delta_pred_real = self.env.delta_pred_real
        self.logger.record("delta_pred_real", delta_pred_real)
        self.logger.record("aK_min_g0", aK_min_g0)
        self.logger.record("aK_min_g1", aK_min_g1)
        self.logger.record("aK_g0", aK_0)
        self.logger.record("aK_g1", aK_1)
        
        

