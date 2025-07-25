import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch
import torch as th
from gym import spaces
from torch.nn import functional as F
import time

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn


from lending_experiment.agents.ppo.sb3.on_policy_algorithm import OnPolicyAlgorithm

from geomloss import SamplesLoss

def estimate_domain_shift(obs_accepted, obs_rejected):
    """Estimate the domain shift between the accepted and rejected populations."""
    wloss = SamplesLoss(loss="sinkhorn", p=1, blur=0.01)

    # calculate the domain shift between the accepted and rejected populations
    dist_distance = wloss(
        obs_accepted,
        obs_rejected,
    )
    return dist_distance


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)
    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            ad_reg: str = "pocar",
            beta_0: float = 1.0,
            beta_1: float = 0.25,
            beta_2: float = 0.,
            beta_3: float = 0.,
            beta_reg: float = 0.35,
            omega: float = 0.1,
            n_steps: int = 2000,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            **kwargs: Any,
    ):

        super(PPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                    batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                    buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.ad_reg = ad_reg
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.beta_reg = beta_reg
        self.omega = omega
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(PPO, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        pred_criterion = th.nn.BCELoss(reduction="none")
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        pred_losses = []

        continue_training = True

        # add rollout data to memory
        actions = self.rollout_buffer.actions
        obs = self.rollout_buffer.observations[actions[:, 0, 0] == 1]
        labels = self.rollout_buffer.labels[actions[:, 0, 0] == 1]
        groups = self.rollout_buffer.groups[actions[:, 0, 0] == 1]


        with th.no_grad():
            obs = th.Tensor(obs).to(self.device)
            probs = self.policy.prob_loan(obs).cpu().numpy()
            obs = obs.cpu().numpy()
        
        self.memory.add(
            obs=obs,
            label=labels,
            group=groups,
            prob=probs
        )


        n_steps = 50
        # learn the prediction model
        if self.ad_reg == "sellf":
            for i, rollout_data in enumerate(self.memory.get(self.batch_size)):
                preds = self.policy.prob_label(rollout_data.observations)
                with th.no_grad():
                    prob_loan = self.policy.prob_loan(rollout_data.observations)
                    # print 0.05 quantile of prob_loan
                    prob_loan = th.clamp(prob_loan, min=0.05, max=0.95)

                # bce term
                pred_loss = pred_criterion(preds, rollout_data.labels)
                pred_loss = pred_loss * (1 / prob_loan)
                pred_loss = pred_loss.sum() / (1 / prob_loan).sum()
                pred_losses.append(pred_loss.item())

                loss = pred_loss
                self.policy.pred_optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                #self.policy.optimizer.step()
                self.policy.pred_optimizer.step()

                if i > n_steps:
                    break

            self.policy.scheduler.step()
        
        n_steps = 5

        # estimate error on the rejected population using domain shift
        with th.no_grad():
            predictions = self.policy.prob_label(torch.Tensor(self.rollout_buffer.observations[:, 0, :]).to(self.device))

        labels = torch.Tensor(self.rollout_buffer.labels[:, 0]).to(self.device)
        error_accepted = []
        error_rejected = []
        error_rejected_estimate = []
        accept_rate = []
        accepted_all_idx = (self.rollout_buffer.actions[:, 0, 0] == 1)
        error_accepted_all = (predictions[accepted_all_idx] - labels[accepted_all_idx]).abs().mean()
        for g_i in range(2):
            accepted_idx = (self.rollout_buffer.groups[:, 0, g_i] == 1) & (self.rollout_buffer.actions[:, 0, 0] == 1)
            rejected_idx = (self.rollout_buffer.groups[:, 0, g_i] == 1) & (self.rollout_buffer.actions[:, 0, 0] == 0)
            error_accepted.append((predictions[accepted_idx] - labels[accepted_idx]).abs().mean())
            error_rejected.append((predictions[rejected_idx] - labels[rejected_idx]).abs().mean()) # not used

            obs_accepted = torch.Tensor(self.rollout_buffer.observations[accepted_idx, 0, :]).to(self.device)
            obs_rejected = torch.Tensor(self.rollout_buffer.observations[rejected_idx, 0, :]).to(self.device)

            accept_rate.append(obs_accepted.shape[0] / (obs_accepted.shape[0] + obs_rejected.shape[0]))

            obs_accepted_all = torch.Tensor(self.rollout_buffer.observations[accepted_all_idx, 0, :]).to(self.device)
            dist_distance = estimate_domain_shift(obs_accepted_all, obs_rejected)

            error_rejected_estimate.append(
                error_accepted_all + dist_distance
            )

            print(f"Group {g_i}: e(S) = {error_accepted_all:.4f}, e(T): {error_rejected[-1]:.4f}, hat e(T): {error_rejected_estimate[-1]:.4f}")

            
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Advantages shape: (batch_size,)
                advantages = rollout_data.advantages

                pred_loss = torch.Tensor([0.]).to(self.device)

                # Advantage regularization for fairness here
                if self.ad_reg == "pocar":
                    # Compute value-thresholding (vt) term as part of Eq. 3 from the paper
                    vt_term = torch.min(
                        torch.zeros(rollout_data.deltas.shape[0]).to(self.device),
                        -rollout_data.deltas + torch.tensor(self.omega, dtype=torch.float32)
                    )

                    # Compute decrease-in-violation (div) term as part of Eq. 3 from the paper
                    div_cond = torch.where(rollout_data.deltas > torch.tensor(self.omega, dtype=torch.float32).to(self.device),
                                           torch.tensor(1, dtype=torch.float32).to(self.device),
                                           torch.tensor(0, dtype=torch.float32).to(self.device))
                    div_term = torch.min(torch.zeros(rollout_data.delta_deltas.shape[0]).to(self.device),
                                         -div_cond * rollout_data.delta_deltas)

                    # Bring the 3 terms to scale for numerical stability
                    advantages = (advantages - torch.min(advantages)) / (torch.max(advantages) - torch.min(advantages) + 1e-8)
                    vt_term = (vt_term - torch.min(vt_term)) / (torch.max(vt_term) - torch.min(vt_term) + 1e-8)
                    div_term = (div_term - torch.min(div_term)) / (torch.max(div_term) - torch.min(div_term) + 1e-8)

                    # Add terms to advantages
                    advantages = (self.beta_0 * advantages + self.beta_1 * vt_term + self.beta_2 * div_term)
                elif self.ad_reg == "sellf":
                    vt_term = torch.min(
                        torch.zeros(rollout_data.deltas.shape[0]).to(self.device),
                        -rollout_data.deltas + torch.tensor(self.omega / 2, dtype=torch.float32)
                    )

                    div_cond = torch.where(rollout_data.deltas > torch.tensor(self.omega / 2, dtype=torch.float32).to(self.device),
                                           torch.tensor(1, dtype=torch.float32).to(self.device),
                                           torch.tensor(0, dtype=torch.float32).to(self.device))
                    div_term = torch.min(torch.zeros(rollout_data.delta_deltas.shape[0]).to(self.device),
                                         -div_cond * rollout_data.delta_deltas)

                    # error term is equal to the negative value of delta_b_term
                    error_term = torch.min(
                        torch.zeros(rollout_data.delta_b_terms.shape[0]).to(self.device),
                        -rollout_data.delta_b_terms + torch.tensor(self.omega / 2, dtype=torch.float32)
                    )

                    # only apply error_term if vt_term is zero
                    vt_term_zero = torch.where(vt_term == 0, 
                                               torch.tensor(1, dtype=torch.float32).to(self.device), 
                                               torch.tensor(0, dtype=torch.float32).to(self.device)
                                            )
                    #error_term = error_term * vt_term_zero


                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    vt_term = (vt_term - torch.min(vt_term)) / (torch.max(vt_term) - torch.min(vt_term) + 1e-8)
                    div_term = (div_term - torch.min(div_term)) / (torch.max(div_term) - torch.min(div_term) + 1e-8)
                    error_term = (error_term - torch.min(error_term)) / (torch.max(error_term) - torch.min(error_term) + 1e-8)

                    advantages = (self.beta_0 * advantages + self.beta_1 * vt_term + self.beta_2 * div_term + self.beta_3 * error_term)

                # Normalize advantage
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)


                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
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

                # Accept rate disparity loss
                g0_idx = (rollout_data.groups[:, 0] == 1).nonzero()
                g1_idx = (rollout_data.groups[:, 1] == 1).nonzero()
                prob_accept = self.policy.prob_loan(rollout_data.observations)

                accept_g0_batch = (1 - prob_accept[g0_idx].to(th.float32).mean()) * error_rejected[0]
                accept_g1_batch = (1 - prob_accept[g1_idx].to(th.float32).mean()) * error_rejected[1]
                delta_accept = (accept_g0_batch - accept_g1_batch) ** 2


                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + 10 * delta_accept

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/pred_loss", np.mean(pred_losses))
        self.logger.record("train/accept_rate", np.mean(self.rollout_buffer.actions.flatten()))
        self.logger.record("train/pos_rate", np.mean(self.rollout_buffer.labels.flatten()))
        self.logger.record("train/accept_g0", accept_rate[0])
        self.logger.record("train/accept_g1", accept_rate[1])
        self.logger.record("train/error_accepted", error_accepted_all.item())
        self.logger.record("train/error_a0_g0", error_rejected[0].item())
        self.logger.record("train/error_a0_g1", error_rejected[1].item())
        self.logger.record("train/error_a1_g0", error_accepted[0].item())
        self.logger.record("train/error_a1_g1", error_accepted[1].item())
        self.logger.record("train/error_hat_a0_g0", error_rejected_estimate[0].item())
        self.logger.record("train/error_hat_a0_g1", error_rejected_estimate[1].item())
        self.logger.record("train/b_g0", (1 - accept_rate[0]) * error_rejected[0].item())
        self.logger.record("train/b_g1", (1 - accept_rate[1]) * error_rejected[1].item())
        self.logger.record("train/b_hat_g0", (1 - accept_rate[0]) * error_rejected_estimate[0].item())
        self.logger.record("train/b_hat_g1", (1 - accept_rate[1]) * error_rejected_estimate[1].item())
        self.logger.record("train/delta_b_term", self.rollout_buffer.delta_b_terms.mean().item())
        self.logger.record("train/delta", self.rollout_buffer.deltas.mean().item())
        self.logger.record("train/delta_delta", self.rollout_buffer.delta_deltas.mean().item())
        self.logger.record("train/delta_real", self.rollout_buffer.delta_reals.mean().item())

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "PPO",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "PPO":

        return super(PPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )