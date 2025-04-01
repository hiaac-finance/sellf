import warnings
from typing import Any, Dict, Optional, Tuple, Type, Union, List, Iterator

import numpy as np
import torch
import torch as th
from gym import spaces
from torch.nn import functional as F

from geomloss import SamplesLoss
wloss = SamplesLoss("sinkhorn", p=1, blur=0.01)

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    explained_variance,
    get_schedule_fn,
)

from agents.ppo.sb3.on_policy_algorithm import OnPolicyAlgorithm
from agents.ppo.q_custom_utils import RollingAvg

from config import Config


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
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
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
            # # --------------------------------------------------------------------------------------------
            config_params: Type[Config] = Config(),
    ):
        super(PPO, self).__init__(
            policy,
            env,
            learning_rate=config_params.LEARNING_RATE,
            n_steps=config_params.EP_TIMESTEPS,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=config_params.POLICY_KWARGS,
            verbose=1,
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

            config_params=config_params,
        )
        batch_size = config_params.BATCH_SIZE
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

        self.batch_size = batch_size
        self.n_epochs = config_params.N_EPOCHS
        self.clip_range = config_params.PPO_CLIP_RANGE
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        self.BETA_C_PI = config_params.BETA_C_PI
        self.static_kl_coeff = config_params.STATIC_KL_COEFF
        self.static_kl_targ = config_params.STATIC_KL_TARG
        self.REGULARIZE_ADVANTAGE = config_params.REGULARIZE_ADVANTAGE
        self.BETA_0 = config_params.BETA_0
        self.BETA_1 = config_params.BETA_1
        self.BETA_2 = config_params.BETA_2
        self.BETA_3 = config_params.BETA_3
        self.BETA_4 = config_params.BETA_4
        self.OMEGA = config_params.OMEGA
        self.KL_PEN = config_params.KL_PEN

        self.USE_F_DELTA = config_params.USE_F_DELTA
        self.B_EPSILON = config_params.B_EPSILON

        self.BETA_LAMBDA = config_params.BETA_LAMBDA

        # --- NEW ---
        self.BETA_PF = config_params.BETA_PF
        self.IMPUTATION = config_params.IMPUTATION
        self.IPW = config_params.IPW

        self.config = config_params

        self.rolling_avg_metrics = RollingAvg(20)

        self.cred_score_range = 14

        self.expected_num_tsteps = config_params.EP_TIMESTEPS / config_params.NUM_INDIVIDUALS

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

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        c_pi_theta_losses = []
        lambda_losses = []
        imputation_losses = []

        dpe_list = []
        ipe_list = []
 
        g_0_idx = self.rollout_buffer.g_loc_start
        g_1_idx = g_0_idx + 1

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            approx_static_kl_divs = []

            # Do a complete pass on the rollout buffer
            j = 0
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, q_log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

                # need the indices of the group 0 and group 1 observations
                g0_idxs = rollout_data.observations[:,g_0_idx].nonzero().flatten()
                g1_idxs = rollout_data.observations[:,g_1_idx].nonzero().flatten()
                
                values = values.flatten()

                # Advantages shape: (batch_size,)
                advantages = rollout_data.advantages

                q_advs = rollout_data.q_advs

                # ---------------------------A-PPO/F-PPO stuff--------------------------------------------
                # ----------------------------------------------------------------------------------------
                if self.REGULARIZE_ADVANTAGE:
                    # Compute value-thresholding (vt) term as part of Eq. 3 from the paper
                    vt_term = torch.min(
                        torch.zeros(rollout_data.deltas.shape[0]).to(self.device),
                        -rollout_data.deltas + torch.tensor(self.OMEGA, dtype=torch.float32)
                    )

                    # Compute decrease-in-violation (div) term as part of Eq. 3 from the paper
                    div_cond = torch.where(rollout_data.deltas > torch.tensor(self.OMEGA, dtype=torch.float32).to(self.device),
                                           torch.tensor(1, dtype=torch.float32).to(self.device),
                                           torch.tensor(0, dtype=torch.float32).to(self.device))
                    div_term = torch.min(torch.zeros(rollout_data.delta_deltas.shape[0]).to(self.device),
                                         -div_cond * rollout_data.delta_deltas)

                    # ------------------------------- f-ppo ---------------------------------------------
                    long_term1 = torch.min(torch.zeros(rollout_data.long_delta.shape[0]).to(self.device),
                                          -div_cond * rollout_data.long_delta)
                    long_term2 = torch.max(torch.zeros(rollout_data.long_delta.shape[0]).to(self.device),
                                           -(1. - div_cond) * rollout_data.long_delta)
                    # ------------------------------------------------------------------------------------------

                    # Bring the 3 terms to scale for numerical stability
                    advantages = (advantages - torch.min(advantages)) / (torch.max(advantages) - torch.min(advantages) + 1e-8)
                    vt_term = (vt_term - torch.min(vt_term)) / (torch.max(vt_term) - torch.min(vt_term) + 1e-8)
                    div_term = (div_term - torch.min(div_term)) / (torch.max(div_term) - torch.min(div_term) + 1e-8)

                    # Add terms to advantages
                    # ------------------------------- f-ppo -----------------------------------------
                    long_term1 = (long_term1 - torch.min(long_term1)) / (torch.max(long_term1) - torch.min(long_term1) + 1e-8)
                    long_term2 = (long_term2 - torch.min(long_term2)) / (torch.max(long_term2) - torch.min(long_term2) + 1e-8)

                    advantages = (self.BETA_0 * advantages + self.BETA_1 * vt_term + self.BETA_2 * div_term + self.BETA_3 * long_term1 + self.BETA_4 * long_term2)  
                # ----------------------------------------------------------------------------------------
                # ----------------------------------------------------------------------------------------

                if self.ADJUST_QUALIF:
                    # Compute the adjusted advantage based on the disparity of qualification rates Y
                    disp = torch.min(
                        torch.zeros(rollout_data.deltas.shape[0]).to(self.device),
                        -rollout_data.deltas + torch.tensor(self.OMEGA, dtype=torch.float32)
                    )          
                    disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)
                    advantages = (self.BETA_0 * advantages + self.BETA_1 * disp)



                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    q_advs = ((q_advs - q_advs.mean()) / (q_advs.std() + 1e-8))

                # q_log_p contains the log likelihood of the actions sampled in the trajectory
                q_log_p = log_prob.clone()
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                if self.BETA_C_PI < 0.01:
                    with torch.no_grad():
                        q_ratio = th.exp(q_log_p - rollout_data.old_log_prob)
                else:
                    q_ratio = th.exp(q_log_p - rollout_data.old_log_prob)

                # Compute the KL divergence between the old and new policies, q_log_prob has log likelihoods for both actions
                if self.KL_PEN:
                    static_kl_d = (th.exp(rollout_data.q_log_prob)*(rollout_data.q_log_prob - q_log_prob)).sum(axis=1).mean()
                else:
                    # for logging purposes only
                    with th.no_grad():
                        static_kl_d = (th.exp(rollout_data.q_log_prob)*(rollout_data.q_log_prob - q_log_prob.detach())).sum(axis=1).mean()

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                ppo_loss = th.min(policy_loss_1, policy_loss_2).mean()

                # -------------- New ---------------------------------------------
                # Compute the policy loss for PPO-C and PPO-CB
                if self.BETA_C_PI > 0.0 or self.USE_F_DELTA:
                    # Compute the policy constraint term
                    c_pi_theta = (q_advs[g0_idxs]*q_ratio[g0_idxs]).mean() \
                          - (q_advs[g1_idxs]*q_ratio[g1_idxs]).mean() 
                    
                    # Compute the lambda term
                    if self.USE_F_DELTA: # lambda eqn 
                        pa1_g0 = self.policy.prob_loan(rollout_data.observations[g0_idxs]).view(-1,1)
                        pa1_g1 = self.policy.prob_loan(rollout_data.observations[g1_idxs]).view(-1,1)
                        
                        diff_n1 = th.abs(pa1_g0 - pa1_g1.T)
                        diff_d1 = self.B_EPSILON + th.abs(rollout_data.benefit_deltas[g0_idxs].view(-1,1) - rollout_data.benefit_deltas[g1_idxs].view(-1,1).T)

                        lambda_loss = self.B_EPSILON * th.sum(diff_n1/diff_d1)
                        lambda_loss = lambda_loss / diff_n1.numel()
                        lambda_losses.append(lambda_loss.item())

                        with torch.no_grad():
                            dpe = th.mean(rollout_data.benefit_deltas[g0_idxs]*pa1_g0.view(-1)) - th.mean(rollout_data.benefit_deltas[g1_idxs]*pa1_g1.view(-1))
                            ipe = th.mean(rollout_data.v_ps[g0_idxs] - rollout_data.expected_v_base[g0_idxs]) - th.mean(rollout_data.v_ps[g1_idxs] - rollout_data.expected_v_base[g1_idxs])

                    else:
                        # for logging
                        with torch.no_grad():
                            pa1_g0 = self.policy.prob_loan(rollout_data.observations[g0_idxs]).view(-1,1)
                            pa1_g1 = self.policy.prob_loan(rollout_data.observations[g1_idxs]).view(-1,1)
                            
                            diff_n1 = th.abs(pa1_g0 - pa1_g1.T)
                            diff_d1 = self.B_EPSILON + th.abs(rollout_data.benefit_deltas[g0_idxs].view(-1,1) - rollout_data.benefit_deltas[g1_idxs].view(-1,1).T)

                            lambda_loss = self.B_EPSILON * th.sum(diff_n1/diff_d1)
                            lambda_loss = lambda_loss / diff_n1.numel()
                            lambda_losses.append(lambda_loss.item())
                            lambda_loss = th.tensor(0.0).to(self.device)

                            dpe = th.mean(rollout_data.benefit_deltas[g0_idxs]*pa1_g0.view(-1)) - th.mean(rollout_data.benefit_deltas[g1_idxs]*pa1_g1.view(-1))
                            ipe = th.mean(rollout_data.v_ps[g0_idxs] - rollout_data.expected_v_base[g0_idxs]) - th.mean(rollout_data.v_ps[g1_idxs] - rollout_data.expected_v_base[g1_idxs])
                    dpe_list.append(dpe.item())
                    ipe_list.append(ipe.item())

                    c_pi_theta = c_pi_theta**2

                    if self.KL_PEN:
                        policy_loss = -(ppo_loss - self.BETA_C_PI*c_pi_theta - self.static_kl_coeff*static_kl_d - self.BETA_LAMBDA * lambda_loss)
                    else:
                        policy_loss = -(ppo_loss - self.BETA_C_PI*c_pi_theta - self.BETA_LAMBDA * lambda_loss)
                elif self.BETA_PF > 0:
                    ys = rollout_data.ys
                    imputation_loss = torch.tensor([0.0])
                    if self.IMPUTATION:
                        ys_probs = self.policy.predict_label(rollout_data.observations)
                        ys_pred = torch.argmax(ys_probs, dim=1).view(-1, 1).to(torch.float32).detach()
                        # replace y_pred with y1 for imputation
                        imputation_loss = F.cross_entropy(ys_probs, ys.to(torch.long).flatten())
                        ys[rollout_data.actions == 0] = ys_pred[rollout_data.actions == 0]
                    elif self.IPW:
                        # Y(a) = Y(a) * I(a) / p(a)
                        ys = ys * rollout_data.actions / self.policy.prob_loan(rollout_data.observations).view(-1,1)
                    imputation_losses.append(imputation_loss.item())
                    
                    # EQUAL QUALIFICATION RATE  (not working)
                    # disp = (ys[g0_idxs].mean() - ys[g1_idxs].mean())**2

                    # EQUAL TRUE POSITIVE RATE (not working)

                    # ys_idxs = ys.nonzero().flatten()
                    # # get indexes that are from at g0_idxs and y1_idxs
                    # ys_g0_idxs = np.intersect1d(ys_idxs, g0_idxs)
                    # ys_g1_idxs = np.intersect1d(ys_idxs, g1_idxs)
                    # if len(ys_g0_idxs) == 0:
                    #     pa1_g0 = 0
                    # else:
                    #     pa1_g0 = self.policy.prob_loan(rollout_data.observations[ys_g0_idxs]).mean()
                    
                    # if len(ys_g1_idxs) == 0:
                    #     pa1_g1 = 0
                    # else:
                    #     pa1_g1 = self.policy.prob_loan(rollout_data.observations[ys_g1_idxs]).mean()
                    # disp = (pa1_g0 - pa1_g1)**2


                    # EQUAL ERROR RATE
                    p = self.policy.prob_loan(rollout_data.observations).view(-1,1)
                    # calculate binary cross entropy
                    xen = F.binary_cross_entropy(p, ys, reduction='none')
                    disp = (xen[g0_idxs].mean() - xen[g1_idxs].mean()) ** 2
                    
                    if self.KL_PEN:
                        policy_loss = -(ppo_loss - self.BETA_PF * disp - self.static_kl_coeff*static_kl_d - self.BETA_PF * imputation_loss)
                    else:
                        policy_loss = -(ppo_loss - self.BETA_PF * disp - self.BETA_PF * imputation_loss)

                    with torch.no_grad():
                        c_pi_theta = th.tensor(0.0).to(self.device)
                        lambda_loss = th.tensor(0.0).to(self.device)
                        lambda_losses.append(lambda_loss.item())

                else:
                # loss for other variants 
                    with torch.no_grad():
                        c_pi_theta = th.tensor(0.0).to(self.device)
                        lambda_loss = th.tensor(0.0).to(self.device)
                        lambda_losses.append(lambda_loss.item())

                    if self.KL_PEN:
                        policy_loss = -(ppo_loss - self.static_kl_coeff*static_kl_d)
                    else:
                        policy_loss = -ppo_loss

                    dpe_list.append(0.0)
                    ipe_list.append(0.0)

                # ----------------------------------------------------------------
                # Logging
                approx_static_kl_divs.append(static_kl_d.item())

                pg_losses.append(ppo_loss.item())
                c_pi_theta_losses.append(c_pi_theta.item())
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

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
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

                if torch.isnan(self.policy.mlp_extractor.shared_net[0].weight).any():
                    import pdb; pdb.set_trace()

                j += 1

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/static_kl", np.mean(approx_static_kl_divs))
        self.logger.record("train/imputation_loss", np.mean(imputation_losses))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        # ---------------------------------------------------------------------------------
        self.logger.record("train/cumulative_reward", self.rollout_buffer.rewards.sum().item())
        self.logger.record("train/policy_constraint_loss", np.mean(c_pi_theta_losses)),
        self.logger.record("train/lambda_loss", np.mean(lambda_losses))
        self.logger.record("train/dpe_loss", np.mean(dpe_list))
        self.logger.record("train/ipe_loss", np.mean(ipe_list))

        g0_idxs = self.rollout_buffer.observations[:,g_0_idx].nonzero()
        g1_idxs = self.rollout_buffer.observations[:,g_1_idx].nonzero()
        g0_gx = self.rollout_buffer.gx[g0_idxs]
        g1_gx = self.rollout_buffer.gx[g1_idxs]
        g0_sum_gx = g0_gx.sum().item()
        g1_sum_gx = g1_gx.sum().item()
        
        self.logger.record("train/cumulative_gx_g0", g0_sum_gx)
        self.logger.record("train/cumulative_gx_g1", g1_sum_gx)
        self.logger.record("train/cumulative_gx_diff", abs(g0_sum_gx - g1_sum_gx))

        self.logger.record("train/soft_de", self.rollout_buffer.decomps['final']['de'])
        self.logger.record("train/soft_ie", self.rollout_buffer.decomps['final']['ie'])
        self.logger.record("train/soft_se", self.rollout_buffer.decomps['final']['se'])
        self.logger.record("train/c_pi_theta", self.rollout_buffer.decomps['final']['c_pi_theta'])

        self.rolling_avg_metrics.update(
            np.mean(c_pi_theta_losses), 
            self.rollout_buffer.decomps['final']['ie'], 
            self.rollout_buffer.decomps['final']['se'],
            self.rollout_buffer.decomps['final']['de'],
            self.rollout_buffer.decomps['final']['c_pi_theta'],
            np.mean(lambda_losses),
            np.mean(dpe_list),
        )

        self.logger.record("train/rolling_policy_constraint_loss", self.rolling_avg_metrics.avg_policy_constraint())
        self.logger.record("train/rolling_soft_ie", self.rolling_avg_metrics.avg_soft_ie())
        self.logger.record("train/rolling_soft_se", self.rolling_avg_metrics.avg_soft_se())
        self.logger.record("train/rolling_soft_de", self.rolling_avg_metrics.avg_soft_de())
        self.logger.record("train/rolling_c_pi_theta", self.rolling_avg_metrics.avg_c_pi_theta())
        self.logger.record("train/rolling_lambda_loss", self.rolling_avg_metrics.avg_lambda_loss())
        self.logger.record("train/rolling_dpe_loss", self.rolling_avg_metrics.avg_dpe_loss())


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
