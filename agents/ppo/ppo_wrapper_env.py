import gym
import numpy as np
import torch
from gym import spaces
from copy import deepcopy
from collections import deque
from geomloss import SamplesLoss

from environments.distributions import Bernoulli

wloss = SamplesLoss("sinkhorn", p=1, blur=0.01)


class PPOEnvWrapper(gym.Wrapper):
  def __init__(self,
               env,
               reward_fn,
               config_params=None,
               is_eval=False,
    ):
    super(PPOEnvWrapper, self).__init__(env)

    self.DRIFT_PROBS = config_params.DRIFT_PROBS
    self.USE_CREDIT_DRIFT = config_params.USE_CREDIT_DRIFT

    self.pi_0 = Bernoulli(config_params.PI0_PROB_A1)

    self.DIST_EST = config_params.DIST_EST

    # don't want to compute wloss if we are not using it, slows down training
    if config_params.REGULARIZE_ADVANTAGE == True and config_params.BETA_3 > 0.0:
      self.is_bigd_fppo = True
      print('Using BigD-FPPO')
    else:
      self.is_bigd_fppo = False
    
    if self.DIST_EST:
        self.observation_space = spaces.Box(
        low=np.inf,
        high=np.inf,
        shape=(3 * env.observation_space['applicant_features'].shape[0] + env.state.params.num_groups,),
      )
    else:
      self.observation_space = spaces.Box(
        low=np.inf,
        high=np.inf,
        shape=(env.observation_space['applicant_features'].shape[0] + env.state.params.num_groups,),
        # -------------------------------------------------------------
      )

    self.action_space = spaces.Discrete(n=2)

    self.env = env
    self.reward_fn = reward_fn(omega = config_params.OMEGA,)

    self.timestep = 0

    self.tp = np.zeros(self.env.state.params.num_groups,)
    self.fp = np.zeros(self.env.state.params.num_groups,)
    self.tn = np.zeros(self.env.state.params.num_groups,)
    self.fn = np.zeros(self.env.state.params.num_groups,)
    self.tpr = np.zeros(self.env.state.params.num_groups,)
    self.delta = np.zeros(1, )
    self.old_bank_cash = 0

    # --- NEW ---
    # OBSERVED MEASURE OF DISPARITY
    self.tp_obs = np.zeros(self.env.state.params.num_groups,)
    self.fp_obs = np.zeros(self.env.state.params.num_groups,)
    self.tn_obs = np.zeros(self.env.state.params.num_groups,)
    self.fn_obs = np.zeros(self.env.state.params.num_groups,)
    self.tpr_obs = np.zeros(self.env.state.params.num_groups,)
    self.acc = np.zeros(self.env.state.params.num_groups,)
    self.acc_obs = np.zeros(self.env.state.params.num_groups,)
    
    if is_eval:
      self.ep_timesteps = config_params.EVAL_EP_TIMESTEPS
      self.ZETA_0 = config_params.EVAL_ZETA_0
      self.ZETA_1 = config_params.EVAL_ZETA_1
    else:
      self.ep_timesteps = config_params.EP_TIMESTEPS
      self.ZETA_0 = config_params.ZETA_0
      self.ZETA_1 = config_params.ZETA_1
      

    self.QUAL_CHANGE = config_params.QUAL_CHANGE
    self.WINDOW = config_params.WINDOW
    self.ONLY_OBSERVATION = config_params.ONLY_OBSERVATION

  def process_observation(self, obs):
    credit_score = obs['applicant_features']
    group = obs['group']
    hist0 = self.history[0]
    hist1 = self.history[1]
    norm = sum(hist0 + hist1) + 1.

    if self.DIST_EST:
      return np.concatenate(
        (credit_score,
        group,
        hist0 / norm,
        hist1 / norm,
        ),
        axis=0
      )
    else:
      return np.concatenate(
        (credit_score,
        group,
        ),
        axis=0
      )

  def compute_tpr(self, tp, fn):
    # tp: true positive, 2-dimensional for 2 groups
    # fn: false negative, 2-dimensional for 2 groups
    return np.divide(
      tp,
      tp + fn,
      out=np.zeros_like(tp),
      where=(tp + fn) != 0
    )
  
  def compute_acc(self, tp, fn, tn, fp):
    return np.divide(
      tp + tn,
      tp + fn + tn + fp,
      out = np.zeros_like(tp),
      where=(tp + fn + tn + fp) != 0 
    )

  def reset(self):
    self.timestep = 0
    self.tp = np.zeros(self.env.state.params.num_groups,)
    self.fp = np.zeros(self.env.state.params.num_groups,)
    self.tn = np.zeros(self.env.state.params.num_groups,)
    self.fn = np.zeros(self.env.state.params.num_groups,)
    self.tpr = np.zeros(self.env.state.params.num_groups,)
    self.delta = np.zeros(1, )
    self.old_bank_cash = 0
    self.delta_delta = 0

    # --- NEW ---
    # OBSERVED MEASURE OF DISPARITY
    self.tp_obs = np.zeros(self.env.state.params.num_groups,)
    self.fp_obs = np.zeros(self.env.state.params.num_groups,)
    self.tn_obs = np.zeros(self.env.state.params.num_groups,)
    self.fn_obs = np.zeros(self.env.state.params.num_groups,)
    self.tpr_obs = np.zeros(self.env.state.params.num_groups,)
    self.group_hist = np.zeros((self.ep_timesteps, self.env.state.params.num_groups))
    self.quali_hist = np.zeros((self.ep_timesteps, self.env.state.params.num_groups))
    self.quali_obs = np.zeros(self.env.state.params.num_groups,)
    self.acc = np.zeros(self.env.state.params.num_groups,)
    self.acc_obs = np.zeros(self.env.state.params.num_groups,)

    # ----------------------------------- add history and population ----------------------------
    self.history = np.zeros((self.env.state.params.num_groups, self.env.observation_space['applicant_features'].shape[0]))
    self.population = deque(maxlen=self.WINDOW)
    self.dist = 0
    self.dist_dist = 0
    # --------------------------------------------------------------------------------------------

    return self.process_observation(self.env.reset())
  
  def next_x_given_action(self, action):
    """
    Returns the next x given the current state and action for the current applicant.
    Needed because environment does not track individuals, this is done in the
    rollout buffer.
    """
    curr_x = np.argmax(self.env.state.applicant_features)
    credit_drift = self.env.state.credit_drift

    if self.USE_CREDIT_DRIFT:
      curr_x = curr_x + credit_drift
    if action == 1:
      if self.env.state.will_default:
        next_x = max(0, curr_x - 1)
      else:
        next_x = min(len(self.env.state.applicant_features)-1, curr_x + 1)
    elif action == 0:
      next_x = curr_x
      next_x = min(len(self.env.state.applicant_features)-1, next_x)
      next_x = max(0, next_x)

    return next_x
  
  def get_g_pi0(self, curr_x, g):
    """
    Returns the qualification gain according to baseline policy pi_0 that always denies.
    Done this way in case the baseline policy is changed.
    """
    action = np.int16(self.pi_0.sample(rng=self.env.state.rng))
    next_x = self.next_x_given_action(action)

    return self.QUAL_CHANGE(curr_x, next_x), action, next_x

  def step(self, action):
    old_delta = self.delta

    curr_x = np.argmax(self.env.state.applicant_features)
    # Update instance variables before we step the environment
    group_id = np.argmax(self.env.state.group)
    if action == 1:
      # Check if individual would default
      if self.env.state.will_default:
        self.fp[group_id] += 1
      else:
        self.tp[group_id] += 1
      
    elif action == 0:
      if self.env.state.will_default:
        self.tn[group_id] += 1
      else:
        self.fn[group_id] += 1

    # --- NEW ---
    # UPDATE OBSERVED (and non-observed) MEASURES
    self.group_hist[self.timestep, group_id] = 1
    if action == 1:
      # Check if individual would default
      if self.env.state.will_default:
        self.fp_obs[group_id] += 1
      else:
        self.tp_obs[group_id] += 1
        self.quali_hist[self.timestep, group_id] = 1
      
    elif action == 0: # CONSIDER THAT ALWAYS IS DEFAULT
      self.tn_obs[group_id] += 1

    t0 = max(0, self.timestep - 50)
    self.quali_obs = np.divide(
      self.quali_hist[t0:self.timestep+1, :].sum(axis=0),
      self.group_hist[t0:self.timestep+1, :].sum(axis=0),
      out=np.zeros_like(self.quali_hist[0]),
      where=self.group_count != 0
    )
    next_x = self.next_x_given_action(action)

    g_r = self.QUAL_CHANGE(curr_x, next_x)
    # qualification gain according to baseline policy pi_0 that always denies
    g_r_pi0, action_pi0, next_x_pi0 = self.get_g_pi0(curr_x, group_id)

    self.tpr = self.compute_tpr(tp=self.tp,
                                fn=self.fn)
    self.old_bank_cash = self.env.state.bank_cash
    self.tpr_obs = self.compute_tpr(tp=self.tp_obs, fn=self.fn_obs)
    self.acc = self.compute_acc(tp=self.tp, fn=self.fn, tn=self.tn, fp=self.fp)
    self.acc_obs = self.compute_acc(tp=self.tp_obs, fn=self.fn_obs, tn=self.tn_obs, fp=self.fp_obs)

    # Update delta terms (for A-PPO)
    self.delta = np.abs(self.tpr[0] - self.tpr[1])
    self.delta_delta = self.delta - old_delta

    # ------------------- update population and distribution -----------------
    old_dist = deepcopy(self.dist)

    state_feats = deepcopy(self.env.state.applicant_features)
    state_default = deepcopy(self.env.state.will_default)

    # Update population and history
    if len(self.population) == self.WINDOW:
      old_id, old_feats, old_default, old_action = self.population.popleft()
      self.history[old_id] -= old_feats

      if old_action == 1:
        if old_default:
          self.fp[old_id] -= 1
        else:
          self.tp[old_id] -= 1
      elif old_action == 0:
        if old_default:
          self.tn[old_id] -= 1
        else:
          self.fn[old_id] -= 1

    self.population.append((group_id, state_feats, state_default, action))

    self.history[group_id] = self.history[group_id] + state_feats

    # Update dist
    if self.is_bigd_fppo:
      self.dist = wloss(torch.tensor(self.history[0]).view(-1, 1), torch.tensor(self.history[1]).view(-1, 1)).item()
      self.dist_dist = self.dist - old_dist
    # -------------------------------------------------------------------------
    obs, _, done, info = self.env.step(action)

    r = self.reward_fn(old_bank_cash=self.old_bank_cash,
                       bank_cash=self.env.state.bank_cash,
                       tpr=self.tpr_obs if self.ONLY_OBSERVATION else self.tpr,
                       zeta0=self.ZETA_0,
                       zeta1=self.ZETA_1)

    self.timestep += 1
    if self.timestep >= self.ep_timesteps:
      done = True

    obs = self.process_observation(obs)

    # ---------------------------------------------
    return obs, r, done, info, g_r, g_r_pi0, action_pi0, next_x_pi0