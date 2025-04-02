import gym
import numpy as np
import torch
from gym import spaces
from copy import deepcopy
from collections import deque

from environments.distributions import Bernoulli



class RPMEnvWrapper(gym.Wrapper):
  def __init__(self,
               env,
               reward_fn,
               config_params=None,
               is_eval=False,
    ):
    super(RPMEnvWrapper, self).__init__(env)

    self.DRIFT_PROBS = config_params.DRIFT_PROBS
    self.USE_CREDIT_DRIFT = config_params.USE_CREDIT_DRIFT

    self.pi_0 = Bernoulli(config_params.PI0_PROB_A1)

    self.DIST_EST = config_params.DIST_EST

    
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

    
    if is_eval:
      self.ep_timesteps = config_params.EVAL_EP_TIMESTEPS
    else:
      self.ep_timesteps = config_params.EP_TIMESTEPS
      

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

    # ----------------------------------- add history and population ----------------------------
    self.history = np.zeros((self.env.state.params.num_groups, self.env.observation_space['applicant_features'].shape[0]))
    self.population = deque(maxlen=self.WINDOW)
    self.dist = 0
    self.dist_dist = 0
    # --------------------------------------------------------------------------------------------

    return self.process_observation(self.env.reset())
  

  def step(self, action):
    old_delta = self.delta

    curr_x = np.argmax(self.env.state.applicant_features)
    # Update instance variables before we step the environment
    group_id = np.argmax(self.env.state.group)
    if action == 1:
      # Check if individual would default
      if self.env.state.y == 0:
        self.fp[group_id] += 1
      else:
        self.tp[group_id] += 1
      
    elif action == 0:
      if self.env.state.y == 0:
        self.tn[group_id] += 1
      else:
        self.fn[group_id] += 1


    
    self.old_bank_cash = self.env.state.bank_cash

    # Update delta terms (for A-PPO)
    self.delta = np.abs(self.tpr[0] - self.tpr[1])
    self.delta_delta = self.delta - old_delta

    # ------------------- update population and distribution -----------------

    state_feats = deepcopy(self.env.state.applicant_features)
    state_default = deepcopy(self.env.state.y == 0)

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

    # -------------------------------------------------------------------------
    obs, _, done, info = self.env.step(action)

    #r = self.reward_fn(old_bank_cash=self.old_bank_cash,
    #                   bank_cash=self.env.state.bank_cash,
    #                   tpr=self.tpr_obs if self.ONLY_OBSERVATION else self.tpr,
    #                   zeta0=self.ZETA_0,
    #                   zeta1=self.ZETA_1)
    r = 0

    self.timestep += 1
    if self.timestep >= self.ep_timesteps:
      done = True

    obs = self.process_observation(obs)

    # ---------------------------------------------
    return obs, r, done, info