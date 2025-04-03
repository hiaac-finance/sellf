
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
)
from stable_baselines3.common.type_aliases import Schedule


class ActorPolicy(ActorCriticPolicy):
    """
    Policy class for actor algorithms (has a policy).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        super(ActorPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init, 
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Forward pass in the actor network.

        :param obs: Observation
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        mean_actions = self.action_net(latent_pi)
        return mean_actions

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        # Preprocess the observation if needed
        features = self.extract_features(observation)

        latent_pi, _ = self.mlp_extractor(features)
        mean_actions = self.action_net(latent_pi)
        actions = th.argmax(mean_actions, dim=1)
        return actions

    def prob_loan(self, obs: th.Tensor) -> th.Tensor:
        """
        Get probability of receiving a loan according to the current policy,
        given the observations.

        :param obs:
        :return: probability of loans
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        mean_actions = self.action_net(latent_pi)
        # Get the probability of receiving a loan
        prob = th.sigmoid(mean_actions)
        return prob[:, 1]