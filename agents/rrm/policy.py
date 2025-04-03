
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
)
from stable_baselines3.common.type_aliases import Schedule


class ActorPolicy(BasePolicy):
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

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(ActorPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        self._build(lr_schedule)


    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        
        mlp = []
        last_layer = self.features_dim
        for layer_dim in self.net_arch:
            mlp.append(nn.Linear(
                last_layer, layer_dim
            ))
            mlp.append(self.activation_fn())
            last_layer = layer_dim
        mlp.append(nn.Linear(last_layer, 1))
        mlp.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*mlp).to(self.device)
        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Forward pass in the networks (actor)

        :param obs: Observation
        :return: action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        pred = self.mlp(features)
        return pred

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        features = self.extract_features(observation)
        pred = self.mlp(features)
        action = (pred > 0.5).to(th.long).view(-1)
        return action

register_policy("ActorPolicy", ActorPolicy)