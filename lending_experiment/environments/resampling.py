import attr
from copy import deepcopy
import numpy as np
import pickle as pkl

import gym
from gym import spaces
from time import time


class ResamplingEnv(gym.Env):
    """
    Resampling environment to facilitate calculating metrics along individuals.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        n_groups: int = 2,
        cost: float = 0.8,
        n_applicants: int = 10000,
        n_features: int = 10,
        utility_method: str = "accuracy",
        delta_method: str = "full",
        seed=None,
    ):
        assert utility_method in ["accuracy", "qualification", "tpr"]
        assert delta_method in ["full", "imputation", "accepted"]
        self.n_groups = n_groups
        self.cost = cost
        self.n_applicants = n_applicants
        self.n_features = n_features
        self.utility_method = utility_method
        self.delta_method = delta_method
        self.action_space = spaces.Discrete(2)

        resource_space = spaces.Box(
            low=0.0,
            high=100_000,
            dtype=np.float64,
            shape=(),
        )
        applicant_space = spaces.Box(
            low=0.0, high=1.0, dtype=np.float32, shape=(self.n_features,)
        )
        group_space = spaces.MultiBinary(n=self.n_groups)

        self.observable_state_vars = {
            "resource": resource_space,
            "applicant_features": applicant_space,
            "group": group_space,
        }
        self.observation_space = spaces.Dict(self.observable_state_vars)
        self.init_data = dict(
            [
                (col, np.zeros(self.n_applicants, dtype=np.float32))
                for col in [
                    "group",
                    "features",
                    "label",
                    "action",
                    "pred",
                ]
            ]
        )
        self.init_data["features"] = np.zeros(
            (self.n_applicants, self.n_features), dtype=np.float32
        )
        self.init_data["group"] = self.init_data["group"].astype(np.int32)
        self.data = deepcopy(self.init_data)
        self.timestep = 0

        self.get_label_pred = lambda x, g: 0
        self.get_action = lambda x, g: 0
        self.get_action_prob = lambda x, g: 0
        self.get_action_prob_list = lambda x, g: 0
        self.get_action_prob_batch = lambda x, g: 0
        self.get_pred_batch = lambda x, g: 0
        self.delta = 0
        self.delta_obs = 0
        self.error_rejected = [0, 0]
        self.utility_values = [0, 0]
        self.utility_values_obs = [0, 0]
        self.seed(seed)
        self._load_data()
        self._state_init()

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def _load_data(self):
        """This function should fill the entries in self.init_data"""
        return

    def _state_init(self):
        self.resource = 1_000
        self.sample_applicant()

    def sample_applicant(self):
        selected = np.random.choice(self.n_applicants, size=1)[0]
        self.idx = selected
        return

    def reset(self):
        """Resets the environment."""
        self.timestep = 0
        self.data = deepcopy(self.init_data)
        self._state_init()
        self.compute_disparity()
        return self._get_observable_state()

    def update_models(self):
        for idx in range(self.n_applicants):
            group = self.init_data["group"][idx]
            features = self.init_data["features"][idx]
            label = self.init_data["label"][idx]
            pred = self.get_label_pred(features, group)
            action = self.get_action(features, group)
            self.init_data["pred"][idx] = pred
            self.init_data["action"][idx] = action

    def compute_disparity(self):
        # calculate real disparity
        for i in range(self.n_groups):
            labels = self.data["label"][self.data["group"] == i]
            actions = self.data["action"][self.data["group"] == i]
            self.utility_values[i] = self.compute_utility(actions, labels)

        self.delta = self.utility_values[1] - self.utility_values[0]
        old_delta = self.delta_obs
        if self.delta_method == "full":
            self.delta_obs = self.delta
        elif self.delta_method == "accepted":
            for i in range(self.n_groups):
                accepted_group = (self.data["group"] == i) & (self.data["action"] == 1)
                if accepted_group.sum() > 0:
                    self.utility_values_obs[i] = self.compute_utility(
                        self.data["action"][accepted_group],
                        self.data["label"][accepted_group],
                    )
                else:
                    self.utility_values_obs[i] = 0
        elif self.delta_method == "imputation":
            accept_rate = np.zeros(self.n_groups)
            pred_rate = np.zeros(self.n_groups)
            for i in range(self.n_groups):
                labels = self.data["label"][self.data["group"] == i]
                actions = self.data["action"][self.data["group"] == i]
                preds = self.data["pred"][self.data["group"] == i]
                imputation = actions * labels + (1 - actions) * preds
                self.utility_values_obs[i] = self.compute_utility(actions, imputation)

                #### calculate delta_pred
                accept_rate[i] = actions.mean()
                pred_rate[i] = imputation.mean()
                self.error_rejected[i] = (
                    (preds - labels)[actions == 0].mean()
                    if (actions == 0).sum() > 0
                    else 0
                )

        self.delta_obs = self.utility_values_obs[1] - self.utility_values_obs[0]
        self.delta_delta = abs(self.delta_obs) - abs(old_delta)

        #### calculate delta_pred
        if self.delta_method == "imputation":
            if self.utility_method in ["qualification", "accuracy"]:
                self.delta_pred_real = self.error_rejected * (1 - accept_rate)
            else:
                self.delta_pred = self.error_rejected * (1 - accept_rate) / pred_rate
            self.delta_pred_real = self.delta_pred_real[1] - self.delta_pred_real[0]

    def compute_utility(self, action, label):
        if self.utility_method == "accuracy":
            return (action == label).mean()
        elif self.utility_method == "qualification":
            return label.mean()
        elif self.utility_method == "tpr":
            return action[label == 1].mean() if (label == 1).sum() > 0 else 0

    def _get_observable_state(self):
        group = np.zeros(self.n_groups, dtype=np.float32)
        group[int(self.data["group"][self.idx])] = 1
        return {
            "resource": np.array(self.resource),
            "applicant_features": np.array(self.data["features"][self.idx]),
            "group": group,
        }

    def _is_done(self):
        return self.resource <= 0 or self.timestep >= 10_000

    def step(self, action):
        self.timestep += 1
        old_resource = self.resource
        label = self.data["label"][self.idx]
        pred = self.data["pred"][self.idx]
        self.update_resource(action, label)
        self.compute_disparity()
        self.update_applicant(self.idx, action)
        self.sample_applicant()
        observation = self._get_observable_state()
        reward = self.resource - old_resource
        return observation, reward, self._is_done(), {}

    def update_resource(self, action, label):
        if action == 0:  # reject
            return
        self.resource += label - self.cost
        self.resource = max(0, self.resource)

    def update_applicant(self, idx, action):
        # Implement logic to update the applicant based on the action taken
        features = self.data["features"][idx]
        label = self.data["label"][idx]
        features = self.update_features(features, action, label)
        self.data["features"][idx] = features

        group = self.data["group"][idx]
        label = self.get_label(features, group)
        pred = self.get_label_pred(features, group)
        action = self.get_action(features, group)
        self.data["label"][idx] = label
        self.data["pred"][idx] = pred
        self.data["action"][idx] = action

    def update_features(self, features, action, label):
        return features


class LendingEnv(ResamplingEnv):
    """
    Environment for lending experiments.
    """

    def __init__(
        self,
        cost: float = 0.8,
        n_applicants: int = 10000,
        utility_method: str = "accuracy",
        delta_method: str = "full",
        distributions: str = "fico",
        seed=None,
    ):
        assert distributions in [
            "fico",
            "fico_equal",
            "fico_hard",
            "fico_fast",
            "fico_no_decay",
            "fico_test",
            "setting1",
            "setting2",
        ]
        self.n_groups = 2
        self.cost = cost
        self.n_applicants = n_applicants
        self.n_features = 10
        if "setting" in distributions:
            self.n_features = 14
        self.utility_method = utility_method
        self.delta_method = delta_method
        self.distributions = distributions
        super().__init__(
            n_groups=self.n_groups,
            cost=self.cost,
            n_applicants=self.n_applicants,
            n_features=self.n_features,
            utility_method=self.utility_method,
            delta_method=self.delta_method,
            seed=seed,
        )

    def _load_probs(self):
        shift_probs = [
            [0, 1, 0],
            [0, 1, 0],
        ]
        if self.distributions == "fico":
            with open("data/fico.pkl", "rb") as f:
                data = pkl.load(f)
            group_probs = data["group_likelihoods"]
            cluster_probs = data["cluster_probabilities"]
            success_probs = data["success_probabilities"]
        elif (
            self.distributions == "fico_equal" or self.distributions == "fico_no_decay"
        ):
            group_probs = [0.5, 0.5]
            with open("data/fico.pkl", "rb") as f:
                data = pkl.load(f)
            cluster_probs = data["cluster_probabilities"]
            success_probs = data["success_probabilities"]
        elif self.distributions == "fico_test":
            group_probs = [0.5, 0.5]
            with open("data/fico.pkl", "rb") as f:
                data = pkl.load(f)
            cluster_probs = data["cluster_probabilities"]
            success_probs = data["success_probabilities"]
            self.cost = 0.5
        elif self.distributions == "fico_hard":
            group_probs = [0.5, 0.5]
            with open("data/fico.pkl", "rb") as f:
                data = pkl.load(f)
            cluster_probs = data["cluster_probabilities"]
            success_probs = data["success_probabilities"]
            shift_probs = [
                [0.25, 0.7, 0.05],
                [0.05, 0.85, 0.1],
            ]
        elif self.distributions == "setting1":
            group_probs = [0.5, 0.5]
            cluster_probs = [
                [
                    0.0,
                    0.0,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.1,
                    0.1,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.0,
                    0.0,
                ],
                [
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.1,
                    0.1,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
            success_probs = [
                0.773,
                0.804,
                0.833,
                0.857,
                0.879,
                0.898,
                0.914,
                0.928,
                0.939,
                0.949,
                0.958,
                0.965,
                0.970,
                0.975,
            ]
            success_probs = [success_probs, success_probs]
        elif self.distributions == "setting2":
            group_probs = [0.5, 0.5]
            cluster_probs = [
                [
                    0.0,
                    0.0,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.1,
                    0.1,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.0,
                    0.0,
                ],
                [
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.1,
                    0.1,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
            success_probs = [
                0.506,
                0.594,
                0.677,
                0.750,
                0.812,
                0.861,
                0.898,
                0.927,
                0.948,
                0.963,
                0.974,
                0.982,
                0.987,
                0.991,
            ]
            success_probs = [success_probs, success_probs]

        return group_probs, cluster_probs, success_probs, shift_probs

    def _load_data(self):
        groups_probs, cluster_probs, success_probs, shift_probs = self._load_probs()

        def sample_label(x, g):
            # if x is not an scalar, get the argmax
            if not np.isscalar(x):
                x = np.argmax(x)
            y = np.random.binomial(n=1, p=success_probs[g][x])
            shift_prob = shift_probs[g]
            options = [0, y, 1]
            y = np.random.choice(options, p=shift_prob)
            return y

        self.get_label = sample_label

        num_groups = len(groups_probs)
        num_features = len(cluster_probs[0])

        for i in range(self.n_applicants):
            g = np.random.choice(num_groups, p=groups_probs)
            x = np.random.choice(num_features, p=cluster_probs[g])
            label = self.get_label(x, g)
            # one hot encode group and x
            features = np.zeros(num_features, dtype=np.float32)
            features[x] = 1
            group = np.zeros(num_groups, dtype=np.float32)
            group[g] = 1
            pred = self.get_label_pred(features, group)
            action = self.get_action(features, group)
            self.init_data["group"][i] = g
            self.init_data["features"][i] = features
            self.init_data["label"][i] = label
            self.init_data["pred"][i] = pred
            self.init_data["action"][i] = action

    def update_features(self, features, action, label):
        if action == 0:
            return features

        score = np.argmax(features)
        num_features = len(features)
        if label == 1:
            new_score = min(score + 1, num_features - 1)
        elif label == 0:
            new_score = max(score - 1, 0)

        if self.distributions == "fico_no_decay":
            new_score = max(score, new_score)
            if np.random.rand() < 1:
                new_score = min(score + 1, num_features - 1)

        features[score] = 0
        features[new_score] = 1
        return features


class EnemEnv(ResamplingEnv):
    """
    Environment for school admission experiments.
    """

    def __init__(
        self,
        cost: float = 0.5,
        n_applicants: int = 4_000,
        utility_method: str = "accuracy",
        delta_method: str = "full",
        seed=None,
    ):
        super().__init__(
            n_groups=2,
            n_features=128,
            cost=cost,
            n_applicants=n_applicants,
            utility_method=utility_method,
            delta_method=delta_method,
            seed=seed,
        )

    def _load_data(self):
        with open("data/enem_pool.pkl", "rb") as f:
            pool = pkl.load(f)

        with open("data/enem_model.pkl", "rb") as f:
            self.model = pkl.load(f)

        def sample_label(x, g):
            p = self.model.predict_proba(x[:-1].reshape(1, -1))[0, 1]
            gain = x[-1] * 1
            p = min(gain + p, 1)
            return 1 if np.random.rand() < p else 0

        self.get_label = sample_label

        for i in range(self.n_applicants):
            group = pool[i]["group"]
            features = pool[i]["features"]

            label = self.get_label(features, group)
            pred = self.get_label_pred(features, group)
            action = self.get_action(features, group)
            self.init_data["group"][i] = group
            self.init_data["features"][i] = features
            self.init_data["label"][i] = label
            self.init_data["pred"][i] = pred
            self.init_data["action"][i] = action

    def update_features(self, features, action, label):
        if action == 1 or label == 1:
            features[-1] = 1

        group = features[0]
        age_groups = 2
        age = np.argmax(features[1 : 1 + age_groups])
        new_age = min(age + 1, age_groups - 1)

        new_features = features.copy()
        new_features[1 + age] = 0
        new_features[1 + new_age] = 1
        return new_features

    def _is_done(self):
        return self.resource <= 0 or self.timestep >= 2048
