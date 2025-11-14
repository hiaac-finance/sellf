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
        history_size : int = 10_000,
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
        self.history_size = n_applicants #history_size
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

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def _load_data(self):
        """This function should fill the entries in self.init_data"""
        return

    def sample_applicant(self):
        selected = np.random.choice(self.n_applicants, size=1)[0]
        self.idx = selected
        return

    def reset(self):
        """Resets the environment."""
        self.timestep = 0
        self.resource = 1_000
        self.data = deepcopy(self.init_data)
        self.start_history()
        self.sample_applicant()
        self.compute_disparity()
        return self._get_observable_state()

    def start_history(self):
        self.history = dict(
            [
                (col, np.zeros(self.history_size, dtype=np.float32))
                for col in ["group", "label", "action", "pred"]
            ]
        )
        self.history_pos = 0
        # rum some iterations without updating features
        for _ in range(self.history_size):
            # randomly sample an applicant
            idx = np.random.choice(self.n_applicants, size=1)[0]
            group = self.init_data["group"][idx]
            features = self.init_data["features"][idx]
            label = self.get_label(features, group)
            pred = self.get_label_pred(features, group)
            action = self.get_action(features, group)

            self.history["group"][self.history_pos] = group
            self.history["label"][self.history_pos] = label
            self.history["action"][self.history_pos] = action
            self.history["pred"][self.history_pos] = pred
            self.history_pos += 1

        self.history_pos = 0

    def compute_disparity(self):
        # calculate real disparity
        groups = self.history["group"]
        labels = self.history["label"]
        actions = self.history["action"]
        preds = self.history["pred"]
        for i in range(self.n_groups):
            labels_g = labels[groups == i]
            actions_g = actions[groups == i]
            self.utility_values[i] = self.compute_utility(actions_g, labels_g)

        self.delta = self.utility_values[1] - self.utility_values[0]
        old_delta = self.delta_obs
        if self.delta_method == "full":
            self.utility_values_obs = self.utility_values
        elif self.delta_method == "accepted":
            for i in range(self.n_groups):
                accepted_group = (groups == i) & (actions == 1)
                if accepted_group.sum() > 0:
                    self.utility_values_obs[i] = self.compute_utility(
                        actions[accepted_group],
                        labels[accepted_group],
                    )
                else:
                    self.utility_values_obs[i] = 0
        elif self.delta_method == "imputation":
            accept_rate = np.zeros(self.n_groups)
            pred_rate = np.zeros(self.n_groups)
            for i in range(self.n_groups):
                labels_g = labels[groups == i]
                actions_g = actions[groups == i]
                preds_g = preds[groups == i]
                imputation = actions_g * labels_g + (1 - actions_g) * preds_g
                self.utility_values_obs[i] = self.compute_utility(actions_g, imputation)

                #### calculate delta_pred
                accept_rate[i] = actions_g.mean()
                pred_rate[i] = imputation.mean()
                self.error_rejected[i] = (
                    (preds_g - labels_g)[actions_g == 0].mean()
                    if (actions_g == 0).sum() > 0
                    else 0
                )

        self.delta_obs = self.utility_values_obs[1] - self.utility_values_obs[0]
        self.delta_delta = abs(self.delta_obs) - abs(old_delta)

        #### calculate delta_pred
        if self.delta_method == "imputation":
            if self.utility_method in ["qualification", "accuracy"]:
                self.delta_pred_real = self.error_rejected * (1 - accept_rate)
            else:
                self.delta_pred_real = (
                    self.error_rejected * (1 - accept_rate) / pred_rate
                )
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
        group = self.data["group"][self.idx]
        features = self.data["features"][self.idx]
        self.history["group"][self.history_pos] = group

        label = self.get_label(features, group)
        pred = self.get_label_pred(features, group)
        self.update_resource(action, label)


        # different strategies between methods
        if self.utility_method == "qualification":
            self.update_applicant(self.idx, action, label)
            new_features = self.data["features"][self.idx]
            new_label = self.get_label(new_features, group)
            new_pred = self.get_label_pred(new_features, group)
            new_action = self.get_action(new_features, group)
            self.history["label"][self.history_pos] = new_label
            self.history["pred"][self.history_pos] = new_pred
            self.history["action"][self.history_pos] = new_action
        else:
            self.history["label"][self.history_pos] = label
            self.history["pred"][self.history_pos] = pred
            self.history["action"][self.history_pos] = action
            self.update_applicant(self.idx, action, label)

        self.history_pos += 1
        if self.history_pos >= self.history_size:
            self.history_pos = 0

        self.compute_disparity()
        self.sample_applicant()
        observation = self._get_observable_state()
        reward = self.resource - old_resource
        return (
            observation,
            reward,
            self._is_done(),
            {
                "group": group,
                "label": label,
                "delta": self.delta,
                "delta_obs": self.delta_obs,
                "delta_delta": self.delta_delta,
            },
        )

    def update_resource(self, action, label):
        if action == 0:  # reject
            return
        self.resource += label - self.cost
        self.resource = max(0, self.resource)

    def update_applicant(self, idx, action, label):
        # Implement logic to update the applicant based on the action taken
        features = self.data["features"][idx]
        features = self.update_features(features, action, label)
        self.data["features"][idx] = features

        return

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
            "fico_equal",
            "fico_no_decay",
            "fico_easy",
        ]
        self.n_groups = 2
        self.cost = cost
        if distributions == "fico_easy":
            self.cost = 0.5
        self.n_applicants = n_applicants
        self.n_features = 10
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
        elif self.distributions in ["fico_equal", "fico_no_decay", "fico_easy"]:
            group_probs = [0.5, 0.5]
            with open("data/fico.pkl", "rb") as f:
                data = pkl.load(f)
            cluster_probs = data["cluster_probabilities"]
            success_probs = data["success_probabilities"]
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
            # one hot encode group and x
            features = np.zeros(num_features, dtype=np.float32)
            features[x] = 1
            group = np.zeros(num_groups, dtype=np.float32)
            group[g] = 1
            self.init_data["group"][i] = g
            self.init_data["features"][i] = features

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

        features[score] = 0
        features[new_score] = 1
        return features


class EnemEnv(ResamplingEnv):
    """
    Environment for school admission experiments.
    """

    def __init__(
        self,
        cost: float = 0.4,
        n_applicants: int = 4_000,
        utility_method: str = "accuracy",
        delta_method: str = "full",
        seed=None,
    ):
        super().__init__(
            n_groups=2,
            n_features=126,
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
            gain = x[-1] * 0.5
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
            # self.init_data["label"][i] = label
            # self.init_data["pred"][i] = pred
            # self.init_data["action"][i] = action

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


class COMPASEnv(ResamplingEnv):
    """
    Environment for school admission experiments.
    """

    def __init__(
        self,
        cost: float = 0.9,
        n_applicants: int = 4_000,
        utility_method: str = "accuracy",
        delta_method: str = "full",
        seed=None,
    ):
        super().__init__(
            n_groups=2,
            n_features=13,
            cost=cost,
            n_applicants=n_applicants,
            utility_method=utility_method,
            delta_method=delta_method,
            seed=seed,
        )

    def _load_data(self):
        with open("data/compas_pool.pkl", "rb") as f:
            pool = pkl.load(f)

        with open("data/compas_model.pkl", "rb") as f:
            self.model = pkl.load(f)

        def sample_label(x, g):
            age_cat = np.argmax(x[0:5])
            priors_count_cat = np.argmax(x[5:13])

            p = 1 - self.model.get((g, age_cat, priors_count_cat), 0.0)
            return 1 if np.random.rand() < p else 0

        self.get_label = sample_label

        for i in range(self.n_applicants):
            group = pool[i]["group"]
            features = pool[i]["features"]

            self.init_data["group"][i] = group
            self.init_data["features"][i] = features
            # self.init_data["label"][i] = label
            # self.init_data["pred"][i] = pred
            # self.init_data["action"][i] = action

    def update_features(self, features, action, label):
        if action == 0: # jail
            return features
        
        age_cat = np.argmax(features[0:5])
        priors_count_cat = np.argmax(features[5:13])

        # if bail
        if label == 1: # reicividism
            new_priors_count_cat = min(priors_count_cat + 1, 7)
        else:
            new_priors_count_cat = priors_count_cat

        features[5 + priors_count_cat] = 0
        features[5 + new_priors_count_cat] = 1
        return features

    def _is_done(self):
        return self.resource <= 0 or self.timestep >= 2048
