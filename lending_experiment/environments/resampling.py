import attr
import copy
import numpy as np
import pickle as pkl

import gym
from gym import spaces


class ResamplingEnv(gym.Env):
    """
    Resampling environment to facilitate calculating metrics along individuals.
    """

    metadata = {"render.modes": ["human"]}
    group_membership_var = "group"

    def __init__(
        self,
        n_groups: int = 2,
        cost: float = 0.8,
        n_applicants: int = 10000,
        n_features: int = 10,
        utility_method: str = "accuracy",
        delta_method: str = "full",
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

        self.get_label = lambda x: 0
        self.get_action = lambda x: 0
        self.get_label_pred = lambda x: 0

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

        self.init_utility_matrix = np.zeros(
            (self.n_applicants, self.n_groups), dtype=np.float32
        )
        self.init_active_matrix = np.zeros(
            (self.n_applicants, self.n_groups), dtype=np.float32
        )
        self.init_action_matrix = np.zeros(
            (self.n_applicants, self.n_groups),
            dtype=np.float32,
        )
        self.utility_matrix = np.zeros(
            (self.n_applicants, self.n_groups), dtype=np.float32
        )
        self.active_matrix = np.zeros(
            (self.n_applicants, self.n_groups), dtype=np.float32
        )
        self.action_matrix = np.zeros(
            (self.n_applicants, self.n_groups),
            dtype=np.float32,
        )
        self.init_utility_real_matrix = np.zeros(
            (self.n_applicants, self.n_groups), dtype=np.float32
        )
        self.init_active_real_matrix = np.zeros(
            (self.n_applicants, self.n_groups), dtype=np.float32
        )
        self.utility_real_matrix = np.zeros(
            (self.n_applicants, self.n_groups), dtype=np.float32
        )
        self.active_real_matrix = np.zeros(
            (self.n_applicants, self.n_groups), dtype=np.float32
        )
        self.init_pool = []
        self.pool = []
        self.load_pool()
        self._state_init()

    def _state_init(self):
        self.resource = 1_000
        self.delta = 0
        self.delta_real = 0
        self.delta_delta = 0
        self.sample_applicant()

    def reset(self):
        """Resets the environment."""
        self.timestep = 0
        self.pool = copy.deepcopy(self.init_pool)
        self.utility_matrix = copy.deepcopy(self.init_utility_matrix)
        self.active_matrix = copy.deepcopy(self.init_active_matrix)
        self.action_matrix = copy.deepcopy(self.init_action_matrix)
        self.utility_real_matrix = copy.deepcopy(self.init_utility_real_matrix)
        self.active_real_matrix = copy.deepcopy(self.init_active_real_matrix)
        self._state_init()
        self.compute_disparity()
        return self._get_observable_state()

    def _is_done(self):
        return self.resource <= 0 or self.timestep >= 10_000

    def _step_impl(self, action):
        self.timestep += 1
        self.update_resource(action)
        idx = self.idx
        self.update_utility(
            idx=idx,
            label=self.pool[idx]["label"],
            pred=self.pool[idx]["pred"],
            group_idx=np.argmax(self.pool[idx]["group"]),
            action=action,
            init=False,
        )
        self.compute_disparity()
        self.update_applicant(idx, action)
        self.sample_applicant()

    def _get_observable_state(self):
        return {
            "resource": np.array(self.resource),
            "applicant_features": np.array(self.pool[self.idx]["features"]),
            "group": np.array(self.pool[self.idx]["group"]),
        }

    def step(self, action):
        old_resource = self.resource
        self._step_impl(action)
        observation = self._get_observable_state()
        reward = self.resource - old_resource
        return observation, reward, self._is_done(), {}

    def load_pool(self):
        """Load the pool of applicants."""
        pass

    def update_resource(self, action):
        if action == 0:  # reject
            return
        label = self.pool[self.idx]["label"]
        self.resource += label - self.cost
        self.resource = max(0, self.resource)

    def update_applicant(self, idx, action):
        # Implement logic to update the applicant based on the action taken
        pass

    def update_utility(self, idx, label, pred, group_idx, action, init=False):
        """Update the difference in utility for the current applicant. Also updates the utility in the pool."""
        # First, calculate real utility
        active = 1
        if self.utility_method == "accuracy":
            utility_value = 1 if label == action else 0
        elif self.utility_method == "qualification":
            utility_value = label
        elif self.utility_method == "tpr":
            utility_value = action
            active = 1 if label == 1 else 0

        if init:
            self.init_utility_real_matrix[idx, group_idx] = utility_value
            self.init_active_real_matrix[idx, group_idx] = active
            self.init_action_matrix[idx, group_idx] = action
        else:
            self.utility_real_matrix[idx, group_idx] = utility_value
            self.active_real_matrix[idx, group_idx] = active
            self.action_matrix[idx, group_idx] = action

        if self.delta_method == "full":
            pass  # Full delta is already calculated in the utility_real_matrix
        elif self.delta_method == "imputation":
            label = label if action == 1 else pred
            if self.utility_method == "accuracy":
                utility_value = 1 if label == action else 0
                active = 1
            elif self.utility_method == "qualification":
                utility_value = label
                active = 1
            elif self.utility_method == "tpr":
                utility_value = action
                active = 1 if label == 1 else 0
        elif self.delta_method == "accepted":
            label = label
            if self.utility_method == "accuracy":
                utility_value = 1 if label == action else 0
                active = 1 if action else 0
            elif self.utility_method == "qualification":
                utility_value = label
                active = 1 if action else 0
            elif self.utility_method == "tpr":
                utility_value = action
                active = 1 if action * label == 1 else 0

        if init:
            self.init_utility_matrix[idx, group_idx] = utility_value
            self.init_active_matrix[idx, group_idx] = active
        else:
            self.utility_matrix[idx, group_idx] = utility_value
            self.active_matrix[idx, group_idx] = active

    def compute_disparity(self):
        # Multiplity utility by active matrix
        self.utility_matrix *= self.active_matrix
        self.utility_real_matrix *= self.active_real_matrix

        # First, calculate real utility
        cur_util = np.sum(self.utility_real_matrix, axis=0)
        group_counts = np.sum(self.active_real_matrix, axis=0)
        cur_util = np.divide(
            cur_util,
            group_counts,
            out=np.zeros_like(cur_util),
            where=group_counts != 0,
        )
        self.delta_real = np.max(cur_util) - np.min(cur_util)

        accept_rate = np.sum(self.action_matrix, axis=0)
        accept_rate = np.divide(
            accept_rate,
            group_counts,
            out=np.zeros_like(accept_rate),
            where=group_counts != 0,
        )

        # Then, calculate utility
        cur_util = np.sum(self.utility_matrix, axis=0)
        group_counts = np.sum(self.active_matrix, axis=0)
        cur_delta = self.delta
        cur_util = np.divide(
            cur_util,
            group_counts,
            out=np.zeros_like(cur_util),
            where=group_counts != 0,
        )
        self.delta = np.max(cur_util) - np.min(cur_util)
        self.delta_delta = self.delta - cur_delta

        #for i in range(2):
        #   #self.error_on_accept[i] = ...
        #   a = accept_rate[i]
        #   r = 1 - a
        #   if self.utility_method in ["qualification", "accuracy"]:
        #       self.pred_error = r * (
        #           self.error_on_accept[i] + np.sqrt(a - a**2) / (a * r)
        #       )
        #   else:
        #       raise Exception("not implemented yet")
        #self.delta_pred = np.max(self.pred_error) - np.min(self.pred_error)

    def sample_applicant(self):
        selected = np.random.choice(len(self.pool), size=1)[0]
        self.idx = selected
        return

    def update_models(self):
        self.init_active_matrix *= 0
        self.init_utility_matrix *= 0
        self.init_active_real_matrix *= 0
        self.init_utility_real_matrix *= 0
        for idx in range(len(self.pool)):
            action = self.get_action(idx)
            pred = self.get_label_pred(idx)
            label = self.pool[idx]["label"]
            group_idx = self.pool[idx]["group"].argmax()
            self.pool[idx]["pred"] = pred
            self.update_utility(idx, label, pred, group_idx, action, init=True)

        self.utility_matrix = copy.deepcopy(self.init_utility_matrix)
        self.active_matrix = copy.deepcopy(self.init_active_matrix)
        self.utility_real_matrix = copy.deepcopy(self.init_utility_real_matrix)
        self.active_real_matrix = copy.deepcopy(self.init_active_real_matrix)

        self.compute_disparity()


class LendingEnv(ResamplingEnv):
    """
    Environment for lending experiments.
    """

    def load_pool(self):
        with open("data/fico.pkl", "rb") as f:
            data = pkl.load(f)
        groups_probs = data["group_likelihoods"]
        cluster_probs = data["cluster_probabilities"]
        success_probs = data["success_probabilities"]

        def sample_label(g, x):
            return np.random.binomial(n=1, p=success_probs[g][x])

        self.get_label = sample_label

        num_groups = len(groups_probs)
        num_features = len(cluster_probs[0])

        for i in range(self.n_applicants):
            g = np.random.choice(num_groups, p=groups_probs)
            x = np.random.choice(num_features, p=cluster_probs[g])
            y = self.get_label(g, x)
            # one hot encode group and x
            features = np.zeros(num_features, dtype=np.float32)
            features[x] = 1
            group = np.zeros(num_groups, dtype=np.float32)
            group[g] = 1

            self.init_pool.append(
                {
                    "features": features,
                    "group": group,
                    "label": y,
                    "pred": None,
                }
            )

        self.pool = copy.deepcopy(self.init_pool)

    def update_applicant(self, idx, action):
        if action == 0:
            return

        applicant = self.pool[idx]
        score = np.argmax(applicant["features"])
        num_features = len(applicant["features"])
        if applicant["label"] == 1:
            new_score = min(score + 1, num_features - 1)
        elif applicant["label"] == 0:
            new_score = max(score - 1, 0)

        self.pool[idx]["features"][score] = 0
        self.pool[idx]["features"][new_score] = 1

        group = np.argmax(applicant["group"])
        y = self.get_label(group, new_score)
        self.pool[idx]["label"] = y

        pred = self.get_label_pred(idx)
        self.pool[idx]["pred"] = pred


class EnemEnv(ResamplingEnv):
    """
    Environment for school admission experiments.
    """

    def load_pool(self):
        with open("data/enem_pool.pkl", "rb") as f:
            self.init_pool = pkl.load(f)

        with open("data/enem_model.pkl", "rb") as f:
            self.model = pkl.load(f)

        def sample_label(x):
            p = self.model.predict_proba(x.reshape(1, -1))[0, 1]
            return 1 if np.random.rand() < p else 0

        self.label_fn = sample_label
        self.pool = copy.deepcopy(self.init_pool)

    def updated_applicant(self, idx, action):
        if action == 1:
            return
        age_groups = 6
        applicant = self.pool[idx]
        features = applicant["features"]
        group = np.argmax(applicant["group"])
        age_features = features[(age_groups * group) : (age_groups * (group + 1))]
        age = np.argmax(age_features)
        new_age = min(age + 1, age_groups - 1)

        new_features = features.copy()
        new_features[int(age_groups * group + age)] = 0
        new_features[int(age_groups * group + new_age)] = 1

        self.pool[idx]["features"] = new_features

        label = self.get_label(group, new_age)
        self.pool[idx]["label"] = label

        pred = self.get_label_pred(self.pool[idx])
        self.pool[idx]["pred"] = pred
