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

        self.get_label = lambda feat, g : 0
        self.get_action = lambda feat, g : 0
        self.get_label_pred = lambda feat, g : 0
        self.get_action_prob = lambda feat, g : 0
        self.get_action_prob_list = lambda feat, g : [0.5]

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
                (col, np.zeros((self.n_applicants, self.n_groups), dtype=np.float32))
                for col in [
                    "group",
                    "utility_real",
                    "active_real",
                    "utility",
                    "active",
                    "action",
                    "action_prob",
                    "pred",
                    "error",
                ]
            ]
        )
        self.pi_history = []
        self.group_masks = None
        self.hist_max_len = 20
        self.data = copy.deepcopy(self.init_data)
        self.init_pool = []
        self.pool = []
        self.pool_accepted = [[], []]
        self.load_pool()
        self._state_init()

    def _state_init(self):
        self.resource = 1_000
        self.delta = 0
        self.delta_real = 0
        self.delta_delta = 0
        self.delta_pred = 0
        self.delta_var = 0
        self.sample_applicant()

    def reset(self):
        """Resets the environment."""
        self.timestep = 0
        self.pool = copy.deepcopy(self.init_pool)
        self.data = copy.deepcopy(self.init_data)
        self._state_init()
        self.compute_disparity()
        return self._get_observable_state()

    def _is_done(self):
        return self.resource <= 0 or self.timestep >= 10_000

    def _step_impl(self, action):
        self.timestep += 1
        self.update_resource(action)
        idx = self.idx
        features = self.pool[idx]["features"]
        label = self.pool[idx]["label"]
        pred = self.pool[idx]["pred"]
        group_idx = np.argmax(self.pool[idx]["group"])
        if action == 1 and len(self.pool_accepted[group_idx]) < 20_000:
            self.pool_accepted[group_idx].append({
                "features" : features,
                "label" : label,
                "pred" : pred,
                "error" : 0 if pred == label else 1,
                "one_minus_pi" : np.prod([1 - x for x in self.get_action_prob_list(features, group_idx)]),
                "one_minus_pi_last" : self.data["action_prob"][idx, group_idx],
            })

        self.update_utility(
            idx=idx,
            label=label,
            pred=pred,
            group_idx=group_idx,
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
        features = self.pool[idx]["features"]
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
            self.init_data["utility_real"][idx, group_idx] = utility_value
            self.init_data["active_real"][idx, group_idx] = active
            self.init_data["action"][idx, group_idx] = action
            self.init_data["action_prob"][idx, group_idx] = self.get_action_prob(features, group_idx)
            self.init_data["error"][idx, group_idx] = pred - label
            self.init_data["pred"][idx, group_idx] = pred
        else:
            self.data["utility_real"][idx, group_idx] = utility_value
            self.data["active_real"][idx, group_idx] = active
            self.data["action"][idx, group_idx] = action
            self.data["action_prob"][idx, group_idx] = self.get_action_prob(features, group_idx)
            self.data["error"][idx, group_idx] = pred - label
            self.data["pred"][idx, group_idx] = pred

        if self.delta_method == "full":
            pass  # Full delta is already calculated
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
            self.init_data["utility"][idx, group_idx] = utility_value
            self.init_data["active"][idx, group_idx] = active
        else:
            self.data["utility"][idx, group_idx] = utility_value
            self.data["active"][idx, group_idx] = active

    def compute_disparity(self):
        # calculate disparity using self.data
        # multiply utility by active
        self.data["utility_real"] *= self.data["active_real"]
        self.data["utility"] *= self.data["active"]

        # calculate group counts from active
        group_counts_real = np.sum(self.data["active_real"], axis=0)
        group_counts = np.sum(self.data["active"], axis=0)

        # calculate utility
        utility_real = np.sum(self.data["utility_real"], axis=0)
        utility = np.sum(self.data["utility"], axis=0)

        utility_real = np.true_divide(
            utility_real,
            group_counts_real,
            where=group_counts_real != 0,
            out=np.zeros_like(utility_real),
        )
        utility = np.true_divide(
            utility, group_counts, where=group_counts != 0, out=np.zeros_like(utility)
        )

        cur_delta = self.delta
        self.delta_real = np.max(utility_real) - np.min(utility_real)
        self.delta = np.max(utility) - np.min(utility)
        self.delta_delta = self.delta - cur_delta

        # calculate accept rate
        group_counts = [0, 0]
        for i in range(2):
            group_counts[i] = np.sum(self.data["group"][:, i] == 1)
        accept = np.sum(self.data["action"], axis=0)
        accept_rate = np.true_divide(
            accept,
            group_counts,
            where=group_counts != 0,
            out=np.zeros_like(accept),
        )

        pred_mean = np.sum(self.data["pred"], axis=0)
        pred_mean = np.true_divide(
            pred_mean,
            group_counts,
            where=group_counts != 0,
            out=np.zeros_like(pred_mean),
        )

        if len(self.pool_accepted[0]) == 0 or len(self.pool_accepted[1]) == 0:
            self.delta_pred = 0
            self.delta_var = 0
            self.error_rate = [0 , 0]
            self.prob_dist = [0, 0]
            self.var_gap = [0, 0]
            return

        error_rate = [0, 0]
        dist_term = [0, 0]
        error_bound = [0, 0]
        delta_pred = [0, 0]
        for i in range(2):
            error_rate[i] = np.mean([app["error"] for app in self.pool_accepted[i]])
            one_minus_pi_last = np.array([app["one_minus_pi_last"] for app in self.pool_accepted[i]])
            one_minus_pi = np.array([1 - app["one_minus_pi"] for app in self.pool_accepted[i]])
            q = np.mean(one_minus_pi)
            r = 1 - accept_rate[i]
            w = (q / r) * (one_minus_pi_last / one_minus_pi)
            # check if w is nan
            dist_term[i] = np.sqrt(np.mean((w - 1) ** 2))
            error_bound[i] = error_rate[i] + dist_term[i]

            if self.utility_method in ["qualification", "accuracy"]:
                delta_pred[i] = error_bound[i] * (1 - accept_rate[i])
            else:
                delta_pred[i] = error_bound[i] * (1 - accept_rate[i]) / pred_mean[i]
            
            

        self.delta_pred = np.max(delta_pred) - np.min(delta_pred)
        self.delta_var = 0
        self.error_rate = error_rate
        self.prob_dist = dist_term
        self.var_gap = [0, 0]

    def sample_applicant(self):
        selected = np.random.choice(len(self.pool), size=1)[0]
        self.idx = selected
        return

    def update_models(self):
        for k, v in self.init_data.items():
            self.init_data[k] = np.zeros_like(v)
        for idx in range(len(self.pool)):
            features = self.pool[idx]["features"]
            group_idx = self.pool[idx]["group"].argmax()
            action = self.get_action(features, group_idx)
            pred = self.get_label_pred(features, group_idx)
            label = self.pool[idx]["label"]
            self.pool[idx]["pred"] = pred
            self.init_data["group"][idx, group_idx] = 1
            self.update_utility(idx, label, pred, group_idx, action, init=True)
        self.data = self.init_data.copy()

        # update accepted pool
        for i in range(2):
            for j, applicant in enumerate(self.pool_accepted[i]):
                features = applicant["features"]
                group = i
                prob = self.get_action_prob(features, group)
                self.pool_accepted[i][j]["one_minus_pi"] *= (1 - prob)

        self.compute_disparity()


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
        group_ratios: str = "data",
    ):
        assert group_ratios in ["data", "equal"]
        self.n_groups = 2
        self.cost = cost
        self.n_applicants = n_applicants
        self.n_features = 10
        self.utility_method = utility_method
        self.delta_method = delta_method
        self.group_ratios = group_ratios
        super().__init__(
            n_groups=self.n_groups,
            cost=self.cost,
            n_applicants=self.n_applicants,
            n_features=self.n_features,
            utility_method=self.utility_method,
            delta_method=self.delta_method,
        )

    def load_pool(self):
        with open("data/fico.pkl", "rb") as f:
            data = pkl.load(f)
        if self.group_ratios == "data":
            groups_probs = data["group_likelihoods"]
        else:
            groups_probs = [0.5, 0.5]
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
            pred = self.get_label_pred(features, group)
            action = self.get_action(features, group)
            self.init_pool.append(
                {
                    "features": features,
                    "group": group,
                    "label": y,
                    "pred": pred,
                    "action": action,
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

        pred = self.get_label_pred(self.pool[idx]["features"], group)
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

        pred = self.get_label_pred(new_features, group)
        self.pool[idx]["pred"] = pred
